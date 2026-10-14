import json
import logging
import os
import os.path
import traceback

from panoramix.core.arithmetic import is_zero
from panoramix.core.variants import set_free_memory_pointer
from panoramix.matcher import Any, match
from panoramix.utils.helpers import (
    COLOR_GRAY,
    ENDC,
    EasyCopy,
    colorize,
    find_f_list,
    opcode,
    padded_hex,
)
from panoramix.utils.opcode_dict import opcode_dict
from panoramix.utils.signatures import (
    canonical_type,
    fix_input_names,
    get_func_name,
    make_abi,
)
from panoramix.utils.supplement import fetch_sig

logger = logging.getLogger(__name__)

cache_sigs = {
    True: {},
    False: {},
}

LOADER_TIMEOUT = 60


"""

    What the contract does before it gets to a function.

    The functions are decompiled from where the dispatcher jumps to them, but
    what the dispatcher did on the way still holds: the memory it wrote (the
    free memory pointer) and the checks it made - solc checks that there's no
    ether sent before looking at the selector if no function is payable.

    `entry` is that, in order: the memory writes, and ("check", cond, taken,
    other) for every if on the path whose other branch reverts - not the ones
    looking at the selector, the dispatch itself.

"""


# what the instructions before solidity's first mstore do to the stack:
# (popped, pushed) - a library checks its address
STACK_EFFECTS = {
    "address": (0, 1),
    "eq": (2, 1),
    "callvalue": (0, 1),
    "calldatasize": (0, 1),
    "lt": (2, 1),
    "gt": (2, 1),
    "iszero": (1, 1),
}


def sets_free_memory_pointer(parsed_lines):
    """
    Whether the code starts as solidity's does: its first mstore writes a
    number of at least 0x60 at 0x40 - mstore(0x40, 0x80), 0x60 before
    0.4.22, more for what some keep in memory -, the stack before it made
    of pushes, dups and swaps (PUSH1 0x80 DUP1 PUSH1 0x40 MSTORE via IR).
    Vyper's first mstore isn't at 0x40, or of a number there.
    """
    stack = []
    for _, op, param in parsed_lines[:32]:
        if op.startswith("push"):
            stack.append(param or 0)
        elif op.startswith("dup") and len(stack) >= int(op[3:]):
            stack.append(stack[-int(op[3:])])
        elif op.startswith("swap") and len(stack) > int(op[4:]):
            n = int(op[4:])
            stack[-1], stack[-1 - n] = stack[-1 - n], stack[-1]
        elif op == "mstore" and len(stack) >= 2:
            addr, value = stack[-1], stack[-2]
            return addr == 0x40 and type(value) is int and 0x60 <= value < 2**16
        elif op in STACK_EFFECTS and len(stack) >= STACK_EFFECTS[op][0]:
            popped, pushed = STACK_EFFECTS[op]
            stack = stack[: len(stack) - popped] + [None] * pushed
        else:
            return False
    return False


def is_marker(line):
    return type(line) is str or opcode(line) == "jd"


def reverts(trace):
    """True if the trace always reverts (without calling any function)."""
    # (the branches still to look at, rather than a call for each: the ifs
    # can be nested deeper than python's recursion limit)
    todo = [trace]
    while todo:
        trace = [line for line in todo.pop() if not is_marker(line)]
        if not trace:
            return False

        last = trace[-1]
        if opcode(last) == "if":
            todo += [last[3], last[2]]
        elif opcode(last) not in ("revert", "invalid") or any(
            opcode(line) in ("funccall", "if") for line in trace[:-1]
        ):
            return False

    return True


def is_dispatch(cond):
    """
    A condition on the selector, or on calldatasize being at least 4 (solc:
    calldatasize < 4, Vyper: calldatasize > 3).
    """
    s = str(cond)
    if str(("cd", 0)) in s:
        return True

    leaves = set(find_f_list(cond, lambda e: [e] if type(e) in (int, str) else []))
    return leaves <= {"calldatasize", 3, 4, "lt", "gt", "le", "ge", "iszero", "bool"}


def selector_test(cond):
    """
    (hash, taken) if cond compares the selector with a function's hash,
    taken being whether it holds when they're equal, None otherwise.

    The dispatcher jumps to a function when the selector equals its hash,
    or past it when they differ: the function then follows the jump. That's
    a difference, zero if they're equal: a sub, or a xor.
    """
    if opcode(cond) == "iszero":
        res = selector_test(cond[1])
        return res and (res[0], not res[1])

    if match(cond, ("mask_shl", 32, 224, Any, ("cd", 0))) and cond[3] in (0, -224):
        # the selector itself, its being not 0: of the hash 0 - solc's
        # optimizer tests `iszero(selector)` for `selector == 0`
        return 0, False

    if opcode(cond) in ("and", "or"):
        # the selector is the hash and something else holds (Vyper's
        # dispatchers from 0.3.10: calldata.size > 3) - or, negated, it isn't
        # or something else doesn't: the function where it is
        taken = opcode(cond) == "and"
        tests = [t for t in map(selector_test, cond[1:]) if t and t[1] == taken]
        return tests[0] if len(tests) == 1 else None

    if opcode(cond) in ("eq", "xor") and len(cond) == 3:
        for num, other in ((cond[1], cond[2]), (cond[2], cond[1])):
            # (a selector is 4 bytes: a number past them is no hash, but a
            # comparison of more of the calldata - a word of it, say)
            if type(num) is int and 0 <= num < 2**32 and str(("cd", 0)) in str(other):
                return num, opcode(cond) == "eq"

    m = match(cond, ("add", ":int:num", ":other"))
    if m and str(("cd", 0)) in str(m.other):
        if m2 := match(m.other, ("mul", -1, ":selector")):
            # num - selector
            num, other = m.num, m2.selector
        else:
            # selector - (-num)
            num, other = -m.num % 2**256, m.other

        if 0 <= num < 2**32 and str(("cd", 0)) in str(other):
            return num, False

    return None


def strip_markers(trace):
    res = []
    for line in trace:
        if is_marker(line):
            continue
        if opcode(line) == "if":
            _, cond, if_true, if_false = line
            line = ("if", cond, strip_markers(if_true), strip_markers(if_false))
        res.append(line)
    return res


def entry_paths(trace, is_leaf, entry=()):
    """(entry, leaf) for every line of the dispatcher's trace for which is_leaf."""
    # (what's still to walk - lines, from where, what runs before them -
    # rather than a call for each branch: the ifs can be nested deeper than
    # python's recursion limit)
    todo = [(trace, 0, entry)]
    while todo:
        trace, start, entry = todo.pop()
        for idx in range(start, len(trace)):
            line = trace[idx]
            if is_leaf(line):
                yield entry, line

            elif opcode(line) == "setmem":
                entry = entry + (line,)

            elif opcode(line) == "if":
                _, cond, if_true, if_false = line
                # its branches, the one taken first, then what follows it
                todo.append((trace, idx + 1, entry))
                for taken, branch, other in (
                    (False, if_false, if_true),
                    (True, if_true, if_false),
                ):
                    branch_entry = entry
                    if reverts(other) and not is_dispatch(cond):
                        branch_entry += (
                            ("check", cond, taken, tuple(strip_markers(other))),
                        )
                    todo.append((branch, 0, branch_entry))
                break


def apply_entry(entry, trace):
    """The trace of a function, preceded by what runs before it."""
    res = list(trace)
    for item in reversed(entry):
        if opcode(item) == "check":
            _, cond, taken, other = item
            if taken:
                res = [("if", cond, res, list(other))]
            else:
                res = [("if", cond, list(other), res)]
        else:
            res = [item] + res

    return res


def entry_known(entry):
    """The conditions that hold once past the entry."""
    return tuple(
        item[1] if item[2] else is_zero(item[1])
        for item in entry
        if opcode(item) == "check"
    )


def common_entry(entries):
    """What all the given entries start with."""
    entries = list(entries)
    if not entries:
        return ()

    res = []
    for items in zip(*entries):
        if any(i != items[0] for i in items):
            break
        res.append(items[0])

    return tuple(res)


class Loader(EasyCopy):
    signatures = {}

    lines = {}  # global, let's assume one loader for now
    binary = []  # array of ints, each int represents a byte in the source file

    @staticmethod
    def find_sig(sig, add_color=False):
        if "???" in sig:
            return None

        if sig in Loader.signatures:
            if "unknown" not in Loader.signatures[sig]:
                return Loader.signatures[sig]

        if sig in cache_sigs[add_color]:
            return cache_sigs[add_color][sig]

        if len(sig) < 8:
            return None

        a = fetch_sig(sig)
        if a is None:
            return None

        # duplicate of get_func_name from signatures
        assert "inputs" in a
        # (make_abi does the same for the contract's own functions, on the
        # same cached dict, so the name of an external call used to depend
        # on whether the contract had the function itself)
        fix_input_names(a["inputs"])
        res = "{}({})".format(
            a["name"],
            ", ".join(
                [
                    # (a tuple as the types it's made of: its selector is the
                    # hash of that, see canonical_type)
                    colorize(
                        canonical_type(x["type"], x.get("components")),
                        COLOR_GRAY,
                        add_color,
                    )
                    + " "
                    + x["name"]
                    for x in a["inputs"]
                ]
            ),
        )

        cache_sigs[add_color][sig] = res
        return res

    def __init__(self):
        self.last_line = None
        self.jump_dests = []
        self.func_dests = {}  # func_name -> jumpdest
        self.hash_targets = {}  # hash -> (jumpdest, stack)
        self.func_list = []
        # conditions known to hold in the default function: no selector matched
        self.fallback_known = ()
        # what runs before each function (see entry_paths), by hash
        self.entries = {}

        self.binary = None

    def load_addr(self, address):
        assert address.isalnum()
        address = address.lower()

        logger.info("Fetching code for %s...", address)
        from web3 import Web3
        from web3.auto import w3

        code = w3.eth.get_code(Web3.to_checksum_address(address)).hex().removeprefix("0x")
        logger.debug("Code: %s", code)

        self.load_binary(code)

    def run(self, vm):
        assert self.binary is not None, "Did you run load_*() first?"

        try:
            # decompiles the code, starting from location 0
            # and running VM in a special mode that returns 'funccall'
            # in places where it looks like there is a func call

            trace = vm.run(0, timeout=LOADER_TIMEOUT, entry=())

            def func_calls(exp):
                if m := match(exp, ("funccall", ":fx_hash", ":target", ":stack")):
                    return [(m.fx_hash, m.target, m.stack)]
                else:
                    return []

            func_list = find_f_list(trace, func_calls)

            for fx_hash, target, stack in func_list:
                self.add_func(target=target, hash=fx_hash, stack=stack)

            for entry, line in entry_paths(
                trace, lambda line: opcode(line) == "funccall"
            ):
                self.entries.setdefault(padded_hex(line[1], 8), []).append(entry)

            # The default function is reached when none of the selector
            # comparisons matched. Knowing that spares the VM from exploring
            # every other function again when decompiling it.

            def selector_checks(exp):
                if m := match(exp, ("if", ":cond", ":if_true", ":if_false")):
                    if len(m.if_true) > 0 and match(
                        m.if_true[-1], ("funccall", Any, Any, Any)
                    ):
                        return [is_zero(m.cond)]
                    if len(m.if_false) > 0 and match(
                        m.if_false[-1], ("funccall", Any, Any, Any)
                    ):
                        return [m.cond]

                return []

            self.fallback_known = tuple(find_f_list(trace, selector_checks))

            # find default: where the dispatcher goes when no function
            # matches - a branch of its ifs without a function - if it's
            # always the same place. It isn't if there's a receive function
            # for no calldata, or if the dispatcher reverts in some cases
            # itself: the default function is then all of the dispatcher.

            def default_starts(exp):
                # (where it starts, the stack it starts with), None if it's
                # not a jump
                if (m := match(exp, ("if", ":cond", ":if_true", ":if_false"))) and (
                    is_dispatch(m.cond)
                ):
                    res = []
                    for branch in (m.if_true, m.if_false):
                        if find_f_list(branch, func_calls) == []:
                            if branch and (
                                m2 := match(branch[0], ("jd", ":jd", ":stack"))
                            ):
                                res.append((int(m2.jd), m2.stack))
                            else:
                                res.append(None)
                    return res
                return []

            starts = find_f_list(trace, default_starts) if func_list else []
            if len(set(s and s[0] for s in starts)) == 1 and starts[0] is not None:
                default = starts[0]
            else:
                default = None

            if default:
                target, stack = default
                self.add_func(target, name="_fallback", stack=stack)
                for entry, line in entry_paths(
                    trace, lambda line: opcode(line) == "jd" and line[1] == str(target)
                ):
                    self.entries.setdefault("_fallback", []).append(entry)
            else:
                self.add_func(0, name="_fallback")
                self.entries["_fallback"] = [()]

        except Exception:
            logger.exception("Loader issue.")
            self.add_func(0, name="_fallback")

        make_abi(self.hash_targets)
        for hash, (target, stack) in self.hash_targets.items():
            fname = get_func_name(hash)
            self.func_list.append((hash, fname, target, stack))

    def entry(self, hash):
        """What runs before the function (see entry_paths), None if unknown."""
        if hash not in self.entries:
            return None

        # a function the dispatcher jumps to from several places gets what
        # they all have in common
        return common_entry(self.entries[hash])

    def next_line(self, i):
        i += 1
        while i not in self.lines and self.last_line > i:
            i += 1

        if i <= self.last_line:
            return i
        else:
            return None

    def add_func(self, target, hash=None, name=None, stack=()):
        assert hash is not None or name is not None  # we need at least one
        assert not (hash is not None and name is not None)  # we don't want both

        if hash is not None:
            padded = padded_hex(hash, 8)  # lines[i-12][2]
            if padded in self.signatures:
                name = self.signatures[padded]
            else:
                name = "unknown_{}".format(padded)
                self.signatures[padded] = name

        if hash is None:
            self.hash_targets[name] = target, stack
        else:
            self.hash_targets[padded_hex(hash, 8)] = target, stack

        self.func_dests[name] = target

    def disasm(self):
        for line_no, op, param in self.parsed_lines:
            yield f"{hex(line_no)}, {op}, {hex(param) if param is not None else ''}"

    def load_binary(self, source):
        stack = []
        self.binary = []

        if source[:2] == "0x":
            source = source[2:]

        while len(source[:2]) > 0:
            num = int("0x" + source[:2], 16)
            self.binary.append(num)
            stack = [num] + stack
            source = source[2:]

        line = 0

        parsed_lines = []

        while len(stack) > 0:
            popped = stack.pop()

            orig_line = line

            if popped not in opcode_dict:
                op = "UNKNOWN"
                param = popped

            else:
                param = None
                op = opcode_dict[popped]

                if op == "jumpdest":
                    self.jump_dests.append(line)

                if op.startswith("push"):
                    num_words = int(op[4:])

                    param = 0
                    for i in range(num_words):
                        if stack:
                            param = param * 0x100 + stack.pop()
                            line += 1
                        else:
                            # past the end of the code: the bytes are 0
                            param = param * 0x100

            parsed_lines.append((orig_line, op, param))
            line += 1

        self.parsed_lines = parsed_lines
        self.last_line = line
        self.lines = {}

        for line_no, op, param in parsed_lines:
            if op[:3] == "dup":
                param = int(op[3:])
                op = "dup"

            if op[:4] == "swap":
                param = int(op[4:])
                op = "swap"

            self.lines[line_no] = (line_no, op, param)

        # solidity sets its free memory pointer, mem[64], before anything else
        # (but a library's check that it's called with a delegatecall): what
        # the memory model takes it for (see variants)
        set_free_memory_pointer(sets_free_memory_pointer(parsed_lines))

        return self.lines
