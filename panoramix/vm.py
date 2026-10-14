import logging
import time
import sys
from copy import copy

from panoramix.core import algebra, arithmetic, variants
import panoramix.utils.opcode_dict as opcode_dict
from panoramix.core.algebra import (
    add_op,
    apply_mask,
    bits,
    lt_op,
    mask_op,
    mul_op,
    neg_mask_op,
    or_op,
    shl_op,
    shr_op,
    sub_op,
    to_bytes,
    value_range,
    CannotCompare,
)
from panoramix.core.arithmetic import (
    VOLATILE,
    changed_reads,
    is_zero,
    mentions,
    simplify_bool,
)
from panoramix.core.masks import to_mask, to_neg_mask
from panoramix.core.memloc import max_value_bits
from panoramix.matcher import match
from panoramix.prettify import pprint_trace
from panoramix.utils.helpers import (
    C,
    MAX_EXP_SIZE,
    EasyCopy,
    all_concrete,
    contains,
    exp_size,
    find_op_list,
    get_op,
    opcode,
    precompiled,
    precompiled_var_names,
    replace,
    replace_f_stop,
)

from .loader import apply_entry, entry_known, is_dispatch, selector_test
from .stack import Stack, fold_stacks, stack_vars

logger = logging.getLogger(__name__)


"""

    A symbolic EVM.

    It executes the contract, and returns the resulting `trace` of execution - which is the decompiled form.


    The most difficult part of this module is the loop detection and simplification algorithm.
    In over 10 iterations I didn't find a simpler way that is just as effective.

    Unfortunately, because of the complexity, I don't fully understand how it works.
    Ergo, I cannot explain it to you :) Good luck!

    On the upside, some stuff, like apply_stack is quite straightforward.

"""


def mem_load(pos, size=32):
    return ("mem", ("range", pos, size))


def in_dispatcher(node):
    """
    Whether every branch on the way to node is one of the dispatcher's: on the
    selector or on calldatasize (see loader.is_dispatch) - not one of a
    function's body.
    """
    res = getattr(node, "dispatch", None)
    if res is None:
        # (prev is the node it's reached from, once it's set)
        parent = node.prev if isinstance(node.prev, Node) else None
        res = (parent is None or in_dispatcher(parent)) and (
            node.condition is True or is_dispatch(node.condition)
        )
        node.dispatch = res
    return res


def find_nodes(node, f):
    """Find the nodes below `node` (itself included) where f(node) returns true."""
    assert type(node) == Node

    res = []
    to_visit = [node]

    while to_visit:
        n = to_visit.pop()
        if f(n):
            res.append(n)
        to_visit.extend(reversed(n.next))

    return res


# How many nodes a function is explored in, at most - the timeout of the run
# (see VM.run) is the other bound. The biggest functions of the contracts we
# decompile (Seaport's, the Universal Router's execute) take 10,000, most of
# them one path after the other. Past WIDE_NODE_COUNT, the exploration goes
# on only while it narrows down - while fewer than one of 16 nodes is still to
# explore: paths that keep branching (1inch's predicates, which inline each
# other) make an output too long to read anyway. Finding the functions only
# (just_fdests) needs the dispatcher, explored first.
MAX_NODE_COUNT = 20_000
WIDE_NODE_COUNT = 5_000
MAX_FDESTS_NODE_COUNT = 5_000
node_count = 0


"""

    Path conditions.

    Every node carries `known`: the tuple of conditions that are known to hold
    on the execution path leading to it (one per `jumpi` taken along the way).
    When the VM reaches a `jumpi` whose condition is implied by one of them,
    only the feasible branch is followed. Without this, every conditional
    that is (partially) decided by an earlier one doubles the number of paths
    to explore, and contracts that repeat the same checks - e.g. the
    `success`/`returndatasize`/`extcodesize` checks in the SafeERC20 and
    Address libraries, or `token == address(0)` special cases - blow past
    MAX_NODE_COUNT and only get a truncated, exponentially unrolled
    decompilation.

    Facts about values that can change while the contract runs (storage,
    balances, the result of the last external call...) are forgotten as soon
    as the relevant state may have changed, see `forget`. Everything else
    (calldata, msg.sender, chainid...) is constant for the whole execution.

"""

STATE_CHANGING_OPS = (
    "call",
    "staticcall",
    "delegatecall",
    "callcode",
    "create",
    "create2",
    "selfdestruct",
)

# the instructions that put no word on the stack
NO_RESULT_OPS = (
    "stop",
    "calldatacopy",
    "codecopy",
    "extcodecopy",
    "returndatacopy",
    "mcopy",
    "pop",
    "mstore",
    "mstore8",
    "sstore",
    "tstore",
    "jump",
    "jumpi",
    "jumpdest",
    "log0",
    "log1",
    "log2",
    "log3",
    "log4",
    "return",
    "revert",
    "selfdestruct",
    "invalid",
    "assert_fail",
)


def stack_inputs(line):
    """How many words of the stack the instruction of the line uses."""
    _, op, param = line[:3]
    if op == "dup":
        return param
    if op == "swap":
        return param + 1
    diff = opcode_dict.stack_diffs.get(op)
    if diff is None:
        # (an invalid instruction)
        return 0
    return (0 if op in NO_RESULT_OPS else 1) - diff


# the most bytes of memory known_zeroes reads
MAX_ZEROES_SIZE = 2**16

# A codecopy of more bytes than that is left as code.data: the number would be
# unreadable anyway, and one of more than 4300 digits (1786 bytes) can't even be
# turned into a string since python 3.11, which aborts the whole function.
MAX_CODECOPY_SIZE = 1024


"""

    What's in memory.

    A run that starts at the beginning of the code (the dispatcher, or the
    default function when it's all of the dispatcher) starts with a memory
    of zeroes: ("memory_fresh",) is known. What the path writes at a known
    place is known too, ("memory", start, size, value) - the value of a word,
    or a byte, None for the other writes - and a word read from there is
    that, not a variable. That's how Vyper (up to 0.2) reads the selector:
    it writes the calldata at 28, and reads the word at 0.

    Nothing else writes to the memory, not even a call (but where its
    result goes): they stay known until a write to their place, or to a
    place that isn't known. A loop may write anywhere, see set_label.

"""

MEMORY_FACTS = ("memory", "memory_fresh")

# known while no call was made: no data was returned (forgotten with what
# mentions returndatasize, at a call and at the head of a loop)
RETURNDATASIZE_ZERO = ("returndatasize_zero",)

# Where a recursive internal function decompiled apart returns (see
# VM.internal): the address below its arguments, as its caller pushed it.
RETURN_ADDRESS = ("return_address",)


def mask_value(value, size, offset=0, shl=0):
    """mask_op, that also computes a number, and leaves a word as it is."""
    if (size, offset, shl) == (256, 0, 0):
        return value
    if type(value) == int:
        return apply_mask(value, size, offset, shl)
    return mask_op(value, size=size, offset=offset, shl=shl)


def upper_bound(v, known):
    """The highest value v can have, by the known conditions, None if none."""
    res = None
    for fact in known:
        n = None
        if m := match(fact, (":op", v, ":int:n")):
            n = {"lt": m.n - 1, "le": m.n}.get(m.op)
        elif m := match(fact, (":op", ":int:n", v)):
            n = {"gt": m.n - 1, "ge": m.n}.get(m.op)
        elif m := match(fact, ("iszero", (":op", v, ":int:n"))):
            n = {"gt": m.n, "ge": m.n - 1}.get(m.op)
        elif m := match(fact, ("iszero", (":op", ":int:n", v))):
            n = {"lt": m.n, "le": m.n - 1}.get(m.op)
        if n is not None and (res is None or n < res):
            res = n
    return res


def known_bounds(known):
    """
    {x: (lowest, highest)}: what the known conditions say of what they
    compare with a number (see algebra.value_range)
    """
    flip = {"lt": "gt", "gt": "lt", "le": "ge", "ge": "le", "eq": "eq"}
    negate = {"lt": "ge", "ge": "lt", "gt": "le", "le": "gt"}
    top = 2**256 - 1
    res = {}
    for fact in known:
        neg = False
        while opcode(fact) == "iszero":
            neg, fact = not neg, fact[1]
        op = opcode(fact)
        if op == "var_bits" and not neg and type(fact[2]) is int:
            v, lo, hi = fact[1], 0, 2 ** fact[2] - 1
        elif op in flip and len(fact) == 3:
            a, b = fact[1], fact[2]
            if type(a) is int and type(b) is not int:
                a, b, op = b, a, flip[op]
            if type(a) is int or type(b) is not int or not 0 <= b <= top:
                continue
            if neg:
                if op == "eq":
                    continue
                op = negate[op]
            v = a
            lo, hi = {
                "lt": (0, b - 1),
                "le": (0, b),
                "gt": (b + 1, top),
                "ge": (b, top),
                "eq": (b, b),
            }[op]
        else:
            continue
        old_lo, old_hi = res.get(v, (0, top))
        lo, hi = max(lo, old_lo), min(hi, old_hi)
        if lo <= hi:
            res[v] = (lo, hi)
    return res


def and_op(left, right, known):
    """
    left & right - a mask when one of them is one (see Stack.simplify),
    which for a 256 ** e - 1 is when the conditions known say what e is
    (see masks.to_mask).
    """
    exp = arithmetic.eval(("and", left, right))
    if opcode(exp) != "and" or len(exp) != 3:
        return exp
    left, right = exp[1], exp[2]
    bounds = None
    if contains(exp, "exp"):
        bounds = known_bounds(known)
    if mask := to_mask(left, bounds):
        return mask_op(right, *mask)
    if mask := to_mask(right, bounds):
        return mask_op(left, *mask)
    if neg := to_neg_mask(left, bounds):
        return neg_mask_op(right, *neg)
    if neg := to_neg_mask(right, bounds):
        return neg_mask_op(left, *neg)
    return exp


def bound(v, known):
    """
    The highest value v can have, below 2**128: by the known conditions, or
    by what it's made of - a size of what the call has, say, that gas keeps
    far from 2**256 (see memloc.BOUNDED_SYMBOLS). None if there's none.
    """
    if opcode(v) == "min":
        # (the size of what a call wrote, see VM.output_write)
        tops = [
            t
            for t in (x if type(x) is int else bound(x, known) for x in v[1:])
            if t is not None
        ]
        return min(tops) if tops and 0 <= min(tops) < 2**128 else None

    top = upper_bound(v, known)
    if top is None:
        # the bounds of what it's made of (see VM.snapshot)
        bounds = {fact[1]: fact[2] for fact in known if opcode(fact) == "var_bits"}
        if (b := max_value_bits(v, bounds)) <= 128:
            top = 2**b - 1
    return top if top is not None and 0 <= top < 2**128 else None


def write_range(known, start, size):
    """
    The range of memory a write at a place or of a size computed at runtime
    may touch: base + k * i with i below a bound (an element of an array in
    memory, whose index was checked; after data of a size bounded), as
    many bytes as size can be. None if it's not known.
    """
    size_top = size if type(size) == int else bound(size, known)
    if size_top is None:
        return None

    if type(start) == int:
        return start, start + size_top

    if opcode(start) != "add":
        return None

    bases = [t for t in start[1:] if type(t) == int]
    rest = [t for t in start[1:] if type(t) != int]
    if len(bases) != 1 or not rest or not 0 <= bases[0] < 2**128:
        return None
    base = bases[0]
    offset = rest[0] if len(rest) == 1 else ("add",) + tuple(rest)
    if m := match(offset, ("mul", ":int:k", ":i")):
        k, i = m.k, m.i
    elif m := match(offset, ("mask_shl", ":int:size", 0, ":int:shl", ":i")):
        # (i & mask) << shl: at most i << shl
        k, i = 2**m.shl, offset[4]
    else:
        k, i = 1, offset
    top = bound(i, known)
    if top is not None and 0 < k < 2**64:
        return base, base + k * top + size_top

    return None


def write_memory(known, rng, value):
    """What's known once value is written to the memory range rng."""
    start, size = rng[1], rng[2]
    if type(start) != int or type(size) != int:
        if bounds := write_range(known, start, size):
            # somewhere there
            return write_memory(
                known, ("range", bounds[0], bounds[1] - bounds[0]), None
            )
        # anywhere
        return tuple(fact for fact in known if opcode(fact) not in MEMORY_FACTS)

    if size == 0:
        return known

    if size not in (1, 32) and not (type(value) is int and size < 32):
        # (a number of size bytes - codecopy's - is kept)
        value = None

    if value is not None and get_op(value, "mem") is not None:
        # a read of memory (mcopy's), done before the write: what the memory
        # has there then - the write may change it (mcopy(0x90, 0x80, 32)),
        # or a later one - not the read
        m = match(value, ("mem", ("range", ":int:src", 32)))
        value = read_memory(known, m.src) if m and size == 32 else None

    end = start + size
    res = []
    for fact in known:
        if opcode(fact) != "memory" or fact[1] + fact[2] <= start or end <= fact[1]:
            res.append(fact)
            continue

        # what the write leaves of a value written before, on each side
        _, f_start, f_size, f_value = fact
        f_end = f_start + f_size
        if f_start < start:
            n, off = start - f_start, 8 * (f_end - start)
            v = f_value and mask_value(f_value, 8 * n, off, -off)
            res.append(("memory", f_start, n, v))
        if end < f_end:
            n = f_end - end
            v = f_value and mask_value(f_value, 8 * n)
            res.append(("memory", end, n, v))

    return tuple(res) + (("memory", start, size, value),)


def entry_memory(entries):
    """
    What's known of the memory when a function starts: what every path of
    the dispatcher to it wrote, in a memory of zeroes (see loader.entry_paths).
    """
    res = None
    every = set()
    for entry in entries or ():
        known = (("memory_fresh",),)
        for item in entry:
            if opcode(item) == "setmem":
                known = write_memory(known, item[1], item[2])
        every |= set(known)
        res = set(known) if res is None else res & set(known)
    if not res:
        return ()

    # what only some of them wrote isn't known (not even that it's still
    # zero), as where paths merge (see _merge_at)
    for fact in every - res:
        if opcode(fact) == "memory":
            res.add(("memory", fact[1], fact[2], None))
    return tuple(sorted(res, key=str))


def read_memory(known, addr):
    """The word at addr, if what's there is known, None otherwise."""
    if type(addr) != int or not any(opcode(f) in MEMORY_FACTS for f in known):
        return None

    end = addr + 32
    res, covered = 0, 0
    for fact in known:
        if opcode(fact) != "memory":
            continue
        _, start, size, value = fact
        lo, hi = max(start, addr), min(start + size, end)
        if lo >= hi:
            continue
        if value is None:
            return None

        # the bytes lo..hi of the value of the bytes start..start + size, to
        # their place in the word
        off = 8 * (start + size - hi)
        res = or_op(res, mask_value(value, 8 * (hi - lo), off, 8 * (end - hi) - off))
        covered += hi - lo

    if covered < 32 and ("memory_fresh",) not in known:
        return None

    return res


def known_zeroes(known, start, size):
    """
    True if the memory start..start + size is known to be zeroes. Not for
    more than MAX_ZEROES_SIZE bytes: it's read a word at a time, which for
    a call's output of 2**255 bytes, say, wouldn't end.
    """
    if type(start) is not int or type(size) is not int or not 0 < size <= MAX_ZEROES_SIZE:
        return False
    return all(read_memory(known, start + off) == 0 for off in range(0, size, 32))


def forget(known, names):
    """
    What's still known once what the names read may have changed. What's
    in memory only changes by the writes to it (see write_memory), even at
    a call: what a fact about it says goes when its value reads something
    that changed - not that the place was written (it isn't fresh memory,
    of zeroes, any more).
    """
    res = []
    for fact in known:
        if opcode(fact) == "memory":
            if mentions(fact[3], names):
                fact = fact[:3] + (None,)
        elif opcode(fact) != "memory_fresh" and mentions(fact, names):
            continue
        res.append(fact)
    return tuple(res)


def is_known(exp, known):
    """Evaluate `exp` to True/False if it is decided by the known conditions, None otherwise."""
    for fact in reversed(known):
        if opcode(fact) in MEMORY_FACTS or opcode(fact) == "var_bits":
            continue
        res = arithmetic.eval_bool(exp, fact, symbolic=False)
        if res is not None:
            return res

    return None


def decided(value, known):
    """
    value as the number it is where the known conditions decide it: 0 if
    they say it's false - 1 if they say it's true and it's a truth value -,
    else value. (`a && b` leaves a on the stack where it's false: that's 0.)
    """
    if type(value) is int:
        return value
    res = is_known(value, known)
    if res is False:
        return 0
    if res is True and arithmetic.is_bool(value):
        return 1
    return value


class Node:
    def __str__(self):
        return f"Node({self.jd})"

    def __repr__(self):
        # technically not a proper _repr_, but some trace printout functions use this
        # instead of a proper str
        return self.__str__()

    def __init__(self, vm, start, safe, stack, condition=True, trace=None, known=()):
        global node_count

        node_count += 1

        self.vm = vm
        self.prev = []
        self.next = []
        self.trace = trace
        self.start = start
        self.safe = safe
        self.stack = stack
        self.history = {}
        self.depth = 0
        self.label_history = {}
        self.label = None

        # condition: the one under which this node is reached from its parent.
        # known: all the conditions that hold on the path leading here.
        self.condition = condition
        self.known = known

        # set when the execution from this node on got merged with other
        # paths, see merge_branches
        self.merged = False

        stack_obj = Stack(stack)
        self.jd = (start, len(stack), tuple(stack_obj.jump_dests(vm.loader.jump_dests)))

    def make_trace(self):
        res = []
        node = self

        # the nodes form long chains (a jump at the end of each), walked
        # here in a loop rather than recursively - the recursion is only as
        # deep as the ifs are nested
        while node is not None:
            res.extend(node._begin_trace())

            if node.trace is None:
                break

            next_node = None

            for line in node.trace:
                if opcode(line) == "jump" and isinstance(line[1], Node):
                    # always the last line
                    next_node = line[1]

                elif opcode(line) == "if" and isinstance(line[2], Node):
                    # the last line, unless the paths merge again after the if -
                    # see merge_branches - in which case a jump to the merged
                    # node follows.
                    _, cond, if_true, if_false = line
                    res.append(
                        ("if", cond, if_true.make_trace(), if_false.make_trace())
                    )

                else:
                    res.append(line)

            node = next_node

        return res

    def _begin_trace(self):
        """What goes before the node's own lines in the decompiled trace."""
        if self.trace is None:
            return [("undefined", "decompilation didn't finish")]

        if self.vm.just_fdests and (
            self.vm.lines.get(self.start, (None, None))[1] == "jumpdest"
        ):
            # the loader looks for these to find the default function, and
            # the stack it starts with: where the dispatcher falls through to
            # it, and where it jumps to it (solc's `calldatasize < 4`)
            begin = [("jd", str(self.jd[0]), tuple(self.stack))]
        elif self.vm.just_fdests and self.trace != [("revert", None)]:
            t = self.trace[0]
            if match(t, ("jump", ":target_node", ...)):
                begin = [("jd", str(self.jd[0]), tuple(self.stack))]
            else:
                begin = ["?"]
        else:
            begin = []

        if self.is_label():
            begin_vars = []
            for _, var_idx, var_val, _ in self.label.begin_vars:
                begin_vars.append(("setvar", var_idx, var_val))

            if find_nodes(
                self,
                lambda n: n.trace
                and opcode(n.trace[-1]) == "goto"
                and n.trace[-1][1] in (self, self.label),
            ):
                begin.append(("label", self, tuple(begin_vars)))
            else:
                # Nothing loops back here any more: a merge (see
                # merge_branches) took the paths that did to the
                # continuation of an if above, where the loop is found
                # again. What's left here is the first iteration.
                begin.extend(begin_vars)

        return begin

    def set_label(self, loop_dest, vars, stack):
        self.label = loop_dest
        loop_dest.begin_vars = vars

        assert len(self.stack) == len(stack)
        self.stack = stack
        loop_dest.prev_trace = loop_dest.trace
        loop_dest.trace = [("jump", self)]
        loop_dest.next = []
        self.set_prev(loop_dest)

        # This node is now the body of a loop, executed for every iteration,
        # the first one included: what is known there is what was known
        # before the loop, not what the first iteration found out on its way
        # here (its checks are made again, and the first iteration is no
        # longer decompiled on its own). And what we learned about the state
        # before the loop doesn't necessarily hold for the next iterations.
        self.known = tuple(
            fact
            for fact in forget(loop_dest.known, VOLATILE)
            if opcode(fact) not in MEMORY_FACTS
        )

        # Except the code addresses in memory - where Vyper keeps the return
        # address of a private function, e.g. one with a loop - for as long
        # as no iteration changes them: see continue_loops.
        rejected = getattr(self, "rejected_memory", ())
        self.assumed_memory = tuple(
            fact
            for fact in loop_dest.known
            if opcode(fact) == "memory"
            and fact[2] == 32
            and type(fact[3]) == int
            and fact[3] in self.vm.loader.jump_dests
            and fact not in rejected
        )
        self.known += self.assumed_memory

    def set_prev(self, prev):
        self.prev = prev
        self.depth = prev.depth + 1

        self.history = copy(prev.history)
        self.history[prev.jd] = prev

        self.label_history = copy(prev.label_history)
        if prev.label:
            self.label_history[prev.jd] = prev.label

        prev.next.append(self)

    def is_label(self):
        return self.label is not None

    def run(self):
        logger.debug("Node.run(%s)", self)
        self.prev_trace = self.trace
        self.trace = self.vm._run(
            self.start, self.safe, self.stack, self.condition, self.known
        )

        last = self.trace[-1]

        if opcode(last) == "jump":
            n = last[1]

            n.set_prev(self)

        if opcode(last) == "if":
            link_if(self, last)


def switch_values(exp):
    """
    (x, its values) for a part x of exp of a few values: x % n, or a few of
    the bits of something (x & 1 is how Vyper's dispatcher takes x % 2) -
    see VM.code_switch. None if exp has none.
    """
    for m in find_op_list(exp, "mod"):
        if len(m) == 3 and type(m[2]) is int and 2 <= m[2] <= 256:
            return m, range(m[2])
    for m in find_op_list(exp, "mask_shl"):
        if (
            len(m) == 5
            and all(type(e) is int for e in m[1:4])
            and 1 <= m[1] <= 8
            and m[2] + m[3] >= 0
            and type(m[4]) is not int
        ):
            return m, [k << (m[2] + m[3]) for k in range(2 ** m[1])]
    return None


def link_if(node, line):
    """The branches of the if `line` that ends node's trace go on from it -
    and so do those of the ifs made in between (see VM.code_switch)."""
    for child in line[2:4]:
        child.set_prev(node)
        if child.trace and opcode(child.trace[-1]) == "if":
            link_if(child, child.trace[-1])


class VM(EasyCopy):
    def __init__(self, loader, just_fdests=False, internal=None):
        self.loader = loader

        # The recursive internal functions: their entry -> (how many words
        # their arguments take, their results, the stack a call of
        # themselves takes). A call of one isn't unrolled (it would be until
        # the VM stops, the EVM's stack of 1024 words being what bounds the
        # recursion), it's a line - its arguments, variables for its results
        # - and the function is decompiled apart (see the decompiler). Those
        # it's given (the function decompiled apart), and those it finds:
        # where a function is called again in its own call (see
        # recursion_of).
        self.internal = dict(internal or {})
        self.found = {}

        # (line_no, op, param)
        self.lines = loader.lines  # a shortcut

        self.just_fdests = just_fdests

        self.counter = 0
        # (p, jd) of the merges that were too early to tell, see merge_branches
        self.pending_merges = {}
        # how many results of each precompile were named
        self.precompile_results = {}
        # the number of the next loop found (see loop_key)
        self.next_loop_key = 1
        self.known = ()
        self.should_quit = lambda: False
        global node_count
        node_count = 0

    def loop_key(self, head):
        """
        What the variables of the loop that starts at the node head are
        numbered after (see stack.fold_stacks): a number of its own, the same
        when it's found again. Not the depth of a node in the tree - a loop in
        another can start as deep as it: `for i... for j... if (j == i)` was
        `if i == i` - nor where it starts in the code: a function with a loop
        called twice, `(inner(a), inner(b))`, has the result of the first loop
        kept while the second runs. Nor what was found before, by the VMs of
        the functions decompiled before it: its names would depend on them.
        """
        if getattr(head, "loop_key", None) is None:
            head.loop_key = self.next_loop_key
            self.next_loop_key += 1
        return head.loop_key

    def name_past(self, given):
        """
        Its own variables named past those in what it's given: a function's
        stack, what's known there and what runs before it come from the
        dispatcher (see Loader.run), whose variables another VM named - the
        same name would be two variables.
        """
        bases = set(precompiled_var_names.values())
        for var in find_op_list(given, "var"):
            if len(var) != 2:
                continue
            name = var[1]
            if type(name) == int:
                # a loop's (see loop_key)
                self.next_loop_key = max(self.next_loop_key, name // 1000 + 1)
            elif type(name) != str:
                continue
            elif name[:1] == "_" and name[1:].isdigit():
                self.counter = max(self.counter, int(name[1:]))
            else:
                # a precompile's result: signer, signer2...
                for base in bases:
                    count = name[len(base) :]
                    if name.startswith(base) and (count == "" or count.isdigit()):
                        self.precompile_results[base] = max(
                            self.precompile_results.get(base, 0), int(count or 1)
                        )

    def run(
        self,
        start,
        history={},
        condition=None,
        re_run=False,
        stack=(),
        timeout=0,
        known=(),
        entry=None,
        memory=(),
        inside=False,
    ):
        """
        `known` is a tuple of conditions known to hold when `start` is reached,
        e.g. for the default function: that no function selector matched.

        `entry` is what runs before `start` (see loader.entry_paths), and
        goes at the beginning of the trace. If it isn't known, the free memory
        pointer is assumed to be 0x60, as the old compilers set it. `memory`
        is what's known of the memory then (see entry_memory).

        `inside`: start is an internal function's, decompiled apart (see
        internal) - nothing is known of the memory nor of what ran before.
        """
        self.name_past((stack, known, entry, memory))

        time_start = time.monotonic()

        max_nodes = MAX_FDESTS_NODE_COUNT if self.just_fdests else MAX_NODE_COUNT
        # the nodes still to explore, as last counted
        frontier = [0]

        def should_quit():
            return (
                node_count > max_nodes
                or (node_count > WIDE_NODE_COUNT and 16 * frontier[0] > node_count)
                or (timeout and (time.monotonic() - time_start > timeout))
            )

        self.should_quit = should_quit

        if inside:
            before = []
        elif entry is None:
            before = (
                [("setmem", ("range", 0x40, 32), 0x60)]
                if variants.FREE_MEMORY_POINTER
                else []
            )
        else:
            before = []
            known = tuple(known) + entry_known(entry)
            if start == 0 and len(entry) == 0:
                known += (("memory_fresh",),)
            known += tuple(f for f in memory if f not in known)
            # no call was made yet (the dispatcher makes none): no data was
            # returned - until one is, see forget (minimal proxies use
            # returndatasize as a cheap 0)
            known += (RETURNDATASIZE_ZERO,)

        func_node = Node(
            vm=self, start=start, safe=True, stack=list(stack), known=tuple(known)
        )
        trace = before + [
            ("jump", func_node, "safe", tuple()),
        ]

        root = Node(
            vm=self,
            trace=trace,
            start=start,
            safe=True,
            stack=list(stack),
            known=tuple(known),
        )
        func_node.set_prev(root)

        """

            BFS symbolic execution, ends up with a decompiled
            code, with labels and gotos.

            Depth-first would be way easier to implement, but it tends
            to work way slower because of the loops.

        """

        for j in range(20):
            for i in range(200):
                """

                Find all the jumps, and expand them until
                the next jump.

                """

                self.expand_trace(root)

                """
                    find all the jumps that lead to an already
                    reached jumpdest (with similar stack, otherwise
                    we'd catch function calls as all).

                    replace them with 'loop' identifier
                """

                self.replace_loops(root)

                """
                    the calls of a function in its own call: not unrolled
                """

                self.replace_recursions(root)

                """
                    turn them into loops right away: the later this is
                    done, the bigger the subtree that gets thrown away and
                    explored again when a loop is set up.
                """

                self.continue_loops(root)

                """
                    find the ifs whose branches all end up at the same
                    jumpdest, and continue from there only once.
                """

                self.merge_branches(root)

                """
                    repeat until there are no more jumps
                    to explore (so, until the trace didn't change)

                """

                nodes = find_nodes(root, lambda n: n.trace is None)
                frontier[0] = len(nodes)

                if len(nodes) == 0 or should_quit():
                    break

            trace = self.continue_loops(root)

            # tr = root.make_trace()
            nodes = find_nodes(root, lambda n: n.trace is None)

            if len(nodes) == 0 or should_quit():
                break

        if should_quit():
            logger.warning(
                "VM stopped prematurely. Node count %i, after %.2f seconds.",
                node_count,
                time.monotonic() - time_start,
            )

        logger.debug("%i nodes, %.2f seconds", node_count, time.monotonic() - time_start)
        tr = root.make_trace()
        if entry:
            tr = apply_entry(entry, tr)
        return tr

    def expand_trace(self, root):
        nodes = find_nodes(root, lambda n: n.trace is None)

        if self.just_fdests:
            # the dispatcher first, a function's body with the nodes left once
            # it's done: the body of the default function, which takes the
            # paths no selector matches, branches as it goes - explored with
            # it, it took all of them (MAX_FDESTS_NODE_COUNT), and the
            # functions past where the dispatcher was cut were left to the
            # default function, as its ifs on the selector
            dispatch = [n for n in nodes if in_dispatcher(n)]
            if dispatch:
                nodes = dispatch

        for node in nodes:
            if self.should_quit():
                # symbolic execution of a single node can take very long when
                # the expressions get big, so the timeout is checked here too
                break

            node.run()

    def replace_loops(self, root):
        nodes = find_nodes(root, lambda n: n.trace is None)

        for node in nodes:
            if (
                node.jd in node.history
                and node.jd[1] > 0
                and len(node.history[node.jd].stack) == len(node.stack)
            ):  # jd[1] == stack_len
                folded, vars = fold_stacks(
                    node.history[node.jd].stack,
                    node.stack,
                    self.loop_key(node.history[node.jd]),
                )
                loop_line = (
                    "loop",
                    node.history[node.jd],
                    node.stack,
                    folded,
                    tuple(vars),
                )
                node.trace = [loop_line]

    def replace_recursions(self, root):
        """
        Where a node starts the call of an internal function in its own call
        (see recursion_of), the function is a recursive one: every call of it
        is a line (see internal_call) - those of the path explored, the first
        one included, explored again as such.
        """
        if self.just_fdests:
            return

        for node in find_nodes(root, lambda n: n.trace is None):
            if self.internal_call_at(node.start, node.stack):
                # (a call already, once it runs)
                continue
            rec = self.recursion_of(node)
            if rec is None or rec[0] in self.internal:
                continue

            entry, n, m, frame = rec
            self.internal[entry] = (n, m, frame)
            self.found[entry] = (n, m, frame)

            for x in find_nodes(
                root,
                lambda x: x.start == entry
                and x.trace is not None
                and self.internal_call_at(x.start, x.stack),
            ):
                x.trace = None
                x.next = []
            return

    def recursion_of(self, node):
        """
        (entry, n, m, frame) if node starts the call of an internal function
        in its own call - recursion: an ancestor starts there too, with the
        address its call returns to (the topmost on its stack) on node's
        stack, all below it the same, and above it as many words as node has
        above its own, the n arguments; m the words of its results, as one
        of the calls of it returned them (None while none is known: the
        recursion is unrolled another level, until one is); frame the words
        of the stack a call of itself takes. None if node isn't one.
        """
        start = node.start
        if type(start) is not int or start not in self.loader.jump_dests:
            return None

        def top_dest(stack):
            for i in range(len(stack) - 1, -1, -1):
                if type(stack[i]) is int and stack[i] in self.loader.jump_dests:
                    return i
            return None

        q = top_dest(node.stack)
        if q is None:
            return None

        n = frame = None
        m = None
        # the ancestors, the nearest first
        prev = node.prev
        while isinstance(prev, Node):
            if prev.start != start or len(prev.stack) >= len(node.stack):
                prev = prev.prev
                continue
            p = top_dest(prev.stack)
            if (
                p is None
                or p >= q
                or len(prev.stack) - p != len(node.stack) - q
                or tuple(prev.stack[: p + 1]) != tuple(node.stack[: p + 1])
            ):
                prev = prev.prev
                continue
            if n is None:
                entry, n, frame = self.call_entry(prev, p, node, q)
            # its results: where its call returns, the stack below the
            # address the same, and the address gone - not the code there
            # while the call runs (solc's optimizer makes the end of a
            # function, which returns to the address the stack has, the
            # end of others too: it's where they return)
            ret, below = prev.stack[p], tuple(prev.stack[:p])
            for r in find_nodes(
                prev,
                lambda x: x.start == ret
                and len(x.stack) >= p
                and tuple(x.stack[:p]) == below
                and not (len(x.stack) > p and x.stack[p] == ret),
            ):
                m = len(r.stack) - p
                break
            if m is not None:
                return entry, n, m, frame
            prev = prev.prev

        return None

    def call_entry(self, prev, p, node, q):
        """
        (entry, n, frame) of the call of the function prev and node are the
        calls of (the address it returns to at p on prev's stack, at q on
        node's): where it starts, the outermost node of the call of prev
        that the call of node has too (one where as many words are above
        the address) - not a block the first ones go to, which would make
        what they push before arguments of it.
        """

        def within(x, k, stack):
            return len(x.stack) > k and tuple(x.stack[: k + 1]) == tuple(stack[: k + 1])

        outer = []
        x = prev
        while isinstance(x, Node) and within(x, p, prev.stack):
            outer.append(x)
            x = x.prev
        inner = []
        y = node
        while isinstance(y, Node) and within(y, q, node.stack):
            inner.append(y)
            y = y.prev

        for x in reversed(outer):
            for y in inner:
                if y.start == x.start and len(y.stack) - q == len(x.stack) - p:
                    return x.start, len(x.stack) - p - 1, len(y.stack) - len(x.stack)

        return prev.start, len(prev.stack) - p - 1, len(node.stack) - len(prev.stack)

    def internal_call_at(self, start, stack):
        """Whether start, with stack, is the call of a recursive internal function."""
        if type(start) is not int or start not in self.internal:
            return False
        n = self.internal[start][0]
        if len(stack) <= n:
            return False
        ret = stack[len(stack) - n - 1]
        return type(ret) is int and ret in self.loader.jump_dests

    def internal_call(self, start, condition):
        """
        The trace of the call of the recursive internal function at start (the
        stack the VM's): ("internal", start, its arguments, the names of the
        variables of its results), then where it returns, the address below
        its arguments, with them on the stack - the state as a call may leave
        it: the storage written, the memory too.
        """
        n, m, frame = self.internal[start]
        trace = []

        def add(line):
            trace.append(line)

        self.snapshot(add, "delegatecall")
        stack = self.stack.stack
        args = tuple(stack[len(stack) - n :])
        ret = stack[len(stack) - n - 1]
        results = []
        for _ in range(m):
            self.counter += 1
            results.append(f"_{self.counter}")
        trace.append(("internal", start, args, tuple(results)))

        self.known = tuple(
            fact
            for fact in forget(self.known, VOLATILE)
            if opcode(fact) not in MEMORY_FACTS
        )
        node = Node(
            self,
            start=ret,
            safe=False,
            stack=tuple(stack[: len(stack) - n - 1])
            + tuple(("var", r) for r in results),
            condition=condition,
            known=self.known,
        )
        trace.append(("jump", node))
        return trace

    def continue_loops(self, root):
        loop_list = find_nodes(
            root,
            lambda n: n.trace is not None
            and len(n.trace) == 1
            and opcode(n.trace[0]) == "loop",
        )

        for node in loop_list:
            (line,) = node.trace
            op, loop_dest, stack, new_stack, vars = line
            assert op == "loop"

            if loop_dest.is_label():
                broken = tuple(
                    fact
                    for fact in getattr(loop_dest, "assumed_memory", ())
                    if fact not in node.known
                )
                if broken:
                    # an iteration changes what was assumed in memory at the
                    # start of the loop: explored again without it
                    loop_dest.rejected_memory = (
                        getattr(loop_dest, "rejected_memory", ()) + broken
                    )
                    loop_dest.assumed_memory = tuple(
                        f for f in loop_dest.assumed_memory if f not in broken
                    )
                    loop_dest.known = tuple(
                        f for f in loop_dest.known if f not in broken
                    )
                    loop_dest.trace = None
                    loop_dest.next = []
                    return

                old_stack = loop_dest.stack
                beginvars = loop_dest.label.begin_vars
                set_vars = []

                for _, var_idx, val, stack_pos in beginvars:
                    sv = ("setvar", var_idx, stack[stack_pos])
                    set_vars.append(sv)

                if not set_vars:
                    # (a loop of its own, from loop_dest: see set_label)
                    folded, var_list = fold_stacks(
                        old_stack, stack, self.loop_key(loop_dest)
                    )
                    if var_list:
                        node.trace = None
                        node.set_label(loop_dest, tuple(var_list), folded)
                        continue

                    # nothing on the stack changes from one iteration to the
                    # next (the loop keeps what it changes in memory, as Vyper
                    # does): back to the start, rather than a label again
                    node.trace = [("goto", loop_dest, ())]
                    continue

                var_positions = set(stack_pos for *_, stack_pos in beginvars)
                changed = set(
                    idx
                    for idx, (before, after) in enumerate(zip(old_stack, stack))
                    if idx not in var_positions and before != after
                )

                if changed:
                    # The loop variables were found by comparing the stack
                    # before and after the first iteration, and this one
                    # changes something else: e.g. `s += i * i` leaves s at 0
                    # the first time, `if (x[i] > m) m = x[i]` may not change m.
                    # The loop gets explored again, with a variable there too.
                    first = loop_dest.label
                    folded, var_list = stack_vars(
                        first.stack, var_positions | changed, self.loop_key(first)
                    )
                    loop_dest.trace = None
                    loop_dest.next = []
                    loop_dest.set_label(first, tuple(var_list), folded)
                    # the other loops found in this pass may be in the part
                    # of the tree that was just thrown away
                    return

                node.trace = [("goto", loop_dest, tuple(set_vars))]

            else:
                node.trace = None
                node.set_label(loop_dest, tuple(vars), new_stack)

    def merge_branches(self, root):
        """

        When every path going out of an `if` either ends the execution or
        reaches the same jumpdest with the same stack layout, decompile what
        follows that jumpdest once, as the continuation of the `if`, with
        variables standing for the stack values that differ between paths:

            if cond:                        if cond:
                ...                             ...
                jump X (with a on stack)        _1 = a
            else:                           else:
                ...                             ...
                jump X (with b on stack)        _1 = b
                                            X (with _1 on stack)

        Without this, everything after X gets decompiled once for every path
        leading to it, so the number of paths doubles at each such `if`. This
        is what the compiler emits around every external call (was the call
        successful? is there return data? is it shorter than 32 bytes?) and
        a function doing a few of them can't be decompiled at all.

        """
        if self.just_fdests:
            return

        # The exits of a loop: where the exit condition at its head and a
        # break in its body both get to - what follows the loop. That merge
        # is too early to tell when they get there, the path of the body that
        # goes on with the loop being longer (see _merge_unexpanded): it's
        # tried again as the paths get explored - what got to the jumpdest
        # may have been explored further since, a merge cuts the paths there.
        # Without it, what follows the loop is decompiled once more for each
        # break, in the body. The oldest first: of the jumpdests the paths
        # got to, the first one they all go through (the later ones would
        # leave what's between them in the branches).
        for key, (p, jd) in list(self.pending_merges.items()):
            if (
                not self.attached(p, root)
                or self._merge_at(p, jd, loop_exit=True) is not None
            ):
                del self.pending_merges[key]

        unexpanded = find_nodes(root, lambda n: n.trace is None)

        if unexpanded:
            self._merge_unexpanded(root, unexpanded)

    def _merge_unexpanded(self, root, unexpanded):
        """The merges of merge_branches at the jumpdests unexpanded nodes got to."""
        by_jd = {}
        for n in find_nodes(root, lambda n: True):
            by_jd.setdefault(n.jd, []).append(n)

        tried = set()

        for node in unexpanded:
            if node.trace is not None:
                # merged into another node during this pass
                continue

            if getattr(node, "switched", False):
                # a path of a switch (see code_switch), where what it copies
                # is known: merged, it wouldn't be any more
                continue

            if len(by_jd[node.jd]) < 2 or self.ends_execution(node.jd[0]):
                # nothing to merge with, or no point (e.g. a shared revert block)
                continue

            for other in by_jd[node.jd]:
                if other is node:
                    continue

                # the paths to node and other diverged at their closest common
                # ancestor. if it's an `if` that's where they could be merged.
                p = self.common_ancestor(node, other)

                if (
                    p == []
                    or not (p.trace and opcode(p.trace[-1]) == "if")
                    or (id(p), node.jd) in tried
                ):
                    continue

                tried.add((id(p), node.jd))

                merged = self._merge_at(p, node.jd)
                if merged:
                    break
                if merged is None and p.is_label():
                    # at the head of a loop: its exits, maybe (see
                    # merge_branches)
                    self.pending_merges[(id(p), node.jd)] = (p, node.jd)

    @staticmethod
    def attached(node, root):
        """Whether node is still in the tree of root: not in a part thrown
        away since (a loop explored again, a path cut by a merge)."""
        while node is not root:
            parent = node.prev
            if parent == [] or not any(c is node for c in parent.next):
                return False
            node = parent
        return True

    @staticmethod
    def below(node, p):
        """Whether node is p, or a node of the tree below it."""
        while node != [] and node is not p:
            node = node.prev
        return node is p

    @staticmethod
    def in_merged_branch(node, p):
        """
        Whether `node`, below the `if` node `p`, is in a branch of an `if`
        between them whose paths were merged (see _merge_at): a path that
        ends at `node` goes on in the continuation of that `if` (see
        make_trace), not in what follows `p`.
        """
        child, a = node, node.prev
        while a != [] and a is not p:
            tr = a.trace
            if (
                tr
                and len(tr) >= 2
                and opcode(tr[-1]) == "jump"
                and opcode(tr[-2]) == "if"
                and any(child is b for b in tr[-2][2:4])
            ):
                return True
            child, a = a, a.prev
        return False

    @staticmethod
    def common_ancestor(a, b):
        while a != [] and a.depth > b.depth:
            a = a.prev
        while b != [] and b.depth > a.depth:
            b = b.prev
        while a != [] and a is not b:
            a, b = a.prev, b.prev
        return a

    def ends_execution(self, line):
        """True if the basic block starting at `line` can only end the execution."""
        while line in self.lines:
            op = self.lines[line][1]
            if op in (
                "revert",
                "return",
                "stop",
                "invalid",
                "assert_fail",
                "selfdestruct",
            ):
                return True
            if op in ("jump", "jumpi"):
                return False
            line = self.loader.next_line(line)
        return False

    def _merge_at(self, p, jd, loop_exit=False):
        """
        Merge the paths going out of the `if` node `p` at jumpdest `jd`.
        Returns True if it was done, None if it can't be told yet because
        some paths were not explored yet (see merge_branches), False if not.

        loop_exit: only where jd is where the loop p is the head of goes on
        when it's left - its exit condition jumps there, and a path of the
        body goes on with the loop without getting there. (Not a `return` in
        the loop, which gets where the code after the loop returns: solc's
        shared return. Each `return` is better printed where it is.)
        """

        TERMINAL = (
            "revert",
            "return",
            "stop",
            "invalid",
            "assert_fail",
            "selfdestruct",
            "undefined",
            "leave",
        )

        if jd == p.jd or not (p.trace and opcode(p.trace[-1]) == "if"):
            # a loop rather than a merge, or `p` isn't an if any more
            return False

        hits = []
        # the paths that go on with the loop p is the head of
        continues = []

        def visit(start):
            """
            Returns True if every path below `start` either ends, reaches
            `jd`, or loops back (a `continue`: it doesn't get to what follows
            the if either), False if not, None if it's too early to tell.
            """
            result = True
            to_visit = [start]

            while to_visit:
                n = to_visit.pop()

                if n.merged:
                    # goes on in the continuation of the `if` it was merged
                    # at, which will be visited as well - if that's below `p`.
                    # One above it: merged here, the path would go on in what
                    # follows `p` instead (a path that breaks out of two loops
                    # at once)
                    if not self.below(n.merge_point, p):
                        return False
                    continue

                if n.is_label():
                    if n.jd == jd:
                        # the head of a loop, let's keep it that way
                        return False

                    # Otherwise the paths inside the loop body are fair game.
                    # If one of them gets merged before it loops back, the loop
                    # gets peeled: its first iteration stays in the branch, and
                    # what follows is decompiled again from the merge point,
                    # where the loop will be found again. See make_trace for
                    # the label left behind.

                elif n.jd == jd and not self.in_merged_branch(n, p):
                    hits.append(n)
                    continue

                # (a node at jd in a branch of an `if` whose paths were merged
                # already: its path goes on in the continuation of that `if`
                # - with the variables of that merge, set further down -, not
                # in what follows p. It's visited as any other node, and so is
                # that continuation.)

                if n.trace is None:
                    result = None
                    continue

                if n.next:
                    to_visit.extend(reversed(n.next))
                    continue

                op = opcode(n.trace[-1]) if n.trace else None

                if op == "goto" and n.trace[-1][1] in (p, p.label):
                    continues.append(n)

                if op in TERMINAL or op == "goto":
                    continue

                # e.g. 'loop', not yet processed by continue_loops
                result = None

            return result

        _, _, if_true, if_false = p.trace[-1]

        seen = [visit(if_true)]
        if seen[0] is not False:
            seen.append(visit(if_false))
        if False in seen:
            return False
        if None in seen:
            # too early to tell (see merge_branches)
            return None

        if len(hits) < 2:
            return False

        if loop_exit and not (
            continues and any(h is if_true or h is if_false for h in hits)
        ):
            return False

        for h in hits:
            if find_nodes(h, lambda m: m.merged and not self.below(m.merge_point, h)):
                # the path from h was merged with others further down, at an
                # `if` above p (one between them would have kept h from being
                # a hit): cut at h, it wouldn't get to that merge any more
                return False

        stacks = [list(h.stack) for h in hits]
        merged = list(stacks[0])
        setvars = [[] for _ in hits]

        for idx in range(len(merged)):
            vals = [s[idx] for s in stacks]

            if all(v == vals[0] for v in vals):
                continue

            # what a path knows of a value it has of its own: the left of a
            # `&&` that stopped it is false there
            vals = [decided(v, h.known) for v, h in zip(vals, hits)]

            if all(v == vals[0] for v in vals):
                merged[idx] = vals[0]
                continue

            if any(type(v) == int and v in self.loader.jump_dests for v in vals):
                # let's not turn a jump destination into a variable
                return False

            self.counter += 1
            name = f"_{self.counter}"
            merged[idx] = ("var", name)

            for k in range(len(hits)):
                setvars[k].append(("setvar", name, vals[k]))

        logger.debug("merging %i paths at %s", len(hits), hits[0])

        for h, sv in zip(hits, setvars):
            h.trace = sv
            h.next = []
            h.merged = True
            h.merge_point = p

        # what's known at the merge point is what's known on every path
        known = tuple(
            fact for fact in hits[0].known if all(fact in h.known for h in hits[1:])
        )

        # and what only some of them wrote in memory isn't (not even that
        # it's still zero)
        for h in hits:
            for fact in h.known:
                if opcode(fact) == "memory" and fact not in known:
                    unknown = ("memory", fact[1], fact[2], None)
                    if unknown not in known:
                        known += (unknown,)

        node = Node(
            self,
            start=jd[0],
            safe=hits[0].safe,
            stack=tuple(merged),
            condition=True,
            known=known,
        )
        node.set_prev(p)
        p.trace.append(("jump", node))

        return True

    def _run(self, start, safe, stack, condition, known=()):
        logger.debug("VM._run stack=%s", stack)
        self.stack = Stack(stack)
        self.known = known
        self.halted = False
        trace = []

        if self.internal and self.internal_call_at(start, stack):
            return self.internal_call(start, condition)

        i = start
        lines = self.lines

        if i not in lines:
            if type(i) != int:
                return [("undefined", "jump to a parameter computed at runtime", i)]
            elif safe and i == self.loader.last_line:
                # what follows the last instruction (a jumpi's other branch):
                # the code stops there, as at a stop
                return [("stop",)]
            else:
                return [("invalid", "jumdest", i)]

        if not safe and lines[i][1] != "jumpdest":
            return [("invalid", "jump")]

        if lines[i][1] == "jumpdest":
            # This node stands for this jumpdest already: don't create another
            # one for it below, it would look like a one-node loop.
            # (e.g. when this is the fallthrough branch of a jumpi, and the
            # next instruction happens to be a jumpdest)
            i = self.loader.next_line(i)
            if i not in lines:
                # (the last instruction: past it, the code stops)
                return [("stop",)]

        while True:
            try:
                line = lines[i]
            except KeyError:
                # past the last instruction: the code stops there
                trace.append(("stop",))
                return trace

            if self.stack.len() < stack_inputs(line):
                # fewer words on the stack than the instruction uses: the code
                # halts there, as at an invalid instruction
                trace.append(("invalid", "stack underflow"))
                return trace

            res = self.handle_jumps(trace, line, condition)
            if res is not None:
                return res

            if line[1] == "jumpdest":
                n = Node(
                    self,
                    start=i,
                    safe=False,
                    stack=tuple(self.stack.stack),
                    condition=condition,
                    known=self.known,
                )
                logger.debug("jumpdest %s", n)
                trace.append(("jump", n))
                return trace

            else:
                self.apply_stack(trace, line)
                if self.halted:
                    # (the instruction ended the path, see returndatacopy)
                    return trace

            i = self.loader.next_line(i)

        assert False

    def code_switch(self, i):
        """
        A codecopy of a few bytes of the code from where one of few numbers
        says - x % n: Vyper's dispatcher (0.3.10 on) copies from a table
        where the code of a selector is, and jumps there. One path for each
        of the values, where the copy is from a known place: `if x % n ==
        0: ... else: if x % n == 1: ...`, the last without its test (the
        ifs in between are made here, see link_if). None if it's not that,
        or when x % n is known.
        """
        stack = self.stack.stack
        if len(stack) < 3:
            return None
        mem_pos, code_pos, size = stack[-1], stack[-2], stack[-3]
        if not (type(mem_pos) is int and type(size) is int and size <= 32):
            return None

        if type(code_pos) is int:
            return None
        switch = switch_values(code_pos)
        if switch is None:
            return None
        m, values = switch

        cases = [k for k in values if is_known(("eq", m, k), self.known) is not False]
        for fact in self.known:
            if match(fact, ("eq", m, ":int:k")):
                cases = [fact[2]]

        def value(k):
            # the stack where x % n is k
            return [algebra.simplify(replace(e, m, k)) for e in stack]

        if len(cases) < 2:
            if cases:
                self.stack.stack = value(cases[0])
            return None

        nodes = []
        for k in cases:
            cond = ("eq", m, k)
            n = Node(
                self,
                start=i,
                safe=True,
                stack=tuple(value(k)),
                condition=cond,
                known=self.known + (cond,),
            )
            # (not to be merged again, see merge_branches)
            n.switched = True
            nodes.append((cond, n))

        rest = nodes[-1][1]
        for cond, n in reversed(nodes[1:-1]):
            rest = Node(
                self,
                start=("switch", i, cond[2]),
                safe=True,
                stack=tuple(stack),
                trace=[("if", cond, n, rest)],
                known=self.known,
            )
        return [("if", nodes[0][0], nodes[0][1], rest)]

    def handle_jumps(self, trace, line, condition):
        i, op = line[0], line[1]
        stack = self.stack

        if "--explain" in sys.argv and op in (
            "jump",
            "jumpi",
            "selfdestruct",
            "stop",
            "return",
            "invalid",
            "assert_fail",
            "revert",
        ):
            trace.append(C.asm(f"       {stack}"))
            trace.append("")
            trace.append(f"[{line[0]}] {C.asm(op)}")

        if op in (
            "jump",
            "jumpi",
            "selfdestruct",
            "stop",
            "return",
            "invalid",
            "assert_fail",
            "revert",
        ):
            logger.debug("[%s] %s", i, op)

        if op == "codecopy" and (switch := self.code_switch(i)):
            return trace + switch

        if op == "jump":
            target = stack.pop()

            if target == RETURN_ADDRESS:
                # the end of the internal function decompiled apart: what's
                # on the stack, its results
                trace.append(("leave", tuple(self.stack.stack)))
                return trace

            n = Node(
                self,
                start=target,
                safe=False,
                stack=tuple(self.stack.stack),
                condition=condition,
                known=self.known,
            )

            trace.append(("jump", n))
            return trace

        elif op == "jumpi":
            target = stack.pop()
            if_condition = simplify_bool(stack.pop())

            tuple_stack = tuple(self.stack.stack)
            n_true = Node(
                self,
                start=target,
                safe=False,
                stack=tuple_stack,
                condition=if_condition,
                known=self.known + (if_condition,),
            )
            n_false = Node(
                self,
                start=self.loader.next_line(i),
                safe=True,
                stack=tuple_stack,
                condition=is_zero(if_condition),
                known=self.known + (is_zero(if_condition),),
            )

            if self.just_fdests and (test := selector_test(if_condition)):
                fx_hash, taken = test
                if taken:
                    if (
                        type(target) == int
                        and self.lines.get(target, (None, None))[1] != "jumpdest"
                    ):
                        # where a jump can't go: the function is that invalid
                        # jump, as it is from where no instruction is
                        target = -1
                    n_true.trace = [("funccall", fx_hash, target, tuple_stack)]
                else:
                    n_false.trace = [("funccall", fx_hash, n_false.start, tuple_stack)]

            bool_condition = arithmetic.eval_bool(if_condition, symbolic=False)

            if bool_condition is None:
                bool_condition = is_known(if_condition, self.known)

            if bool_condition is not None:
                if bool_condition:
                    trace.append(("jump", n_true))
                    return trace  # res, False

                else:
                    trace.append(("jump", n_false))
                    return trace

            trace.append(
                (
                    "if",
                    if_condition,
                    n_true,
                    n_false,
                )
            )
            logger.debug("jumpi -> if %s", trace[-1])
            return trace

        elif op in ["return", "revert"]:
            p = stack.pop()
            n = stack.pop()

            if n == 0 and op == "return":
                # returning nothing is the same as stopping, while
                # ("return", 0) returns a word, 0
                trace.append(("stop",))
            elif n == 0:
                # likewise, ("revert", 0) reverts with a word, 0
                trace.append((op, None))
            else:
                return_data = mem_load(p, n)
                trace.append(
                    (
                        op,
                        return_data,
                    )
                )

            return trace

        elif op == "selfdestruct":
            trace.append(
                (
                    "selfdestruct",
                    stack.pop(),
                )
            )
            return trace

        elif op in ["stop", "assert_fail", "invalid"]:
            trace.append((op,))
            return trace

        elif op == "UNKNOWN":
            trace.append(("invalid",))
            return trace

        return None

    def snapshot(self, trace, op, target=None):
        """
        Before an instruction that changes the state, what was read from it
        and is still on the stack gets a variable: afterwards, the expression
        of the read (e.g. ("storage", 256, 0, 4)) stands for what's there
        then, not for what was read.

        e.g. arr.push(x) reads the length, stores it plus one, and stores x
        at the old length.
        """
        reads = []
        # (and what the memory was written with: mem[64] = x + returndatasize
        # is x + the size of the data a call returned, until it's written)
        values = [fact[3] for fact in self.known if opcode(fact) == "memory"]
        for item in self.stack.stack + [v for v in values if v is not None]:
            for r in changed_reads(item, op, target):
                if r not in reads:
                    reads.append(r)

        # All read at once, before the instruction. A read in another one
        # (the length of an array in the slot of its last element) gets its
        # variable first, which the other is then read with; and on the
        # stack, the outermost read is the one replaced - one after the
        # other, the inner one replaced first, the other one wouldn't be
        # found any more, and would be read after the instruction.
        names = {}

        def with_names(exp):
            return replace_f_stop(
                exp, lambda e: names.get(e) if type(e) in (tuple, str) else None
            )

        for r in sorted(reads, key=exp_size):
            self.counter += 1
            vname = f"_{self.counter}"
            trace(("setvar", vname, with_names(r)))
            names[r] = ("var", vname)
            if (b := max_value_bits(r)) <= 128:
                # what it was is as small as that (a size, say): for where
                # memory is written (see bound), not to decide conditions
                self.known += (("var_bits", ("var", vname), b),)

        self.stack.stack = [with_names(item) for item in self.stack.stack]
        self.known = tuple(
            (
                fact[:3] + (with_names(fact[3]),)
                if opcode(fact) == "memory" and fact[3] is not None
                else fact
            )
            for fact in self.known
        )

    def apply_stack(self, ret, line):
        def trace(exp, *format_args):
            try:
                logger.debug("Trace: %s", str(exp).format(*format_args))
            except Exception:
                pass

            if type(exp) == str:
                ret.append(exp.format(*format_args))
            else:
                ret.append(exp)
                if opcode(exp) == "setmem":
                    self.known = write_memory(self.known, exp[1], exp[2])

        stack = self.stack

        op = line[1]

        previous_len = stack.len()

        if op in ("sstore", "tstore"):
            self.snapshot(trace, op, stack.stack[-1])
        elif op in STATE_CHANGING_OPS and op != "selfdestruct":
            self.snapshot(trace, op)

        if op == "sstore":
            self.known = forget(self.known, ("storage",))
        elif op == "tstore":
            self.known = forget(self.known, ("tload",))
        elif op in STATE_CHANGING_OPS:
            # Anything can happen in the callee, including reentering this
            # contract and changing its storage.
            self.known = forget(self.known, VOLATILE)

        if "--verbose" in sys.argv or "--explain" in sys.argv:
            trace(C.asm("       " + str(stack)))
            trace("")

            if "push" not in op and "dup" not in op and "swap" not in op:
                trace("[{}] {}", line[0], C.asm(op))
            else:
                if type(line[2]) == str:
                    trace("[{}] {} {}", line[0], C.asm(op), C.asm(" ”" + line[2] + "”"))
                elif line[2] > 0x1000000000:
                    trace("[{}] {} {}", line[0], C.asm(op), C.asm(hex(line[2])))
                else:
                    trace("[{}] {} {}", line[0], C.asm(op), C.asm(str(line[2])))

        param = 0
        if len(line) > 2:
            param = line[2]

        if op == "and":
            left, right = stack.pop(), stack.pop()
            stack.append(and_op(left, right, self.known))

        elif op in [
            "exp",
            "eq",
            "div",
            "lt",
            "gt",
            "slt",
            "sgt",
            "mod",
            "xor",
            "signextend",
            "smod",
            "sdiv",
        ]:
            stack.append(
                arithmetic.eval(
                    (
                        op,
                        stack.pop(),
                        stack.pop(),
                    )
                )
            )

        elif op[:4] == "push":
            stack.append(param)

        elif op == "pop":
            stack.pop()

        elif op == "dup":
            stack.dup(param)

        elif op == "mul":
            stack.append(mul_op(stack.pop(), stack.pop()))

        elif op == "or":
            stack.append(or_op(stack.pop(), stack.pop()))

        elif op == "add":
            stack.append(add_op(stack.pop(), stack.pop()))

        elif op == "sub":
            left = stack.pop()
            right = stack.pop()

            if type(left) == int and type(right) == int:
                stack.append(arithmetic.sub(left, right))
            else:
                stack.append(sub_op(left, right))

        elif op in ["mulmod", "addmod"]:
            stack.append((op, stack.pop(), stack.pop(), stack.pop()))

        elif op == "shl":
            off = stack.pop()
            exp = stack.pop()
            if all_concrete(off, exp):
                stack.append(exp << off if off < 256 else 0)
            else:
                stack.append(shl_op(exp, off))

        elif op == "shr":
            off = stack.pop()
            exp = stack.pop()
            if all_concrete(off, exp):
                stack.append(exp >> off)
            else:
                stack.append(shr_op(exp, off))

        elif op == "sar":
            off = stack.pop()
            exp = stack.pop()
            if all_concrete(off, exp):
                sign = exp & (1 << 255)
                if off >= 256:
                    if sign:
                        stack.append(2**256 - 1)
                    else:
                        stack.append(0)
                else:
                    shifted = exp >> off
                    if sign:
                        shifted |= (2**256 - 1) << (256 - off)
                    stack.append(shifted)
            else:
                # left as it is: no mask copies the sign bit into the bits
                # shifted in
                stack.append(("sar", off, exp))

        elif op == "iszero" and type(stack.peek()) == int:
            # (a number: Stack.cleanup would make it a ("bool", n), which the
            # algebra doesn't compute with - a jump to iszero(0) * 12 + 13)
            stack.append(int(stack.pop() == 0))

        elif op in ["not", "iszero", "clz"]:
            stack.append((op, stack.pop()))

        elif op == "sha3":
            p = stack.pop()
            n = stack.pop()
            res = mem_load(p, n)

            self.counter += 1
            vname = f"_{self.counter}"
            vval = (
                "sha3",
                res,
            )

            trace(("setvar", vname, vval))
            stack.append(("var", vname))

        elif op == "calldataload":
            stack.append(
                (
                    "cd",
                    stack.pop(),
                )
            )

        elif op == "byte":
            # the idx-th byte of val, the most significant one first - 0 for
            # an idx of 32 or more: as it is when idx isn't known (a mask at
            # an offset computed from it wouldn't say that)
            idx = stack.pop()
            val = stack.pop()
            if type(idx) is not int:
                stack.append(("byte", idx, val))
            elif idx >= 32:
                stack.append(0)
            else:
                off = 248 - 8 * idx
                stack.append(mask_value(val, 8, off, -off))

        elif op == "selfbalance":
            stack.append(
                (
                    "balance",
                    "address",
                )
            )

        elif op == "balance":
            addr = stack.pop()
            if opcode(addr) == "mask_shl" and addr[1:4] == (160, 0, 0):
                # (the address: the low 160 bits of the word, as printed)
                addr = addr[4]
            stack.append(("balance", addr))

        elif op == "swap":
            stack.swap(param)

        elif op[:3] == "log":
            p = stack.pop()
            s = stack.pop()
            topics = []
            param = int(op[3])
            for i in range(param):
                el = stack.pop()
                topics.append(el)

            trace(
                (
                    "log",
                    mem_load(p, s),
                )
                + tuple(topics)
            )

        elif op == "sload":
            sloc = stack.pop()
            stack.append(("storage", 256, 0, sloc))

        elif op == "sstore":
            sloc = stack.pop()
            val = stack.pop()
            trace(("store", 256, 0, sloc, val))

        elif op == "tload":
            stack.append(("tload", stack.pop()))

        elif op == "tstore":
            key = stack.pop()
            val = stack.pop()
            trace(("tstore", key, val))

        elif op == "mload":
            memloc = stack.pop()

            if (val := read_memory(self.known, memloc)) is not None:
                stack.append(val)
            else:
                self.counter += 1
                vname = f"_{self.counter}"
                trace(("setvar", vname, ("mem", ("range", memloc, 32))))
                stack.append(("var", vname))

        elif op == "mstore":
            memloc = stack.pop()
            val = stack.pop()
            trace(
                (
                    "setmem",
                    ("range", memloc, 32),
                    val,
                )
            )

        elif op == "mstore8":
            memloc = stack.pop()
            val = stack.pop()

            # the lowest byte of val, into one byte of memory
            trace(
                (
                    "setmem",
                    ("range", memloc, 1),
                    mask_value(val, 8),
                )
            )

        elif op == "mcopy":
            dst = stack.pop()
            src = stack.pop()
            size = stack.pop()

            if size != 0:
                trace(("setmem", ("range", dst, size), ("mem", ("range", src, size))))

        elif op == "extcodecopy":
            addr = stack.pop()
            mem_pos = stack.pop()
            code_pos = stack.pop()
            data_len = stack.pop()

            trace(
                (
                    "setmem",
                    ("range", mem_pos, data_len),
                    ("extcodecopy", addr, ("range", code_pos, data_len)),
                )
            )

        elif op == "codecopy":
            mem_pos = stack.pop()
            call_pos = stack.pop()
            data_len = stack.pop()

            binary = self.loader.binary
            if (
                (type(call_pos), type(data_len)) == (int, int)
                and 0 <= call_pos
                and 0 <= data_len <= MAX_CODECOPY_SIZE
            ):
                # the bytes of the code, and zeroes past its end: solc fills a
                # new array with them, copying from codesize
                res = 0
                for i in range(call_pos, min(call_pos + data_len, len(binary))):
                    res = res << 8
                    res += binary[i]
                res <<= 8 * max(0, call_pos + data_len - max(call_pos, len(binary)))
                trace(
                    ("setmem", ("range", mem_pos, data_len), res)
                )  # ('bytes', data_len, res)))

            else:
                trace(
                    (
                        "setmem",
                        ("range", mem_pos, data_len),
                        (
                            "code.data",
                            call_pos,
                            data_len,
                        ),
                    )
                )

        elif op == "codesize":
            stack.append(len(self.loader.binary))

        elif op == "calldatacopy":
            mem_pos = stack.pop()
            call_pos = stack.pop()
            data_len = stack.pop()

            if data_len != 0:
                call_data = ("call.data", call_pos, data_len)
                #                call_data = mask_op(('call.data', bits(add_op(data_len, call_pos))), size=bits(data_len), shl=bits(call_pos))
                trace(("setmem", ("range", mem_pos, data_len), call_data))

        elif op == "returndatacopy":
            mem_pos = stack.pop()
            ret_pos = stack.pop()
            data_len = stack.pop()

            # a copy of more than there is halts, as an invalid opcode does
            end = add_op(ret_pos, data_len)
            size = 0 if RETURNDATASIZE_ZERO in self.known else "returndatasize"
            beyond = ("lt", size, end)
            try:
                # (sizes: far from 2**256)
                decided = lt_op(size, end)
            except CannotCompare:
                decided = is_known(beyond, self.known)
            if decided is None:
                trace(("if", beyond, [("invalid", "returndatacopy")], []))
                self.known += (is_zero(beyond),)
            elif decided:
                trace(("invalid", "returndatacopy"))
                self.halted = True
                return

            if data_len != 0:
                return_data = ("ext_call.return_data", ret_pos, data_len)
                #                return_data = mask_op(('ext_call.return_data', bits(add_op(data_len, ret_pos))), size=bits(data_len), shl=bits(ret_pos))
                trace(("setmem", ("range", mem_pos, data_len), return_data))

        elif op == "call":
            self.handle_call(op, trace)

        elif op == "staticcall":
            self.handle_call(op, trace)

        elif op == "delegatecall":
            gas = stack.pop()
            addr = stack.pop()

            arg_start = stack.pop()
            arg_len = stack.pop()
            ret_start = stack.pop()
            ret_len = stack.pop()

            call_trace = (
                "delegatecall",
                gas,
                addr,
            )  # arg_start, arg_len, ret_start, ret_len)

            call_trace += self.call_data(arg_start, arg_len)

            trace(call_trace)

            self.call_len = ret_len
            stack.append("delegate.return_code")

            self.output_write(
                trace, lambda n: ("ext_call.return_data", 0, n), ret_start, ret_len
            )

        elif op == "callcode":
            gas = stack.pop()
            addr = stack.pop()
            value = stack.pop()

            arg_start = stack.pop()
            arg_len = stack.pop()
            ret_start = stack.pop()
            ret_len = stack.pop()

            call_trace = (
                "callcode",
                gas,
                addr,
                value,
            )

            call_trace += self.call_data(arg_start, arg_len)

            trace(call_trace)

            self.call_len = ret_len
            stack.append("callcode.return_code")

            self.output_write(
                trace, lambda n: ("ext_call.return_data", 0, n), ret_start, ret_len
            )

        elif op == "create":
            wei, mem_start, mem_len = stack.pop(), stack.pop(), stack.pop()

            call_trace = ("create", wei)

            code = mem_load(mem_start, mem_len)
            call_trace += (code,)

            trace(call_trace)

            stack.append("create.new_address")

        elif op == "create2":
            wei, mem_start, mem_len, salt = (
                stack.pop(),
                stack.pop(),
                stack.pop(),
                stack.pop(),
            )

            call_trace = ("create2", wei, ("mem", ("range", mem_start, mem_len)), salt)

            trace(call_trace)

            stack.append("create2.new_address")

        elif op == "pc":
            stack.append(line[0])

        elif op == "msize":
            self.counter += 1
            vname = f"_{self.counter}"
            trace(("setvar", vname, "msize"))
            stack.append(("var", vname))

        elif op in ("extcodesize", "extcodehash", "blockhash", "blobhash"):
            stack.append(
                (
                    op,
                    stack.pop(),
                )
            )

        elif op in [
            "callvalue",
            "caller",
            "address",
            "number",
            "gas",
            "origin",
            "timestamp",
            "chainid",
            "difficulty",
            "gasprice",
            "coinbase",
            "gaslimit",
            "calldatasize",
            "returndatasize",
            "basefee",
            "blobbasefee",
        ]:
            if op == "returndatasize" and RETURNDATASIZE_ZERO in self.known:
                stack.append(0)
            else:
                stack.append(op)

        else:
            # TODO: Maybe raise an error directly?
            assert op not in [
                "jump",
                "jumpi",
                "revert",
                "return",
                "stop",
                "jumpdest",
                "UNKNOWN",
            ]

        if stack.len() - previous_len != opcode_dict.stack_diffs[op]:
            logger.error("line: %s", line)
            logger.error("stack: %s", stack)
            logger.error(
                "expected %s, got %s stack diff",
                opcode_dict.stack_diffs[op],
                stack.len() - previous_len,
            )
            assert False, f"opcode {op} not processed correctly"

        if (
            op not in ("dup", "swap")
            and not op.startswith("push")
            and type(stack.peek()) == tuple
            and exp_size(stack.peek()) > MAX_EXP_SIZE
        ):
            # Some arithmetic (e.g. the Newton iterations in mulDiv) makes
            # expressions grow exponentially, and the algebra with them.
            # Give the big ones a name, as is done for memory reads.
            self.counter += 1
            vname = f"_{self.counter}"
            trace(("setvar", vname, stack.pop()))
            stack.append(("var", vname))

        stack.cleanup()

    def output_write(self, trace, data, ret_start, ret_len):
        """
        What a call returns, written to the ret_len bytes of memory at
        ret_start it gives for it: min(ret_len, return_data.size) of them -
        the others stay as they were, a callee may return less (a token of
        before ERC-20 returns nothing). data: the return data, (name, 0, n)
        its n first bytes.

        Where the memory is known to be zeroes (a solidity ecrecover), that
        is the return data padded with zeroes; otherwise the write is of the
        bytes returned, and a read of it where return_data.size >= ret_len
        is known (the check solc makes after a call) is of the return data
        (see simplify.apply_constraint).
        """
        try:
            if not lt_op(0, ret_len):
                return
        except CannotCompare:
            pass
        if known_zeroes(self.known, ret_start, ret_len):
            trace(("setmem", ("range", ret_start, ret_len), data(ret_len)))
            return
        size = ("min", ret_len, "returndatasize")
        trace(("setmem", ("range", ret_start, size), data(size)))

    def call_data(self, arg_start, arg_len):
        """
        (selector, params): the data of a call, the arg_len bytes of memory at
        arg_start - its 4 first bytes and the others, where it has 4 bytes or
        more for sure; else (None, all of them): the data of a param (a
        `x.call(data)`) can be shorter, and then it has no selector. (None,
        None): no data.
        """
        if arg_len == 0:
            return None, None
        if arg_len == 4:
            return mem_load(arg_start, 4), None
        if type(arg_len) is int:
            split = arg_len > 4
        else:
            # (as the word it is: a sum that can't wrap)
            lo, hi = value_range(arg_len, known_bounds(self.known))
            split = lo >= 4 and hi < 2**256
        if split:
            return mem_load(arg_start, 4), mem_load(
                add_op(arg_start, 4), sub_op(arg_len, 4)
            )
        return None, mem_load(arg_start, arg_len)

    def handle_call(self, op, trace):
        stack = self.stack

        gas = stack.pop()
        addr = stack.pop()
        if op == "call":
            wei = stack.pop()
        else:
            assert op == "staticcall"
            wei = 0

        arg_start = stack.pop()
        arg_len = stack.pop()
        ret_start = stack.pop()
        ret_len = stack.pop()

        if addr == 4:  # Identity
            # its return data is what it's given: as much of it as there's
            # room for is written (and it's what the next returndatacopy
            # copies)
            args = mem_load(arg_start, arg_len)
            trace(("precompiled", "copied", "identity", args))
            if ret_len == arg_len:
                size = ret_len
            elif type(ret_len) is int and type(arg_len) is int:
                size = min(ret_len, arg_len)
            else:
                size = ("min", ret_len, arg_len)
            try:
                if lt_op(0, size):
                    trace(
                        (
                            "setmem",
                            ("range", ret_start, size),
                            mem_load(arg_start, size),
                        )
                    )
            except CannotCompare:
                trace(("setmem", ("range", ret_start, size), mem_load(arg_start, size)))

            stack.append("memcopy.success")

        elif type(addr) == int and addr in precompiled:
            args = mem_load(arg_start, arg_len)

            # a name for each result: signer, signer2...
            base = precompiled_var_names[addr]
            count = self.precompile_results.get(base, 0) + 1
            self.precompile_results[base] = count
            var_name = base if count == 1 else f"{base}{count}"

            trace(("precompiled", var_name, precompiled[addr], args))
            if ret_len == 32 and known_zeroes(self.known, ret_start, 32):
                # the word it returns, or zeroes (ecrecover of a wrong
                # signature returns nothing): the result
                trace(("setmem", ("range", ret_start, 32), ("var", var_name)))
            else:
                self.output_write(
                    trace, lambda n: ("ext_call.return_data", 0, n), ret_start, ret_len
                )

            stack.append("{}.result".format(precompiled[addr]))

        else:
            assert op in ("call", "staticcall")
            call_trace = (
                op,
                gas,
                addr,
                wei,
            )

            call_trace += self.call_data(arg_start, arg_len)

            trace(call_trace)
            #           trace(('comment', mem_load(arg_start, arg_len)))

            self.call_len = ret_len

            stack.append("ext_call.success")

            self.output_write(
                trace, lambda n: ("ext_call.return_data", 0, n), ret_start, ret_len
            )
