import collections
import json
import logging
import re
from copy import deepcopy

from panoramix.core.arithmetic import simplify_bool
from panoramix.core.masks import mask_to_type
from panoramix.core.memloc import byte_elements, keep_width
from panoramix.matcher import Any, match
from panoramix.prettify import (
    BUILTIN_NAMES,
    explain_text,
    pprint_logic,
    prettify,
    pretty_memory,
    set_names,
)
from panoramix.utils.helpers import (
    COLOR_BLUE,
    COLOR_BOLD,
    COLOR_GRAY,
    COLOR_GREEN,
    COLOR_HEADER,
    COLOR_OKGREEN,
    COLOR_UNDERLINE,
    COLOR_WARNING,
    ENDC,
    FAIL,
    C,
    EasyCopy,
    color,
    find_f,
    find_f_list,
    internal_name,
    opcode,
)
from panoramix.utils.signatures import (
    calldata_params,
    clear_func,
    get_abi_name,
    get_func_name,
    get_func_params,
    set_func,
    set_func_params,
    set_func_params_if_none,
)

logger = logging.getLogger(__name__)


def find_parents(exp, child):
    if type(exp) not in (list, tuple):
        return []

    res = []

    for e in exp:
        if e == child:
            res.append(exp)
        res.extend(find_parents(e, child))

    return res


def word(exp):
    """A word of data (see memloc, "bytes"), as its value."""
    if m := match(exp, ("bytes", 32, ":val")):
        return m.val
    return exp


def disjuncts(cond):
    """The conditions cond holds where one of them does: of `a | b`, a and b."""
    if opcode(cond) == "or" and len(cond) > 1:
        return [d for e in cond[1:] for d in disjuncts(simplify_bool(e))]
    return [cond]


# What a function returning a constant may read: where the contract is.
CONSTANT_SYMBOLS = ("address", "codesize")

# The operations that read what can change: the calldata, the state, the
# environment.
VARYING_READS = (
    "cd",
    "call.data",
    "storage",
    "tload",
    "balance",
    "extcodesize",
    "extcodehash",
    "extcodecopy",
    "blockhash",
    "blobhash",
)


def varies(exp):
    """True if exp reads something that isn't a constant of the contract."""
    if type(exp) is str:
        return exp not in CONSTANT_SYMBOLS

    if type(exp) is list:
        return any(varies(e) for e in exp)

    if type(exp) is tuple and exp:
        if exp[0] in VARYING_READS:
            return True
        if exp[0] == "var":
            # its value is set in the trace, and checked there
            return False
        return any(varies(e) for e in exp[1:])

    return False


class Function(EasyCopy):
    def __init__(self, hash, trace):
        self.hash = hash
        self.name = get_func_name(hash)
        self.color_name = get_func_name(hash, add_color=True)
        self.abi_name = get_abi_name(hash)

        self.const = None
        self.read_only = None
        self.payable = None
        # what the function does when it's sent ether, if it isn't payable
        self.value_fails = None

        self.hash = hash

        self.trace = deepcopy(trace)
        self.orig_trace = deepcopy(self.trace)

        # A list of (kind, name)
        self.inferred_params = self.make_params()

        if "unknown" in self.name:
            self.make_names()

        self.trace = self.cleanup_masks(self.trace)
        self.ast = None

        self.analyse()

        assert self.payable is not None

        self.is_regular = self.const is None and self.getter is None

    def cleanup_masks(self, trace):
        """
        A param as it is once it's checked: after `require _param1 ==
        address(_param1)`, address(_param1) is _param1 - the calldata has no
        more than an address there. Not before, nor without the check (solc
        < 0.5 has none): a param is the word of calldata, whatever its type.
        """

        def validated(cond):
            """(cd, x) if cond, true, reverts unless cd == x"""
            if m := match(cond, ("iszero", ("eq", ":a", ":b"))):
                for cd, x in ((m.a, m.b), (m.b, m.a)):
                    if opcode(cd) == "cd" and clean_form(x, cd):
                        return cd, x
            if (
                (m := match(cond, ("mask_shl", ":int:size", ":int:off", 0, ":cd")))
                and opcode(m.cd) == "cd"
                and m.size + m.off == 256
                and m.off > 0
            ):
                # the bits above the type are 0
                return m.cd, ("mask_shl", m.off, 0, 0, m.cd)
            return None

        def clean_form(x, cd):
            return (
                match(x, ("mask_shl", Any, 0, 0, cd))
                or match(x, ("bool", cd))
                or match(x, ("signextend", Any, cd))
                or match(x, ("mask_shl", Any, Any, 0, cd))
            )

        def reverts(branch):
            return len(branch) == 1 and branch[0] in (("revert", None), ("invalid",))

        def subst(exp, known):
            if not known:
                return exp
            if type(exp) not in (list, tuple):
                return exp
            if type(exp) == tuple and exp in known:
                return known[exp]
            res = type(exp)(subst(e, known) for e in exp)
            if type(exp) == tuple:
                # an element of bytes stays as wide as it's written
                for i in byte_elements(exp):
                    if res[i] != exp[i]:
                        res = res[:i] + (keep_width(exp[i], res[i]),) + res[i + 1 :]
            return res

        def rem(trace, known):
            res = []
            for idx, line in enumerate(trace):
                if m := match(line, ("if", ":cond", ":if_true", ":if_false")):
                    cond = subst(m.cond, known)
                    if_true, if_false = m.if_true, m.if_false
                    v = None
                    if reverts(if_true):
                        v = validated(m.cond)
                        if v:
                            if_false = rem(if_false, {**known, v[1]: v[0]})
                            if_true = rem(if_true, known)
                    elif reverts(if_false) and (
                        mm := match(m.cond, ("eq", ":a", ":b"))
                    ):
                        for cd, x in ((mm.a, mm.b), (mm.b, mm.a)):
                            if opcode(cd) == "cd" and clean_form(x, cd):
                                v = cd, x
                                if_true = rem(if_true, {**known, x: cd})
                                if_false = rem(if_false, known)
                                break
                    if not v:
                        if_true = rem(if_true, known)
                        if_false = rem(if_false, known)
                    res.append(("if", cond, if_true, if_false))
                elif opcode(line) == "while":
                    _, cond, path, jd, setvars = line
                    res.append(
                        (
                            "while",
                            subst(cond, known),
                            rem(path, known),
                            jd,
                            subst(setvars, known),
                        )
                    )
                else:
                    res.append(subst(line, known))
            return res

        return rem(trace, {})

    def rename_params(self, storage_names):
        """
        The params (of the abi) that have the name of a storage variable
        (`owner`, which would read as both), or of something the EVM gives
        (`caller`), have it after a _ (`_owner`).
        """
        inputs = get_func_params(self.hash)
        if not inputs:
            return
        taken = BUILTIN_NAMES | frozenset(storage_names)
        names = {p["name"] for p in inputs}
        renames = {}
        for p in inputs:
            # (once for a name two of them have)
            if p["name"] in taken and p["name"] not in renames:
                new = "_" + p["name"]
                while new in taken or new in names:
                    new = "_" + new
                renames[p["name"]] = new
                names.add(new)
        if not renames:
            return

        set_func_params(
            self.hash, [dict(p, name=renames.get(p["name"], p["name"])) for p in inputs]
        )

        def renamed(name):
            # (desc of desc.srcToken, amounts of amounts[2])
            base = re.match(r"\w*", name).group(0)
            return renames[base] + name[len(base) :] if base in renames else name

        self.inferred_params = {
            k: (t, renamed(n)) for k, (t, n) in self.inferred_params.items()
        }
        self.name = get_func_name(self.hash)
        self.color_name = get_func_name(self.hash, add_color=True)

    def make_names(self):
        new_name = self.name.split("(")[0]

        self.name = "{}({})".format(
            new_name,
            ", ".join((p[0] + " " + p[1]) for p in self.inferred_params.values()),
        )
        self.color_name = "{}({})".format(
            new_name,
            ", ".join(
                (p[0] + " " + COLOR_GREEN + p[1] + ENDC)
                for p in self.inferred_params.values()
            ),
        )

        self.abi_name = "{}({})".format(
            new_name, ",".join(p[0] for p in self.inferred_params.values())
        )

    def ast_length(self):
        if self.trace is None:
            return 0, 0
        return len((self.print().split("\n"))), len(self.print())

    def priority(self):
        # sorts functions in this order:
        # - self-destructs
        # - (read-only? would be nice, but some read-only funcs can be very long, e.g. etherdelta)
        # - length

        if self.trace is None:
            return 0

        if "selfdestruct" in str(self.trace):
            return -1

        else:
            return self.ast_length()[1]

    def make_params(self):
        """
        figures out parameter types from the decompiled function code.

        does so by looking at all 'cd'/calldata occurences and figuring out
        how they are accessed - are they masked? are they used as pointers?

        """

        params = get_func_params(self.hash)
        if params is not None:
            # (none for a function the abi says has none: a word of the
            # calldata it reads isn't a param then)
            res = calldata_params(params)
        else:
            # good testing: solidstamp, auditContract
            # try to find all the references to parameters and guess their types

            def reads(exp, res):
                # (a mask of a read is that read - and not, as find_f_list
                # would have it, the read inside it as well, unmasked)
                if match(exp, ("mask_shl", Any, Any, Any, ("cd", Any))):
                    res.append(exp)
                    rest = exp[1:4] + exp[4][1:]
                elif match(exp, ("cd", Any)):
                    res.append(exp)
                    rest = exp[1:]
                elif type(exp) in (list, tuple):
                    rest = exp
                else:
                    return res
                for e in rest:
                    reads(e, res)
                return res

            occurences = reads(self.trace, [])

            # the params (by their position in the calldata) and the sizes of
            # the masks applied to them, None when used as they are
            uses = {}
            pointers = set()
            for o in occurences:
                if m := match(o, ("mask_shl", ":size", ":off", Any, ("cd", ":idx"))):
                    idx = m.idx
                    # only the lowest bits tell a type: taking a byte out of
                    # the middle doesn't make a param an uint8
                    size = m.size if m.off == 0 else None

                elif m := match(o, ("cd", ":idx")):
                    idx = m.idx
                    size = None

                if type(idx) is not int:
                    # an element of an array, or its length: what it's read
                    # relatively to is a pointer to it
                    pointers.update(
                        find_f_list(
                            idx, lambda e: [e[1]] if match(e, ("cd", int)) else []
                        )
                    )
                    continue

                if idx == 0:
                    continue

                uses.setdefault(idx, []).append(size)

            for idx in uses:
                if (idx - 4) % 32 != 0:
                    logger.warning("unusual cd (not aligned)")
                    return {}

            sizes = {}
            for idx, idx_sizes in uses.items():
                if idx in pointers:
                    sizes[idx] = -1
                elif valid := self.validation(idx):
                    sizes[idx] = valid
                elif None not in idx_sizes and all(
                    type(s) is int and 0 < s <= 256 for s in idx_sizes
                ):
                    # masked everywhere, as the compilers did before
                    # validating the params (by masks of known sizes)
                    sizes[idx] = min(idx_sizes)
                else:
                    sizes[idx] = 256

            for idx in pointers - set(sizes):
                if type(idx) is int and idx > 0 and (idx - 4) % 32 == 0:
                    sizes[idx] = -1

            # for every idx check if it's a bool by any chance
            for idx in sizes:
                li = find_parents(self.trace, ("cd", idx))
                for e in li:
                    if opcode(e) not in ("bool", "if", "iszero"):
                        break

                    if m := match(e, ("mask_shl", Any, ":off", Any, ":val")):
                        off, val = m.off, m.val
                        assert val == ("cd", idx)
                        if off != 0:
                            sizes[idx] = -2  # it's a tuple!
                else:
                    sizes[idx] = 1

            res = {}
            count = 1
            for k in sizes:
                if type(k) != int:
                    logger.warning(f"unusual calldata reference {k}")
                    return {}

            for idx in sorted(sizes.keys()):
                size = sizes[idx]

                if type(size) is str:
                    kind = size
                elif size == -2:
                    kind = "tuple"
                elif size == -1:
                    kind = "array"
                elif size == 1:
                    kind = "bool"
                elif type(size) is int and 0 < size < 256 and size % 8 == 0:
                    # (of a whole number of bytes, its type: uint24, not the
                    # uint32 that's the next one mask_to_type knows)
                    kind = mask_to_type(size) or f"uint{size}"
                else:
                    kind = mask_to_type(size, force=True)

                assert kind != None, size

                res[idx] = (kind, f"_param{count}")
                count += 1

        return res

    def validation(self, idx):
        """
        The type of the param at idx, if the function checks that its value is
        a valid one - as solc does since 0.8, reverting if it's not.
        """
        cd = ("cd", idx)

        def f(exp):
            if (m := match(exp, ("eq", ":a", ":b"))) and cd in (m.a, m.b):
                other = m.b if m.a == cd else m.a
                if m := match(other, ("mask_shl", ":int:size", ":int:off", 0, cd)):
                    if m.off == 0 and m.size % 8 == 0 and 0 < m.size <= 256:
                        return [mask_to_type(m.size) or f"uint{m.size}"]
                    if m.off + m.size == 256 and m.size % 8 == 0:
                        return [f"bytes{m.size // 8}"]
                if match(other, ("bool", cd)):
                    return ["bool"]
                if m := match(other, ("signextend", ":int:b", cd)):
                    return [f"int{8 * (m.b + 1)}"]
            return []

        found = set(find_f_list(self.trace, f))
        if len(found) == 1:
            return found.pop()

        return None

    def serialize(self):
        trace = self.trace

        res = {
            "hash": self.hash,
            "name": self.name,
            "color_name": self.color_name,
            "abi_name": self.abi_name,
            "length": self.ast_length(),
            "getter": self.getter,
            "const": self.const,
            "payable": self.payable,
            "value_fails": self.value_fails,
            "print": self.print(),
            "trace": trace,
            "params": self.inferred_params,
        }
        try:
            assert json.dumps(res)  # check if serialisation works well
        except Exception:
            logger.error("failed serialization %s", self.name)
            raise

        return res

    def print(self):
        out = self._print()
        return "\n".join(out)

    def _print(self):
        set_func(self.hash)
        set_func_params_if_none(self.inferred_params)
        # (desc of desc.srcToken, amounts of amounts[2])
        set_names(
            params=[
                re.match(r"\w*", p[1]).group(0) for p in self.inferred_params.values()
            ]
        )

        if self.const is not None:
            val = self.const
            if opcode(val) == "return":
                val = val[1]

            return [
                COLOR_HEADER
                + "const "
                + ENDC
                + str(self.color_name.split("()")[0])
                + " = "
                + COLOR_BOLD
                # what the function returns: its data, as a return's
                + ", ".join(pretty_memory(val, abi_text=True))
                + ENDC
            ]

        else:
            comment = ""

            if not self.payable:
                # sent ether, it reverts with no data - or, the check of old
                # compilers, it runs an invalid opcode (see OUTPUT.md)
                comment = "# not payable"
                if self.value_fails == "invalid":
                    comment += " (invalid)"

            if self.name == "_fallback(?)":
                if self.payable:
                    comment = "# default function"
                else:
                    comment += ", default function"

            header = [
                color("def ", C.header)
                + self.color_name
                + (color(" payable", C.header) if self.payable else "")
                + ": "
                + color(comment, C.gray)
            ]

            if self.ast is not None:
                res = list(pprint_logic(self.ast))
            else:
                res = list(pprint_logic(self.trace))

            if len(res) == 0:
                res = ["  stop"]

            return header + res

    def analyse(self):
        assert len(self.trace) > 0

        def find_returns(exp):
            if opcode(exp) == "return":
                return [exp]
            else:
                return []

        exp_text = []

        self.returns = find_f_list(self.trace, find_returns)

        exp_text.append(("possible return values", prettify(self.returns)))

        # the check that there's no value, after what has no effect (Vyper
        # writes some constants to memory first)
        k = 0
        while k < len(self.trace) - 1 and opcode(self.trace[k]) in (
            "setmem",
            "setvar",
        ):
            k += 1
        first = self.trace[k]

        # a function is printed as payable unless the check is there: its
        # body is what it does whatever ether it's sent (one that always
        # reverts does it with the ether too - with the data it reverts with)
        self.payable = True
        if opcode(first) == "if":
            cond, fails, rest = simplify_bool(first[1]), first[2], first[3]
            if cond == ("iszero", "callvalue"):
                cond, fails, rest = "callvalue", rest, fails
            # (Vyper 0.3.10 checks the size of the calldata with it, `if
            # call.value or calldata.size < 36: revert`: that check stays)
            others = [d for d in disjuncts(cond) if d != "callvalue"]
            if (
                len(others) < len(disjuncts(cond))
                and fails
                and (fails[0] == ("revert", None) or opcode(fails[0]) == "invalid")
            ):
                if others:
                    other = others[0]
                    for d in others[1:]:
                        other = ("or", other, d)
                    rest = [("if", other, fails, rest)]
                self.trace = self.trace[:k] + rest
                self.payable = False
                self.value_fails = opcode(fails[0])

        exp_text.append(("payable", self.payable))

        self.read_only = True
        # (what changes the state, or does something that stays: a log - or
        # may: a recursive internal function, see vm.VM.internal)
        for op in [
            "store",
            "tstore",
            "selfdestruct",
            "call",
            "delegatecall",
            "callcode",
            "create",
            "create2",
            "log",
            "internal",
        ]:
            if f"'{op}'" in str(self.trace):
                self.read_only = False

        exp_text.append(("read_only", self.read_only))

        """
            const func detection
        """

        self.const = (
            self.read_only
            and len(self.returns) == 1
            and not varies(self.trace)
            # (sent ether, a const reverts with no data: see OUTPUT.md)
            and self.value_fails == "revert"
        )

        if self.const:
            self.const = self.returns[0]
            if len(self.const) == 3 and opcode(self.const[2]) == "data":
                self.const = self.const[2]
            if len(self.const) == 3 and opcode(self.const[2]) == "mask_shl":
                self.const = self.const[2]
            if len(self.const) == 3 and type(self.const[2]) == int:
                self.const = self.const[2]
        else:
            self.const = None

        if self.const:
            exp_text.append(("const", self.const))

        """
            getter detection
        """

        self.getter = None
        if self.const is None and self.read_only and len(self.returns) == 1:
            ret = word(self.returns[0][1])
            if match(ret, ("bool", ("storage", Any, Any, ":loc"))):
                self.getter = (
                    ret  # we have to be careful when using this for naming purposes,
                )
                # because sometimes the storage can refer to array length

            elif opcode(ret) == "mask_shl" and opcode(ret[4]) == "storage":
                self.getter = ret[4]
            elif opcode(ret) == "storage":
                self.getter = ret
            elif opcode(ret) == "data":
                terms = [word(t) for t in ret[1:]]
                # for structs, we check if all the parts of the struct are storage from the same
                # location. if so, we return the location number

                t0 = terms[0]  # 0xFAFfea71A6da719D6CAfCF7F52eA04Eb643F6De2 - documents
                if m := match(t0, ("storage", 256, 0, ":loc")):
                    loc = m.loc
                    for e in terms[1:]:
                        if not match(e, ("storage", 256, 0, ("add", Any, loc))):
                            break
                    else:
                        self.getter = t0

                # kitties getKitten - with more cases this and the above could be uniformed
                if self.getter is None:
                    prev_loc = -1
                    for e in terms:

                        def l2(x):
                            if m := match(x, ("sha3", ("data", Any, ":l"))):
                                if type(m.l) == int and m.l < 1000:
                                    return m.l
                            if (
                                opcode(x) == "sha3"
                                and type(x[1]) == int
                                and x[1] < 1000
                            ):
                                return x[1]
                            return None

                        loc = find_f(e, l2)
                        if not loc or (prev_loc != -1 and prev_loc != loc):
                            break
                        prev_loc = loc

                    else:
                        self.getter = ("struct", ("loc", loc))

            else:
                pass

        if self.getter:
            exp_text.append((f"getter for", prettify(self.getter)))

        explain_text("function traits", exp_text)

        return self


class InternalFunction(EasyCopy):
    """
    A recursive internal function, decompiled apart (see vm.VM.internal):
    its params the words its calls push above the address it returns to,
    its results those it leaves there.
    """

    def __init__(self, entry, params, results, frame, trace):
        self.entry = entry
        self.hash = self.name = internal_name(entry)
        self.params = params
        self.results = results
        # the words of the stack a call of itself takes: of the EVM's 1024,
        # what bounds the recursion
        self.frame = frame
        self.trace = deepcopy(trace)
        self.orig_trace = deepcopy(self.trace)
        self.ast = None
        # (not a getter nor a constant: see storage)
        self.getter = None
        self.const = None

    def param_names(self):
        return [f"_param{i + 1}" for i in range(self.params)]

    def print(self):
        # (not the params of the function of the abi printed before: its
        # calldata as it is)
        clear_func()
        set_names(params=self.param_names())
        header = (
            color("def ", C.header)
            + self.name
            + "("
            + ", ".join(self.param_names())
            + "): "
            + color(
                f"# recursive, a call takes {self.frame} of the 1024 words of the stack",
                C.gray,
            )
        )
        res = list(pprint_logic(self.ast if self.ast is not None else self.trace))
        return "\n".join([header] + res)

    def serialize(self):
        return {
            "name": self.name,
            "entry": self.entry,
            "params": self.params,
            "results": self.results,
            "frame": self.frame,
            "print": self.print(),
            "trace": self.trace,
        }
