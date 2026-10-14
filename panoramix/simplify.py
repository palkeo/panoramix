import collections
import time
import logging
import sys
from copy import copy

import panoramix.core.algebra as algebra
import panoramix.core.arithmetic as arithmetic
from panoramix.core.algebra import (
    _max_op,
    add_op,
    all_concrete,
    apply_mask,
    apply_mask_to_storage,
    bits,
    calc_max,
    div_op,
    divisible_bytes,
    flatten_adds,
    ge_zero,
    get_sign,
    is_word,
    le_op,
    lt_op,
    mask_op,
    max_op,
    max_to_add,
    may_be_wide,
    min_op,
    minus_op,
    mul_op,
    neg_mask_op,
    or_op,
    safe_le_op,
    safe_max_op,
    safe_min_op,
    shl_op,
    shr_op,
    signextend_op,
    simplify,
    simplify_max,
    sub_op,
    to_bytes,
    try_add,
)
from panoramix.core.arithmetic import changed_reads, is_zero, to_real_int
from panoramix.core.masks import get_bit, to_mask, to_neg_mask
from panoramix.core.memloc import (
    apply_mask_to_range,
    byte_elements,
    fill_mem,
    implicit,
    keep_width,
    keep_widths,
    max_value,
    max_value_bits,
    memloc_overwrite,
    range_overlaps,
    resize_bytes,
    sized,
    sizeof,
    split_setmem,
    split_store,
    splits_mem,
    value_bits,
    width_of,
    words,
)
from panoramix.core.memloc import safe_ge_zero as mem_ge_zero
from panoramix.core.memloc import safe_le_op as mem_le_op
from panoramix.matcher import Any, match
from panoramix.prettify import (
    explain,
    pformat_trace,
    pprint_repr,
    pprint_trace,
    pretty_repr,
)
from panoramix.utils.helpers import (
    C,
    MAX_EXP_SIZE,
    cached,
    contains,
    exp_size,
    find_f_list,
    find_f_set,
    find_op_list,
    is_array,
    opcode,
    replace,
    replace_f,
    replace_vars,
    replace_f_stop,
    rewrite_trace,
    rewrite_trace_full,
    rewrite_trace_ifs,
    to_exp2,
    walk_trace,
)

from panoramix.postprocess import cleanup_mul_1
from panoramix.rewriter import postprocess_exp, postprocess_trace

logger = logging.getLogger(__name__)


"""

    Simplifier engine.

    It takes the trace, and performs rewrites in the following loop:
        - simplify expressions
        - inline and cleanup variables (when possible)
        - inline and cleanup memory (when possible)
        - split setmems
        - split storage writes into separate lines
        - replace msize with actual values if possible
        - if there are any obvious always-fulfilled conditions, remove them
        - look for loops that are really setmems and clean those too
        - for loops that access storage, rewrite them so they are easier to parse by storage parser

    Finally, after all the processing is done:
    - replace `max(2+x, 3+y)` with `2 + max(x, 1+y)` - more human-readable, but would cause problems in the loop
    - do some other stuff that is human readable but would make the processing in the loop difficult


    If you want to see the engine in all it's glory, try:
        panoramix.py kitties tokenMetadata

    If you want to figure out exactly what happens, pprint_trace(trace) and pprint_repr(trace)
    are your friend. Put them in various places in simplify_trace and and see how the code changes.

"""

"""

    Pro tips:

        - rules in simplifier are quite independent. you can disable and move around a lot of the stuff
          and still maintain the mathematical correctedness of the output

        - the worst that will happen is that some edge cases in some contracts won't get simplified, but they
          will still be correct

        - bulk_compare.py is your friend for testing your changes. in Eveem I use something like it for initial
          tests and then do a huge integration comparison before each release, checking thousands of contracts for
          changes

        - if you like this approach, be sure to check out the work of people from Kestrel Institute, and their use
          of ACL2. They built a decompiler with a similar structure, but with simplification rules formally proven(!).

"""


def simplify_trace(trace, timeout=0):
    """
    The trace, simplified. While it is, the comparisons know that a variable
    is one of the values it's set to (algebra.set_variables): a snapshot of
    the free memory pointer is at least 0x60. And which of the words the
    contract is given are small (small_words).
    """
    values = variable_values(trace)
    algebra.set_variables(values, small_words(trace, values))
    try:
        return _simplify_trace(trace, timeout)
    finally:
        algebra.set_variables({})


def small_words(trace, values=None):
    """
    The words of the trace that are below 2**64 (see algebra.set_variables)
    - the others the contract is given, its params, what a call returned,
    may be anything: a param added to a pointer, unchecked, in assembly, may
    make an address wrap around 2**256. A word of the calldata, or computed
    from it (immutable), is small wherever it is when every execution that
    doesn't revert goes through what shows it (spine) - solc checks params
    before it uses them:
    - the size of a range of memory (`mem[a len x]`), the address of one of
      a known size (`mem[x]`): no execution pays for that much memory - not
      a sum, that may be small only modulo 2**256;
    - a check that it's below a number (solc's checks of the offsets and
      lengths of params: `if x > 2**64 - 1: revert`), or below what's small
      (an index, `require i < x`).
    """
    values = values or {}
    res = set()

    def small(x):
        # (a word: not a sum, nor a product, see algebra.value_range)
        if type(x) is not int and opcode(x) not in ("add", "mul"):
            if immutable(x, values):
                res.add(x)

    lines, checks = [], []
    for kind, x in spine(trace):
        (lines if kind == "line" else checks).append(x)

    for _, addr, size in find_f_list(lines, memory_ranges):
        small(size)
        if type(size) is int and size > 0:
            small(addr)

    for cond in checks:
        for x, (_, hi) in cond_bounds(cond).items():
            if hi <= algebra.MEMORY_TOP:
                small(x)

    def is_small(x):
        # (what isn't made of what the contract is given is, see
        # algebra._term_range)
        if type(x) is int:
            return 0 <= x <= algebra.MEMORY_TOP
        return not algebra.unchecked(x, values, res)

    changed = True
    while changed:
        changed = False
        for cond in checks:
            m = match(cond, (":op", ":x", ":y"))
            if not m or m.op not in ("lt", "le", "gt", "ge"):
                continue
            below, above = (m.x, m.y) if m.op in ("lt", "le") else (m.y, m.x)
            if below not in res and is_small(above):
                small(below)
                changed = changed or below in res

    # (as they'll be once the variables set once are replaced by their values,
    # see cleanup_vars)
    once = {("var", name): vals[0] for name, vals in values.items() if len(vals) == 1}
    if once:

        def inline(e):
            for _ in range(10):
                new = replace_f(e, lambda x: once.get(x, x))
                if new == e:
                    break
                e = new
            return e

        for x in list(res):
            small(inline(x))
    return res


def memory_ranges(e):
    """The range of memory e reads or writes, if it does: [range]."""
    if (
        opcode(e) in ("mem", "setmem")
        and len(e) > 1
        and opcode(e[1]) == "range"
        and len(e[1]) == 3
    ):
        return [e[1]]
    return []


def immutable(exp, values, seen=()):
    """
    Whether exp is the same word wherever it is in the trace: made of
    numbers and of the calldata, computed from them (algebra.COMPUTED), or a
    variable set to one of these once (values, see variable_values). Not
    what's read where it says - storage, memory, what a call returned, the
    same every time only until it changes.
    """
    if type(exp) is int or exp == "calldatasize":
        return True
    op = opcode(exp)
    if op in ("cd", "call.data") or op in algebra.COMPUTED:
        return all(immutable(e, values, seen) for e in exp[1:])
    if op == "var" and len(exp) == 2 and exp[1] not in seen:
        vals = values.get(exp[1], [])
        return len(vals) == 1 and immutable(vals[0], values, seen + (exp[1],))
    return False


# what a path that reverts ends with, and what else makes it go elsewhere
REVERTS = ("revert", "invalid", "assert_fail")
EXITS = ("return", "stop", "selfdestruct", "goto", "continue", "undefined", "leave")


def reverts(trace):
    """Whether the trace reverts (or runs an invalid opcode), on every path."""
    if not trace:
        return False
    last = trace[-1]
    if opcode(last) == "if" and len(last) == 4:
        if not (reverts(last[2]) and reverts(last[3])):
            return False
    elif opcode(last) not in REVERTS:
        return False
    return not goes_elsewhere(trace[:-1])


def goes_elsewhere(trace, jd=None):
    """
    Whether a path of the trace may end other than by reverting, or by going
    on with what follows it: a return, a jump back - but a continue of the
    loop of jd, the trace being its body.
    """
    return bool(
        find_f_list(
            trace,
            lambda e: (
                [e]
                if opcode(e) in EXITS
                and not (opcode(e) == "continue" and len(e) > 1 and e[1] == jd)
                else []
            ),
        )
    )


def spine(trace):
    """
    What every execution of the trace that doesn't revert goes through, in
    order: ("line", line) for a line it runs (or the condition of an if, of
    a loop), ("cond", c) for a condition that holds from there on - where an
    if's other branch reverts: a check. Until an execution may go elsewhere:
    a return in a branch, a loop back...
    """
    for idx, line in enumerate(trace):
        op = opcode(line)
        if op == "if" and len(line) == 4:
            _, cond, if_true, if_false = line
            yield "line", cond
            for branch, other, holds in (
                (if_true, if_false, is_zero(cond)),
                (if_false, if_true, cond),
            ):
                if reverts(branch):
                    yield "cond", holds
                    yield from spine(other + trace[idx + 1 :])
                    return
            if goes_elsewhere(if_true) or goes_elsewhere(if_false):
                return
        elif op == "while" and len(line) == 5:
            # (run 0 times or more: what follows is, if it's left at its end)
            _, cond, body, jd, _ = line
            yield "line", cond
            if goes_elsewhere(body, jd):
                return
        elif op == "label":
            return
        else:
            yield "line", line
            if op in EXITS + REVERTS:
                return


def variable_values(trace):
    """{name: [values]}: what each variable is set to in the trace, anywhere."""
    res = {}
    for sv in find_f_list(
        trace, lambda e: [e] if opcode(e) == "setvar" and len(e) == 3 else []
    ):
        res.setdefault(sv[1], []).append(sv[2])
    return res


def _simplify_trace(trace, timeout=0):
    time_start = time.monotonic()

    def should_quit():
        q = timeout and (time.monotonic() - time_start > timeout)
        if q:
            logger.warning("simplify_trace timed out.")
        return q

    old_trace = None
    count = 0
    while trace != old_trace and count < 40 and not should_quit():
        count += 1

        old_trace = trace

        # fun fact:
        # you can do prettify.pprint_trace(trace) or prettify.pprint_repr(trace)
        # between every stage here to see changes that happen to the code.

        trace = replace_f(trace, simplify_exp)
        explain("simplify expressions", trace)

        trace = cleanup_vars(trace)
        explain("cleanup variables", trace)

        trace = cleanup_mems(trace)
        explain("cleanup mems", trace)

        trace = precompiled_results(trace)
        explain("results of precompiled contracts", trace)

        trace = min_ifs(trace)
        explain("minimums", trace)

        trace = rewrite_trace(trace, split_setmem)
        trace = rewrite_trace_full(trace, split_store)
        explain("split setmems & storages", trace)

        trace = cleanup_vars(trace)
        explain("cleanup vars", trace)

        trace = replace_f(trace, simplify_exp)
        explain("simplify expressions", trace)

        trace = cleanup_mul_1(trace)
        explain("simplify expressions", trace)

        trace = replace_bytes_or_string_length(trace)
        explain("replace storage with length", trace)

        trace = cleanup_conds(trace)
        explain("cleanup unused ifs", trace)

        trace = rewrite_trace(trace, loop_to_setmem)
        explain("convert loops to setmems", trace)

        trace = propagate_storage_in_loops(trace)
        explain("move loop indexes outside of loops", trace)

        # there is a logic to this ordering, but it would take a long
        # time to explain. if you play with it, just run through bulk_compare.py
        # and see how it affects the code.

    # final lightweight postprocessing
    # introduces new variables, simplifies code for human readability
    # and does other stuff that would break the above loop

    trace = replace_f(trace, max_to_add)

    trace = replace_f(trace, postprocess_exp)
    trace = replace_f(trace, postprocess_exp)
    trace = rewrite_trace_ifs(trace, postprocess_trace)

    # (a string written to storage is left as the loops that copy it: no
    # notation says a store of all of its bytes, nor does storage.py read it)
    explain("using heuristics to clean up some things", trace)

    if not should_quit():
        # the most expensive pass, skipped if we're already out of time
        for _ in range(3):
            trace = cleanup_mems(trace)

    trace = precompiled_results(trace)
    trace = cleanup_conds(trace)
    explain("final setmem/condition cleanup", trace)

    def fix_storages(exp):
        # a field moved left (see algebra.apply_mask_to_storage): the field,
        # shifted
        if (m := match(exp, ("storage", ":size", ":int:off", ":loc"))) and m.off < 0:
            return ("mask_shl", m.size, 0, -m.off, ("storage", m.size, 0, m.loc))
        return exp

    trace = replace_f(trace, fix_storages)
    trace = cleanup_conds(trace)
    explain("cleaning up storages slightly", trace)

    trace = readability(trace)
    explain("adding nicer variable names", trace)

    trace = cleanup_mul_1(trace)

    # the conditions readability turned around may be known now: `if not
    # (1 | x)` is `if 0`
    trace = cleanup_conds(trace)

    return trace


def unwrap_bytes(exp):
    """
    exp, with the operands that are numbers not wrapped in "bytes", and the
    ones that are bytes not Bytes(n, v) of v bytes of another width than n
    (see memloc.resize_bytes).
    """
    op = opcode(exp)
    if op == "bytes":
        return exp

    if not any(opcode(e) == "bytes" for e in exp[1:]):
        return exp

    keep = byte_elements(exp)

    def unwrap(idx, e):
        if opcode(e) != "bytes":
            return e
        if idx in keep:
            return as_bytes(e)
        if op == "setmem" and idx == 2 and sized(e[2]):
            # bytes of another width than the memory written: the number
            # they make (see memloc.keep_setmem_width)
            return e
        return e[2]

    return (op,) + tuple(unwrap(idx, e) for idx, e in enumerate(exp[1:], 1))


def as_bytes(e):
    """
    e, a ("bytes", n, v) where it's bytes: v, when it's bytes of another
    width than n, as zeroes and v or its last n bytes.
    """
    if sized(e[2]) and (res := resize_bytes(e[2], e[1])) is not None:
        return res
    return e


def simplify_bytes(exp):
    _, size, val = exp
    val = simplify_exp(val)

    if (m := match(val, ("bytes", ":inner_size", ":inner"))) and mem_le_op(
        m.inner_size, size
    ) is True:
        val = m.inner

    if val != 0 and implicit(val, bits(size)):
        return val

    return ("bytes", size, val)


def data_elements(res):
    """
    The elements of a data without "bytes" when they're as wide as their
    value, zeroes followed by a value merged into it, and zeroes into one -
    and without the ones of no bytes (a range of length 0). Two parts of the
    same bytes, one where the other ends, are one.
    """
    res = [
        e[2] if opcode(e) == "bytes" and e[2] != 0 and implicit(e[2], bits(e[1])) else e
        for e in res
        if width_of(e) != 0
    ]

    def zeroes(e):
        # the width of e if it's zeroes, None otherwise
        if e == 0:
            return 256
        if (m := match(e, ("bytes", ":size", 0))) and type(m.size) is int:
            return bits(m.size)
        return None

    merged = []
    for e in res:
        if merged and (joined := join_slices(merged[-1], e)) is not None:
            merged[-1] = joined
            continue
        if merged and (z := zeroes(merged[-1])) is not None:
            w = sizeof(e)
            if zeroes(e) is not None:
                merged[-1] = ("bytes", (z + zeroes(e)) // 8, 0)
                continue
            inner = e[2] if opcode(e) == "bytes" else e
            if (
                type(w) is int
                and w % 8 == 0
                and opcode(merged[-1]) == "bytes"
                # the zeroes, then the low bytes of a number: its low bytes
                # then only if there's nothing above them - and a number of
                # no more than a word (bytes - a range, a data - go after
                # the zeroes: Bytes(n, v) is of a number)
                and z + w <= 256
                and not sized(inner)
                and value_bits(inner) <= w
            ):
                merged[-1] = ("bytes", (z + w) // 8, inner)
                continue
        merged.append(e)

    return [
        e[2] if opcode(e) == "bytes" and implicit(e[2], bits(e[1])) else e
        for e in merged
    ]


def join_slices(a, b):
    """
    The bytes a then b are, where they're the same bytes and b starts where a
    ends: `ext_call.return_data[0 len n], ext_call.return_data[n len 32 - n]`
    is `ext_call.return_data[0 len 32]`. None otherwise.
    """
    if (ma := match(a, ("mem", ("range", ":s1", ":n1")))) and (
        mb := match(b, ("mem", ("range", ":s2", ":n2")))
    ):
        if sub_op(mb.s2, add_op(ma.s1, ma.n1)) == 0:
            return ("mem", ("range", ma.s1, add_op(ma.n1, mb.n2)))
        return None
    if (
        opcode(a) == opcode(b)
        and is_array(opcode(a))
        and len(a) == 3
        and len(b) == 3
        and sub_op(b[1], add_op(a[1], a[2])) == 0
    ):
        return (a[0], a[1], add_op(a[2], b[2]))
    return None


@cached
def simplify_exp(exp):
    if type(exp) == list:
        return exp

    if type(exp) is tuple:
        exp = unwrap_bytes(exp)

    if opcode(exp) == "bytes":
        return simplify_bytes(exp)

    if m := match(exp, ("shr", ":int:off", ":val")):
        # a shift by an amount the vm didn't know, known now (the word of
        # what it was made of)
        exp = shr_op(m.val, m.off % 2**256)

    if m := match(exp, ("shl", ":off", ":val")):
        # (see shl_op: a mask when it's known to be one)
        off = m.off % 2**256 if type(m.off) is int else m.off
        exp = shl_op(m.val, off)

    if m := match(exp, ("signextend", ":b", ":val")):
        exp = signextend_op(m.b, m.val)

    if match(exp, ("mask_shl", Any, Any, Any, 0)):
        # whatever its size and offset
        return 0

    if opcode(exp) == "and":
        _, *terms = exp
        real = 2**256 - 1
        symbols = []
        for t in terms:
            if type(t) in (int, bool) and t >= 0:
                # (False and x: a truth value that is 0)
                real = real & int(t)
            elif opcode(t) == "and":
                symbols += t[1:]
            else:
                symbols.append(t)

        if real == 0:
            return 0

        if real != 2**256 - 1:
            res = (real,)
        else:
            res = tuple()

        res += tuple(symbols)
        exp = ("and",) + res if len(res) > 1 else res[0] if res else 2**256 - 1

    if m := match(exp, ("iszero", ("iszero", ":e"))):
        exp = ("bool", m.e)

    if m := match(exp, ("bool", ("bool", ":e"))):
        exp = ("bool", m.e)

    if (m := match(exp, ("eq", ":sth", 0))) or (m := match(exp, ("eq", 0, ":sth"))):
        exp = ("iszero", m.sth)

    if (
        (m := match(exp, ("eq", ":sth", 1))) or (m := match(exp, ("eq", 1, ":sth")))
    ) and arithmetic.is_bool(m.sth):
        # a truth value is 1 when it's true
        exp = m.sth

    if (m := match(exp, (":op", ":a", ":int:c"))) and m.op in ("lt", "gt", "le", "ge"):
        # a comparison that what's compared can't make true, or false
        top = max_value(m.a)
        c = m.c % 2**256
        if (m.op == "gt" and top <= c) or (m.op == "ge" and top < c):
            return 0
        if (m.op == "lt" and top < c) or (m.op == "le" and top <= c):
            return 1

    if (m := match(exp, (":op", ":int:c", ":a"))) and m.op in ("lt", "gt", "le", "ge"):
        top = max_value(m.a)
        c = m.c % 2**256
        if (m.op == "lt" and top <= c) or (m.op == "le" and top < c):
            return 0
        if (m.op == "gt" and top < c) or (m.op == "ge" and top <= c):
            return 1

    if m := match(exp, (":op", ("mul", -1, ":x"))):
        if m.op in ("iszero", "bool"):
            # -x is 0 when x is
            exp = (m.op, m.x)

    if m := match(exp, ("iszero", ("xor", ":a", ":b"))):
        # a xor b is 0 when they're equal
        exp = ("eq", m.a, m.b)

    if m := match(exp, ("call.data", ":int:pos", 32)):
        # a word of the calldata
        return ("cd", m.pos)

    if m := match(exp, ("call.data", "calldatasize", ":int:size")):
        # past the end of the calldata: zeroes (Vyper copies them to clear
        # the memory)
        return ("bytes", m.size, 0)

    if (m := match(exp, ("iszero", ("add", ":int:c", ":x")))) and len(exp[1]) == 3:
        exp = equals(m.x, m.c)

    elif (m := match(exp, ("iszero", ":diff"))) and (eq := difference_eq(m.diff)):
        exp = eq

    if (
        (m := match(exp, ("mask_shl", ":int:size", 5, 0, ("add", ":int:num", ...))))
        and m.size + 5 >= 256
        and m.num % 32 == 31
        and m.num > 32
    ):
        # floor32(num + x) = (num - 31) + floor32(31 + x), num - 31 being
        # a multiple of 32 - with a mask up to the top of the word, a
        # narrower one would drop the high bits of num - 31
        add_terms = exp[-1][2:]  # the "..."
        exp = (
            "add",
            m.num - 31,
            (
                "mask_shl",
                m.size,
                5,
                0,
                (
                    "add",
                    31,
                )
                + add_terms,
            ),
        )

    if (
        (m := match(exp, ("iszero", ("mask_shl", ":size", ":off", ":shl", ":val"))))
        and all_concrete(m.size, m.off, m.shl)
        and -m.off <= m.shl <= 256 - m.size - m.off
    ):
        # a shift that loses none of the bits of the mask doesn't change
        # whether they're all 0
        exp = ("iszero", ("mask_shl", m.size, m.off, 0, m.val))

    if m := match(exp, ("max", ":single")):
        exp = m.single

    if m := match(exp, ("mem", ("range", Any, 0))):
        # no data: the params of a log, of a call... (in a data, no part of
        # it - and not a number)
        return None

    if (
        (m := match(exp, ("mod", ":exp2", ":int:num")))
        and (size := to_exp2(m.num))
        and size > 1
    ):
        return mask_op(m.exp2, size=size)

    if m := match(exp, ("mod", 0, Any)):
        exp = 0

    # same thing is added in both expressions ?
    if (
        (m := match(exp, (":op", ("add", ...), ("add", ...))))
        and m.op in ("lt", "le", "gt", "ge")
        and all(max_value_bits(t) <= 250 for t in exp[1][1:] + exp[2][1:])
        and len(exp[1]) <= 33
        and len(exp[2]) <= 33
    ):
        # sums that don't wrap around 2**256 compare as integers do: what
        # both have can go. Otherwise it can't: x + 1 < x is an overflow.
        e1 = collections.Counter(exp[1][1:])  # first ...
        e2 = collections.Counter(exp[2][1:])  # second ...
        common = e1 & e2
        t1 = tuple((e1 - common).elements())
        t2 = tuple((e2 - common).elements())
        exp = (m.op, add_op(*t1), add_op(*t2))

    if m := match(exp, ("add", ":e")):
        logger.debug("single add")
        return simplify_exp(m.e)

    if m := match(exp, ("mul", 1, ":e")):
        return simplify_exp(m.e)

    if m := match(exp, ("div", ":e", 1)):
        return simplify_exp(m.e)

    if (m := match(exp, ("mask_shl", 256, 0, 0, ":val"))) and not may_be_wide(m.val):
        return simplify_exp(m.val)

    if m := match(exp, ("mask_shl", ":int:size", ":int:offset", ":int:shl", ":e")):
        exp = mask_op(simplify_exp(m.e), m.size, m.offset, m.shl)

    if (
        m := match(
            exp, ("mask_shl", ":size", 0, 0, ("div", ":expr", ("exp", 256, ":shr")))
        )
    ) and is_word(("mul", 8, m.shr)):
        # (256 ** shr is 0 for a shr of 32 or more, and so is the division
        # by it: the mask's offset is 8 * shr if that's the integer it's made
        # of - 32 - x isn't, nor 8 * x for any x, see value_range)
        shift = bits(m.shr)
        exp = mask_op(simplify_exp(m.expr), m.size, shift, shr=shift)

    if (
        (
            m := match(
                exp, ("mask_shl", Any, Any, ":shl", ("storage", ":size", ":off", Any))
            )
        )
        and not (type(m.off) is int and m.off < 0)
        and safe_le_op(m.size, minus_op(m.shl))
    ):
        # all the bits of a field (at 0: not one moved left, see
        # algebra.apply_mask_to_storage) moved down past 0
        return 0

    if m := match(exp, ("or", ":sth", 0)):
        return m.sth

    if opcode(exp) == "add":
        terms = exp[1:]
        res = 0
        for el in terms:
            el = simplify_exp(el)
            if opcode(el) == "add":
                for e in el[1:]:
                    res = add_op(res, e)
            else:
                res = add_op(res, el)
        exp = res

    if opcode(exp) == "mask_shl":
        exp = cleanup_mask_data(exp)

    if m := match(
        exp, ("mask_shl", ":size", 0, 0, ("mem", ("range", ":mem_loc", ":mem_size")))
    ):
        if divisible_bytes(m.size) and mem_le_op(to_bytes(m.size)[0], m.mem_size):
            return (
                "mem",
                apply_mask_to_range(("range", m.mem_loc, m.mem_size), m.size, 0),
            )

    if (
        m := match(
            exp,
            (
                "mask_shl",
                ":size",
                ":off",
                ":shl",
                ("mem", ("range", ":mem_loc", ":mem_size")),
            ),
        )
    ) and m.shl == minus_op(m.off):
        if (
            divisible_bytes(m.size)
            and divisible_bytes(m.off)
            # (the bytes masked are in the range: of mem[p len 1], bits
            # 248 to 256 aren't)
            and mem_le_op(add_op(to_bytes(m.size)[0], to_bytes(m.off)[0]), m.mem_size)
            is True
        ):
            return (
                "mem",
                apply_mask_to_range(("range", m.mem_loc, m.mem_size), m.size, m.off),
            )

    elif opcode(exp) == "data":
        params = exp[1:]
        res = []

        # simplify inner expressions, and remove nested 'data's
        for e in params:
            # (removes further nested datas, and does other simplifications;
            # each as wide as it was)
            new = simplify_exp(e)
            if new is None and opcode(e) == "mem":
                # a range of length 0 (see below): no bytes
                continue
            e = keep_width(e, new)
            if opcode(e) == "bytes":
                e = as_bytes(e)
            if opcode(e) == "data":
                res.extend(e[1:])
            else:
                res.append(e)

        # sometimes we may have on expression split into two masks next to each other
        # e.g.  (mask_shl 192 64 -64 (cd 36)))) (mask_shl 64 0 0 (cd 36))
        #       (in solidstamp withdrawRequest)
        # merge those into one.

        res2 = res
        res = None
        while (
            res2 != res
        ):  # every swipe merges up to two elements next to each other. repeat until there are no new merges
            # there's a more efficient way of doing this for sure.
            res, res2 = res2, []
            idx = 0
            while idx < len(res):
                el = res[idx]
                if idx == len(res) - 1:
                    res2.append(el)
                    break

                next_el = res[idx + 1]
                idx += 1

                if (
                    (m := match(el, ("mask_shl", ":size", ":offset", ":shl", ":val")))
                    and next_el == ("mask_shl", m.offset, 0, 0, m.val)
                    and add_op(m.offset, m.shl) == 0
                ):
                    res2.append(("mask_shl", add_op(m.size, m.offset), 0, 0, m.val))
                    idx += 1

                else:
                    res2.append(el)

        res = res2

        # could do the same for mem slices, but no case of that happening yet

        res = data_elements(res)

        if len(res) == 1:
            return res[0]
        else:
            return ("data",) + tuple(res)

    if type(exp) == int and to_real_int(exp) > -(
        8**22
    ):  # if it's larger than 30 bytes, it's probably
        # an address, not a negative number
        return to_real_int(exp)

    if m := match(exp, ("and", ":num", ":num2")):
        num = arithmetic.eval(m.num)
        num2 = arithmetic.eval(m.num2)
        if type(num) == int or type(num2) == int:
            return simplify_mask(exp)

    if type(exp) == list:
        r = simplify_exp(tuple(exp))
        return list(r)

    if type(exp) != tuple:
        return exp

    if m := match(
        exp, ("mask_shl", ":int:size", ":int:offset", ":int:shl", ":int:val")
    ):
        return apply_mask(m.val, m.size, m.offset, m.shl)

    if (
        (
            m := match(
                exp, ("mask_shl", ":size", ":int:off", ":shl", ("add", ":int:num", ...))
            )
        )
        and 0 < m.num < 2**m.off
        and all(low_zero_bits(t) >= m.off for t in exp[4][2:])
    ):
        # the other terms are multiples of 2**off: adding less than that
        # doesn't change their bits from off up (ceil32 of a multiple of 32
        # is the number)
        return simplify_exp(("mask_shl", m.size, m.off, m.shl, add_op(*exp[4][2:])))

    if (
        (m := match(exp, ("mask_shl", ":int:size", ":int:off", 0, ":val")))
        and m.size + m.off >= 256
        and (opcode(m.val) in ("add", "mul"))
        and low_zero_bits(m.val) >= m.off
    ):
        # the mask keeps all the bits of the word that may be set
        return simplify_exp(m.val)

    if opcode(exp) == "mul":
        terms = exp[1:]
        res = 1
        for e in terms:
            res = mul_op(res, simplify_exp(e))

        if m := match(res, ("mul", 1, ":res")):
            res = m.res

        return res

    if opcode(exp) == "max":
        terms = exp[1:]
        els = [simplify_exp(e) for e in terms]
        res = 0
        for e in els:
            res = _max_op(res, e)
        return res

    res = tuple()
    for e in exp:
        res += (simplify_exp(e),)

    if res != exp:
        # what's bytes in it as wide as it was (see memloc.keep_widths)
        res = keep_widths(exp, res)

    return res


def simplify_mask(exp):
    op = opcode(exp)

    if op in arithmetic.OPCODES:
        exp = arithmetic.eval(exp)

    if m := match(exp, ("and", ":left", ":right")):
        left, right = m.left, m.right

        if mask := to_mask(left):
            exp = mask_op(right, *mask)

        elif mask := to_mask(right):
            exp = mask_op(left, *mask)

        elif bounds := to_neg_mask(left):
            exp = neg_mask_op(right, *bounds)

        elif bounds := to_neg_mask(right):
            exp = neg_mask_op(left, *bounds)

    elif (m := match(exp, ("div", ":left", ":int:right"))) and (
        shift := to_exp2(m.right)
    ):
        exp = mask_op(m.left, size=256 - shift, offset=shift, shr=shift)

    elif (m := match(exp, ("mul", ":int:left", ":right"))) and (
        shift := to_exp2(m.left)
    ):
        exp = mask_op(m.right, size=256 - shift, shl=shift)

    elif (m := match(exp, ("mul", ":left", ":int:right"))) and (
        shift := to_exp2(m.right)
    ):
        exp = mask_op(m.left, size=256 - shift, shl=shift)

    return exp


def cleanup_mask_data(exp):
    # if there is a mask over some data,
    # it removes pieces of data that for sure won't
    # fit into the mask (doesn't truncate if something)
    # partially fits
    # e.g. mask(200, 8, 8 data(123, mask(201, 0, 0, sth), mask(8,0, 0, sth_else)))
    #      -> mask(200, 0, 0, mask(201, 0, 0, sth))

    def _cleanup_right(exp):
        # removes elements that are cut off by offset
        m = match(exp, ("mask_shl", ":size", ":offset", ":shl", ":val"))
        assert m
        size, offset, shl, val = m.size, m.offset, m.shl, m.val

        if opcode(val) != "data":
            return exp

        last = val[-1]
        if sizeof(last) is not None and mem_le_op(sizeof(last), offset):
            offset = sub_op(offset, sizeof(last))
            shl = add_op(shl, sizeof(last))
            if len(val) == 3:
                val = val[1]
            else:
                val = val[:-1]

            return mask_op(val, size, offset, shl)

        return exp

    def _cleanup_left(exp):
        # removes elements that are cut off by size+offset
        m = match(exp, ("mask_shl", ":size", ":offset", ":shl", ":val"))
        assert m
        size, offset, shl, val = m.size, m.offset, m.shl, m.val

        if opcode(val) != "data":
            return exp

        total_size = add_op(size, offset)  # simplify_exp

        sum_sizes = 0
        val = list(val[1:])
        res = []
        while len(val) > 0:
            last = val.pop()
            if sizeof(last) is None:
                return exp
            sum_sizes = simplify_exp(add_op(sum_sizes, sizeof(last)))
            res.insert(0, last)
            if mem_le_op(total_size, sum_sizes):
                return exp[:4] + (("data",) + tuple(res),)

        return exp

    assert opcode(exp) == "mask_shl"

    prev_exp = None
    while prev_exp != exp:
        prev_exp = exp
        exp = _cleanup_right(exp)

    prev_exp = None
    while prev_exp != exp:
        prev_exp = exp
        exp = _cleanup_left(exp)

    if m := match(exp, ("mask_shl", ":size", 0, 0, ("data", ...))):
        size = m.size
        terms = exp[-1][1:]  # the "..."
        # if size of data is size of mask, we can remove the mask altogether
        sum_sizes = 0
        for e in terms:
            s = sizeof(e)
            if s is None:
                return exp
            sum_sizes = add_op(sum_sizes, s)

        if sub_op(sum_sizes, size) == 0:
            return ("data",) + terms

    return exp


def var_ids(exp, res=None):
    """The ids of the variables exp (a trace, a line...) reads or sets."""
    if res is None:
        res = set()
    if type(exp) in (list, tuple):
        if len(exp) >= 2 and exp[0] in ("var", "setvar") and type(exp[1]) is int:
            res.add(exp[1])
        for e in exp:
            var_ids(e, res)
    return res


def has_loop(trace):
    """Whether there's a loop in the trace."""
    # (the branches still to look at, rather than a call for each: ifs
    # nested deeper than python's recursion limit)
    todo = [trace]
    while todo:
        for line in todo.pop():
            if opcode(line) == "while":
                return True
            if opcode(line) == "if":
                todo.extend(line[2:])
    return False


def replace_while_var(rest, counter_idx, new_idx, taken=None):
    """
    rest, with the variable counter_idx renamed to the first from new_idx
    no other variable has - taken: those read or set in rest and around it
    (the loops it's in, what comes after the ifs it's in). One that's set
    there and never read would be printed with the same name, and read as it.
    """
    if taken is None:
        taken = var_ids(rest)
    while new_idx in taken and new_idx != counter_idx:
        new_idx += 1
    taken.add(new_idx)

    def r(exp):
        if exp == ("var", counter_idx):
            return ("var", new_idx)

        elif m := match(exp, ("setvar", counter_idx, ":val")):
            return ("setvar", new_idx, m.val)

        else:
            return exp

    return simplify_exp(replace_f(rest, r)), new_idx


def canonise_max(exp):
    if opcode(exp) == "max":
        args = []
        for e in exp[1:]:
            if m := match(e, ("mul", 1, ":num")):
                args.append(m.num)
            else:
                args.append(e)

        args.sort(key=lambda x: str(x) if type(x) != int else " " + str(x))
        return ("max",) + tuple(args)
    else:
        return exp


assert canonise_max(("max", ("mul", 1, ("x", "y")), 4)) == ("max", 4, ("x", "y"))


def readability(trace, outer=frozenset()):
    """
    - replaces variable names with nicer ones,
    - fixes empty memory in calls
    - replaces 'max..' in setmems with msize variable
        (max can only appear because of this)

    outer: the variables used around the trace (see replace_while_var).
    """

    trace = replace_f(trace, canonise_max)

    res = []
    for idx, line in enumerate(trace):
        if m := match(line, ("setmem", ("range", ("add", ...), Any), ":mem_val")):
            add_params = line[1][1][1:]  # the "..."
            for m in add_params:
                if opcode(m) == "max":
                    res.append(("setvar", "_msize", m))

                    def x(line):
                        return [replace(line, m, ("var", "_msize"))]

                    rest = rewrite_trace(trace[idx:], x)
                    res.extend(readability(rest, outer))
                    return res

        elif m := match(line, ("if", ":cond", ":if_true", ":if_false")):
            cond, if_true, if_false = m.cond, m.if_true, m.if_false

            # (a variable of a loop in a branch mustn't be one that's used
            # after the if)
            around = outer
            if has_loop(if_true) or has_loop(if_false):
                around = outer | var_ids(trace[idx + 1 :])
            if_true = readability(if_true, around)
            if_false = readability(if_false, around)

            # if if_false ~ [('revert', ...)]: # no lists in Tilde... yet :,)
            if len(if_false) == 1 and opcode(if_false[0]) == "revert":
                res.append(("if", is_zero(cond), if_false, if_true))
            else:
                res.append(("if", cond, if_true, if_false))

            continue

        elif opcode(line) == "while":
            # for whiles, normalize variable names

            a = parse_counters(line)

            rest = trace[idx:]
            taken = var_ids(rest) | outer

            if "counter" in a:
                counter_idx = a["counter"]
                rest, _ = replace_while_var(rest, counter_idx, 0, taken)

            else:
                counter_idx = -1

            new_idx = 1

            cond, path, jds, vars = line[1:]

            for _, v_idx, _ in vars:
                if v_idx != counter_idx:
                    rest, new_idx = replace_while_var(rest, v_idx, new_idx, taken)

            line, rest = rest[0], rest[1:]
            cond, path, jds, vars = line[1:]

            # (the loops in it: not the variables of this one, nor what's
            # used after it)
            path = readability(path, outer | var_ids(line) | var_ids(rest))
            res.append(("while", cond, path, jds, vars))

            res.extend(readability(rest, outer))
            return res

        res.append(line)

    return res


def replace_bytes_or_string_length(trace):
    """
    The length of the bytes (or string) at a slot, as the compiler reads it
    from the word there - twice it in the low byte of a short one, twice it
    plus 1 in a long one:

        (x & (256 * iszero(x & 1) - 1)) >> 1    with x = stor[key]

    is ("length", key) - bits of it when the shift keeps fewer.
    """

    def is_mask(e, key):
        # 256 * iszero(stor[key] & 1) - 1: 255 when the low bit is 0, else -1
        m = match(e, ("add", -1, ("mask_shl", ":int:size", 0, 8, ("iszero", ":low"))))
        return bool(m) and m.size >= 1 and m.low == ("storage", 1, 0, key)

    def replace(expr):
        m = match(
            expr, ("mask_shl", ":int:size", ":int:offset", -1, ("and", ":a", ":b"))
        )
        if not m or m.offset < 1 or m.size + m.offset > 256:
            return
        size, offset = m.size, m.offset
        for word, mask in ((m.a, m.b), (m.b, m.a)):
            if (w := match(word, ("storage", 256, 0, ":key"))) and is_mask(mask, w.key):
                key = w.key
                break
        else:
            return

        if type(key) == int:
            key = ("loc", key)

        if size == 255 and offset == 1:
            return ("storage", 256, 0, ("length", key))
        return ("mask_shl", size, offset - 1, 0, ("storage", 256, 0, ("length", key)))

    return replace_f_stop(trace, replace)


def loop_to_setmem(line):
    if opcode(line) == "while":
        r = _loop_to_setmem(line)

        if r is not None:
            return r

        r = loop_to_setmem_from_storage(line)

        if r is not None:
            return r

    return [line]


def vars_in_expr(expr):
    if m := match(expr, ("var", ":var_id")):
        return frozenset([m.var_id])

    s = frozenset()

    if type(expr) not in (tuple, list):
        return s

    for e in expr:
        s = s | vars_in_expr(e)
    return s


def only_add_in_expr(op):
    if m := match(op, ("setvar", ":idx", ":val")):
        return only_add_in_expr(m.val)
    if opcode(op) == "add":
        terms = op[1:]
        return all(only_add_in_expr(o) for o in terms)
    if match(op, ("var", Any)):
        return True
    if m := match(op, ("sha3", ":term")):  # sha3(constant) is allowed.
        return opcode(m.term) is None
    if opcode(op) is not None:
        return False
    return True


assert only_add_in_expr(("setvar", 100, ("mul", ("var", 100), 1))) is False
assert only_add_in_expr(("setvar", 100, ("add", ("var", 100), 1))) is True


def propagate_storage_in_loop(line, after=()):
    """
    A loop variable that starts at a slot sha3(...) + k, made to start at k
    and read as itself + sha3(...) in the loop. (Not one that `after`, what
    runs after the loop, reads: it'd read it moved - nor one the step of
    another variable reads.)
    """
    op, cond, path, jds, setvars = line
    assert op == "while"

    def storage_sha3(value):
        if opcode(value) == "add":
            terms = value[1:]
            for op in terms:
                if storage_sha3(op) is not None:
                    return storage_sha3(op)

        if m := match(value, ("sha3", ":val")):
            if type(m.val) != int or m.val < 1000:  # used to be int:val here, why?
                return value

    # the steps of the loop, at the continues of its own, as deep as they are
    steps = [instr for c in find_conts(path) if c[1] == jds for instr in c[2]]

    def path_only_add_in_continue(path):
        return all(only_add_in_expr(instr) for instr in steps)

    new_setvars = []

    for setvar in setvars:
        op, var_id, value = setvar
        assert op == "setvar"

        sha3 = storage_sha3(value)
        if not sha3:
            new_setvars.append(setvar)
            continue

        # If the "continue" instructions don't only add stuff to the index variable,
        # it's not safe to proceed. If we would do "i = i * 2", then it doesn't
        # make sense to substract a constant to "i".
        if not path_only_add_in_continue(path):
            new_setvars.append(setvar)
            continue

        if contains(after, ("var", var_id)) or any(
            instr[1] != var_id and contains(instr[2], ("var", var_id))
            for instr in steps
        ):
            new_setvars.append(setvar)
            continue

        new_setvars.append(("setvar", var_id, sub_op(value, sha3)))

        def add_sha3(t):
            # We replace occurrences of var by "var + sha3"
            if t == ("var", var_id):
                return add_op(t, sha3)
            # Important: for "continue" we don't want to touch the variable.
            # TODO: This is only valid if the "continue" contains only
            # operators like "+" or "-". We should check that.
            if opcode(t) == "continue":
                return t

        path = replace_f_stop(path, add_sha3)
        cond = replace_f_stop(cond, add_sha3)

    return [("while", cond, path, jds, new_setvars)]


def propagate_storage_in_loops(trace, after=()):
    """`after`: what runs after `trace` (a branch of an if), its parts."""
    res = []
    for idx, line in enumerate(trace):
        if opcode(line) == "if":
            _, cond, if_true, if_false = line
            rest = (trace[idx + 1 :],) + after
            if_true = propagate_storage_in_loops(if_true, rest)
            if_false = propagate_storage_in_loops(if_false, rest)
            res.append(("if", cond, if_true, if_false))
        elif opcode(line) == "while":
            res.extend(propagate_storage_in_loop(line, (trace[idx + 1 :],) + after))
        else:
            res.append(line)

    return res


def _loop_to_setmem(line):
    def memidx_to_memrange(mem_idx, setvars, stepvars, endvars):
        # (the values of the next iteration, of the last, of the first: each
        # all at once, see replace_vars)
        assert all(opcode(v) == "setvar" for v in list(stepvars) + list(setvars))
        mem_idx_next = replace_vars(mem_idx, {v[1]: v[2] for v in stepvars})

        diff = sub_op(mem_idx_next, mem_idx)

        if diff not in (32, -32):
            return None, None

        mem_idx_last = replace_vars(mem_idx, dict(endvars))
        mem_idx_first = replace_vars(mem_idx, {v[1]: v[2] for v in setvars})

        if diff == 32:
            mem_len = sub_op(mem_idx_last, mem_idx_first)

            return ("range", mem_idx_first, mem_len), diff

        else:
            assert diff == -32

            mem_idx_last = add_op(32, mem_idx_last)
            mem_idx_first = add_op(32, mem_idx_first)

            mem_len = sub_op(mem_idx_first, mem_idx_last)

            return ("range", mem_idx_last, mem_len), diff

    op, cond, path, jds, setvars = line
    assert op == "while"

    if len(path) != 2:
        return None

    if opcode(path[1]) != "continue":
        return None

    if opcode(path[0]) != "setmem":
        return None

    setmem = path[0]
    cont = path[1]

    mem_idx, mem_val = setmem[1], setmem[2]

    op, i, l = mem_idx
    assert op == "range"
    if l != 32:
        return None
    mem_idx = i

    stepvars = cont[2]

    a = parse_counters(line)
    if "endvars" not in a:
        return None

    setvars, endvars = a["setvars"], a["endvars"]

    rng, diff = memidx_to_memrange(mem_idx, setvars, stepvars, endvars)

    if rng is None:
        return None

    if mem_val == 0:
        res = [("setmem", rng, 0)]

    elif opcode(mem_val) == "mem":
        mem_val_idx = mem_val[1]
        op, i, l = mem_val_idx
        assert op == "range"
        mem_val_idx = i
        if l != 32:
            return None

        val_rng, val_diff = memidx_to_memrange(mem_val_idx, setvars, stepvars, endvars)

        if val_rng is None:
            return None

        if val_diff != diff:
            return None  # possible but unsupported

        # we should check for overwrites here, but skipping for now
        # if the part of memcopy loop overwrites source before it's copied,
        # we can end up with unexpected behaviour, could at least show some warning,
        # or set that mem to 'complicated' or sth

        res = [("setmem", rng, ("mem", val_rng))]

    else:
        return None

    for v_idx, v_val in endvars.items():
        res.append(("setvar", v_idx, v_val))

    return res


def loop_to_setmem_from_storage(line):
    assert opcode(line) == "while"
    cond, path, jds, setvars = line[1:]

    if len(path) != 2 or opcode(path[0]) != "setmem" or opcode(path[1]) != "continue":
        return None

    setmem = path[0]
    cont = path[1]

    logger.debug("loop_to setmem_from_storage: %s\n%s\n%s", setmem, cont, cond)

    # (setmem, mem_idx, mem_val)
    mem_idx, mem_val = setmem[1], setmem[2]

    # Extract the interesting variable from mem_idx
    vars_in_idx = vars_in_expr(mem_idx)
    if len(vars_in_idx) != 1:
        return
    if not only_add_in_expr(mem_idx):
        return
    memory_index_var = next(iter(vars_in_idx))

    # Same from mem_val
    vars_in_val = vars_in_expr(mem_val)
    if len(vars_in_val) != 1:
        return
    storage_key_var = next(iter(vars_in_val))
    if opcode(mem_val) != "storage":
        return
    if mem_val[1] != 256 or mem_val[2] != 0:
        return
    storage_key = mem_val[3]
    if not only_add_in_expr(storage_key):
        return

    logger.debug("now look at the continue")
    update_memory_index = (
        "setvar",
        memory_index_var,
        ("add", 32, ("var", memory_index_var)),
    )
    update_storage_key = (
        "setvar",
        storage_key_var,
        ("add", 1, ("var", storage_key_var)),
    )
    if set(cont[2]) != {update_memory_index, update_storage_key}:
        return

    logger.debug("setvars")
    memory_index_start = None
    storage_key_start = None
    memory_index_init = None
    for setvar in setvars:
        if setvar[1] == memory_index_var:
            memory_index_start = replace(mem_idx, ("var", memory_index_var), setvar[2])
            memory_index_init = setvar[2]
        elif setvar[1] == storage_key_var:
            storage_key_start = replace(
                storage_key, ("var", storage_key_var), setvar[2]
            )
        else:
            return

    logger.debug("while condition")
    if memory_index_var not in vars_in_expr(cond):
        return
    if opcode(cond) != "gt":
        return

    mem_count = ("add", cond[1], ("mul", -1, cond[2]))
    mem_count = replace(mem_count, ("var", memory_index_var), memory_index_init)

    mem_rng = ("range", memory_index_start, mem_count)
    storage_rng = ("range", storage_key_start, ("div", mem_count, 32))

    logger.debug("mem_rng: %s, storage_rng: %s", mem_rng, storage_rng)
    return [("setmem", mem_rng, ("storage", 256, 0, storage_rng))]


"""

    simplifier

"""


def apply_constraint(exp, constr):
    """
    exp, where the condition constr holds: a min of two values it decides
    is one of them - min(32, return_data.size), the size of what a call
    wrote (see vm.VM.output_write), is 32 once return_data.size >= 32.
    """
    if "min" not in str(exp):
        return exp

    bounds = cond_bounds(constr)
    if not bounds:
        return exp

    def f(e):
        if opcode(e) == "min" and len(e) == 3:
            le = decide_le(e[1], e[2], bounds)
            if le is True:
                return e[1]
            if le is False:
                return e[2]
        return e

    return replace_f(exp, f)


# what a precompiled contract returns when its call goes through, if it's
# always as many bytes: a word, or two
PRECOMPILED_OUTPUT = {
    "sha256hash.result": 32,
    "ripemd160hash.result": 32,
    "bn256Add.result": 64,
    "bn256ScalarMul.result": 64,
    "bn256Pairing.result": 32,
}


def cond_bounds(cond):
    """
    {x: (lowest, highest)}: the values an expression x can have where the
    condition holds, when it's a comparison of x with a number (unsigned) -
    or the success of a call to a precompiled contract that returns as many
    bytes every time: then return_data.size is that.
    """
    neg = False
    while opcode(cond) in ("iszero", "bool"):
        if opcode(cond) == "iszero":
            neg = not neg
        cond = cond[1]
    if cond in PRECOMPILED_OUTPUT:
        n = PRECOMPILED_OUTPUT[cond]
        return {} if neg else {"returndatasize": (n, n)}
    op = opcode(cond)
    if op not in ("lt", "gt", "le", "ge", "eq", "slt", "sgt", "sle", "sge"):
        return {}
    if len(cond) != 3:
        return {}
    a, b = cond[1], cond[2]
    if type(a) is int and type(b) is not int:
        # c < x is x > c...
        a, b = b, a
        op = {"lt": "gt", "gt": "lt", "le": "ge", "ge": "le", "eq": "eq"}.get(
            op, {"slt": "sgt", "sgt": "slt", "sle": "sge", "sge": "sle"}.get(op)
        )
    if type(b) is not int or type(a) is int or not 0 <= b < 2**256:
        return {}
    if op[0] == "s":
        # signed: as unsigned, of a size far below 2**255 (see
        # memloc.max_value_bits) and a positive number
        if max_value_bits(a) >= 255 or b >= 2**255:
            return {}
        op = op[1:]
    if neg:
        if op == "eq":
            return {}
        op = {"lt": "ge", "ge": "lt", "gt": "le", "le": "gt"}[op]
    top = 2**256 - 1
    lo, hi = {
        "lt": (0, b - 1),
        "le": (0, b),
        "gt": (b + 1, top),
        "ge": (b, top),
        "eq": (b, b),
    }[op]
    if lo > hi:
        return {}
    res = {a: (lo, hi)}
    if opcode(a) == "min":
        # (each of them is at least what the smaller one is)
        for x in a[1:]:
            if type(x) is not int:
                res[x] = (lo, top)
    return res


def decide_le(a, b, bounds):
    """
    a <= b by the bounds of cond_bounds, None if they don't say - as the
    words they are, as the bounds are (-1 is 2**256 - 1).
    """
    if type(a) is int and b in bounds:
        lo, hi = bounds[b]
        a %= 2**256
        return True if lo >= a else False if hi < a else None
    if type(b) is int and a in bounds:
        lo, hi = bounds[a]
        b %= 2**256
        return True if hi <= b else False if lo > b else None
    return None


def low_zero_bits(exp):
    """How many of the lowest bits of exp are known to be 0."""
    if type(exp) == int:
        exp %= 2**256
        return 256 if exp == 0 else (exp & -exp).bit_length() - 1

    if m := match(exp, ("mask_shl", Any, ":int:off", ":int:shl", Any)):
        return max(0, m.off + m.shl)

    if opcode(exp) == "mul":
        return min(256, sum(low_zero_bits(e) for e in exp[1:]))

    if opcode(exp) == "add":
        return min(low_zero_bits(e) for e in exp[1:])

    return 0


def equals(x, c):
    """c + x == 0, as a comparison."""
    if m := match(x, ("mul", -1, ":y")):
        return ("eq", m.y, c)

    return ("eq", x, to_real_int(-c % 2**256))


def difference_eq(exp):
    """
    a == b if exp is a - b (a sum with one term subtracted, and no constant),
    None otherwise: the difference is 0 when they are equal.
    """
    if opcode(exp) != "add" or len(exp) < 3:
        return None

    terms = exp[1:]
    negated = [t for t in terms if match(t, ("mul", -1, Any))]
    if len(negated) != 1 or any(type(t) == int for t in terms):
        return None

    others = [t for t in terms if t is not negated[0]]
    return ("eq", add_op(*others), negated[0][2])


def truth(cond):
    """A simpler condition that holds when cond isn't 0."""
    if m := match(cond, ("mul", -1, ":x")):
        return truth(m.x)

    if (m := match(cond, ("add", ":int:c", ":x"))) and len(cond) == 3:
        return is_zero(equals(m.x, m.c))

    if eq := difference_eq(cond):
        return is_zero(eq)

    if m := match(cond, ("xor", ":a", ":b")):
        return is_zero(("eq", m.a, m.b))

    return cond


def cleanup_conds(trace):
    """

    removes ifs/whiles with conditions that are obviously true
    and replace variables that need to be equal to a constant by that constant

    """

    res = []

    for line in trace:
        if opcode(line) == "while":
            _, cond, path, jds, setvars = line
            # we're not evaluating symbolically, otherwise stuff like
            # stor0 <= stor0 + 1 gets evaluated to `True` - this happens
            # because we're truncating mask256. it should really be
            # mask(256, stor0) <= mask(256, stor0 + 1)
            # which is not always true
            # see 0x014B50466590340D41307Cc54DCee990c8D58aa8.transferFrom
            path = cleanup_conds(path)
            ev = arithmetic.eval_bool(cond, symbolic=False)
            if ev is True:
                res.append(("while", ("bool", 1), path, jds, setvars))
            elif ev is False:
                # removing loop altogether - but not what it sets its
                # variables to before it, which what follows may read
                res.extend(setvars)
            else:
                res.append(("while", cond, path, jds, setvars))

        elif opcode(line) == "if":
            _, cond, if_true, if_false = line
            cond = truth(cond)
            if_true = cleanup_conds(if_true)
            if_false = cleanup_conds(if_false)

            # If the condition is always true/false, remove the if.
            ev = arithmetic.eval_bool(cond, symbolic=False)
            if ev is True:
                res.extend(if_true)
            elif ev is False:
                res.extend(if_false)
            elif not if_true and not if_false:
                # can happen once the branches of a merged if got simplified
                pass
            elif if_true == if_false:
                # whatever the condition
                res.extend(if_true)
            elif not if_true:
                res.append(("if", is_zero(cond), if_false, []))
            else:
                res.append(("if", cond, if_true, if_false))

        else:
            res.append(line)

    return res


@cached
def find_mems(exp):
    def f(exp):
        if opcode(exp) == "mem":
            return set([exp])
        else:
            return set()

    return find_f_set(exp, f)


test_e = ("x", "sth", ("mem", 4), ("t", ("mem", 4), ("mem", 8), ("mem", ("mem", 64))))
assert find_mems(test_e) == {
    ("mem", 64),
    ("mem", ("mem", 64)),
    ("mem", 4),
    ("mem", 8),
}, find_mems(test_e)


def overwrites_mem(line, mem_idx, breaks=True):
    """
    for a given line, returns True if it potentially
    overwrites *any part* of memory index, False if it *for sure* doesn't
    (of a loop without breaks, as its next iterations see it: not what it
    writes on its way out)

    """
    if m := match(line, ("setmem", ":set_idx", Any)):
        return range_overlaps(m.set_idx, mem_idx) is not False

    if opcode(line) == "while":
        return while_touches_mem(line, mem_idx, breaks)

    if opcode(line) == "if":
        # matters for what comes after the if, when its branches merge again
        return any(overwrites_mem(l, mem_idx, breaks) for l in line[2] + line[3])

    if opcode(line) == "internal":
        # a recursive internal function (see vm.VM.internal) may write any
        # memory
        return True

    return False


def min_ifs(trace):
    """
    `if a > b: v = b else: v = a` is `v = min(a, b)` - the size of what a
    call returned that solc decodes, at most as much as it expects.
    """
    res = []
    for line in trace:
        if opcode(line) == "while":
            line = ("while", line[1], min_ifs(line[2])) + tuple(line[3:])
        elif opcode(line) == "if" and len(line) == 4:
            _, cond, if_true, if_false = line
            if_true, if_false = min_ifs(if_true), min_ifs(if_false)
            line = ("if", cond, if_true, if_false)
            if (
                len(if_true) == 1
                and len(if_false) == 1
                and (mt := match(if_true[0], ("setvar", ":v", ":x")))
                and (mf := match(if_false[0], ("setvar", ":v", ":y")))
                and mt.v == mf.v
            ):
                x, y = mt.x, mf.y
                while opcode(cond) == "iszero":
                    cond, x, y = cond[1], y, x
                op = opcode(cond)
                if op in ("lt", "le", "gt", "ge") and len(cond) == 3:
                    a, b = cond[1], cond[2]
                    if op in ("gt", "ge"):
                        a, b = b, a
                    # v is x where a < b (a <= b), y where it isn't
                    if (x, y) == (a, b):
                        line = ("setvar", mt.v, ("min", a, b))
        res.append(line)
    return res


def precompiled_results(trace, name=None):
    """
    `hash = sha256hash(x) # precompiled`: hash is the first word it returns,
    0 if none (OUTPUT.md) - what ext_call.return_data[0] is, until what
    calls return changes. Read as the result it is. (name: the result of the
    one that returned last, before the trace)
    """
    first_word = ("ext_call.return_data", 0, 32)
    res = []
    for line in trace:
        if opcode(line) == "if" and len(line) == 4:
            cond = (
                line[1] if name is None else replace(line[1], first_word, ("var", name))
            )
            line = (
                "if",
                cond,
                precompiled_results(line[2], name),
                precompiled_results(line[3], name),
            )
            if changes_reads(line, first_word):
                name = None
        elif opcode(line) == "while":
            # (what it's in isn't known at its start: it may have been changed)
            line = ("while", line[1], precompiled_results(line[2])) + tuple(line[3:])
            if changes_reads(line, first_word):
                name = None
        elif name is not None:
            if changes_reads(line, first_word):
                name = None
            else:
                line = replace(line, first_word, ("var", name))

        if m := match(line, ("precompiled", ":name", ":func", Any)):
            # (the data identity returns is the data it's given: read as that)
            name = m.name if m.func != "identity" else None
        res.append(line)
    return res


def changes_reads(line, exp):
    """
    True if the line may change something that exp reads from the state (see
    arithmetic.changed_by): the storage it writes, what a call can change...
    """
    op = opcode(line)
    if op == "store":
        return bool(changed_reads(exp, "store", line[3]))
    if op == "tstore":
        return bool(changed_reads(exp, "tstore", line[1]))
    if op in (
        "call",
        "staticcall",
        "delegatecall",
        "callcode",
        "codecall",
        "create",
        "create2",
    ):
        return bool(changed_reads(exp, op))
    if op == "precompiled":
        # a call too: what calls return is what it returns
        return bool(changed_reads(exp, "staticcall"))
    if op == "internal":
        # a recursive internal function (see vm.VM.internal): it may do
        # anything a call does
        return bool(changed_reads(exp, "delegatecall"))
    if op == "if":
        return any(changes_reads(l, exp) for l in line[2] + line[3])
    if op == "while":
        return any(changes_reads(l, exp) for l in line[2])
    return False


def affects(line, exp, breaks=True):
    if changes_reads(line, exp):
        return True

    if type(exp) != tuple and exp != "msize":
        return False

    s = str(exp)

    if "msize" in s:
        # the size of the memory used: any line may make it grow, by a write
        # or a read, what the output shows of them or not
        return True

    if "mem" not in s:
        return False

    mems = find_mems(exp)

    for m in mems:
        m_idx = m[1]
        if overwrites_mem(line, m_idx, breaks):
            return True

    return False


line_test = ("setmem", ("range", 65, 32), "x")
exp_test = ("mul", 8, ("mem", ("range", 64, 32)))
assert affects(line_test, exp_test) == True
exp_test = ("mul", 8, ("mem", ("range", 100, 32)))
assert affects(line_test, exp_test) == False

line_test = ("setmem", ("range", 65, 32), "x")
exp_test = ("mul", 8, ("mem", ("range", 64, 32)))
assert affects(line_test, exp_test) == True
exp_test = ("mul", 8, ("mem", ("range", 100, 32)))
assert affects(line_test, exp_test) == False

line_test = ("setmem", ("range", 65, "sth"), "x")
exp_test = ("mul", 8, ("mem", ("range", 64, 32)))
assert affects(line_test, exp_test) == True
exp_test = ("mul", 8, ("mem", ("range", 100, 32)))
assert affects(line_test, exp_test) == True

line_test = ("setmem", ("range", 65, 32), "x")
exp_test = ("mul", 8, ("mem", ("range", 64, 1)))
assert affects(line_test, exp_test) == False
exp_test = ("mul", 8, ("mem", ("range", 64, "sth")))
assert affects(line_test, exp_test) == True


"""

    Memory cleanup

"""


def trace_uses_mem(trace, mem_idx):
    """

    checks if memory is used anywhere in the trace

    """
    return _mem_use(trace, mem_idx) == USED


USED, OVERWRITTEN, NEITHER = "used", "overwritten", "neither"


def _mem_use(trace, mem_idx):
    """
    USED if the trace reads the memory, OVERWRITTEN if every path through it
    overwrites the memory (or ends the execution) before reading it, NEITHER
    if it does neither and what comes after the trace may still read it.
    """

    for idx, line in enumerate(trace):
        if m := match(line, ("setmem", ":memloc", ":memval")):
            memloc, memval = m.memloc, m.memval
            memval = simplify_exp(memval)

            # (what's written, and where: mem[mem[64]] = x reads mem[64])
            if exp_uses_mem(memval, mem_idx) or exp_uses_mem(memloc, mem_idx):
                return USED

            if any(overwrites_mem(line, inner[1]) for inner in find_mems(mem_idx)):
                # where mem_idx is depends on the memory the line writes:
                # it isn't the same memory afterwards
                return USED

            split = memloc_overwrite(
                mem_idx, memloc
            )  # returns range that we're confident wasn't overwritten by memloc
            res2 = trace[idx + 1 :]
            res = OVERWRITTEN
            for s_idx in split:
                r = _mem_use(res2, s_idx)
                if r == USED:
                    return USED
                if r == NEITHER:
                    res = NEITHER

            return res

        elif opcode(line) == "while":
            if while_uses_mem(line, mem_idx):
                return USED

        elif m := match(line, ("if", ":cond", ":if_true", ":if_false")):
            if exp_uses_mem(m.cond, mem_idx):
                return USED

            res = []
            for branch in (m.if_true, m.if_false):
                r = _mem_use(branch, mem_idx)
                if r == USED:
                    return USED
                if r == NEITHER and trace_ends_execution(branch):
                    r = OVERWRITTEN
                res.append(r)

            if res == [OVERWRITTEN, OVERWRITTEN]:
                return OVERWRITTEN

        elif opcode(line) == "continue":
            return USED

        elif opcode(line) in ("internal", "leave"):
            # a recursive internal function may read any memory (see
            # vm.VM.internal), and its caller once it returns - but
            # solidity's scratch space, which is read where it's written (a
            # hash's words) - and what it's given, what it returns
            if not in_scratch(mem_idx) or exp_uses_mem(line, mem_idx):
                return USED
            if opcode(line) == "leave":
                return OVERWRITTEN

        else:
            if exp_uses_mem(line, mem_idx):
                return USED

            if opcode(line) in ENDS_EXECUTION:
                return OVERWRITTEN

    return NEITHER


ENDS_EXECUTION = ("revert", "return", "stop", "invalid", "assert_fail", "selfdestruct")


def in_scratch(mem_idx):
    """Whether mem_idx is in solidity's scratch space, the first 64 bytes."""
    m = match(mem_idx, ("range", ":int:begin", ":int:size"))
    return bool(m) and m.begin >= 0 and m.size >= 0 and m.begin + m.size <= 64


def trace_ends_execution(trace):
    """True if every path through the trace ends the execution."""
    if len(trace) == 0:
        return False

    last = trace[-1]

    if opcode(last) == "if":
        return trace_ends_execution(last[2]) and trace_ends_execution(last[3])

    return opcode(last) in ENDS_EXECUTION


def cleanup_mems(trace, used_after=None):
    """
    for every setmem, replace future occurences of it with it's value,
    if possible

    `used_after` is what gets executed after `trace`, when `trace` is a
    branch of an if whose branches merge again, or the body of a loop that
    a path leaves by ending.

    """

    used_after = used_after or []
    res = []

    for idx, line in enumerate(trace):
        if match(line, ("setmem", ":rng", ("mem", ":rng"))) or match(
            line, ("setmem", ("range", Any, 0), Any)
        ):
            # (a copy of memory onto itself, or of no bytes: nothing changes)
            continue

        if opcode(line) in ["call", "staticcall", "delegatecall", "callcode"]:
            fname, fdata = line[-2:]

            if match(fdata, ("mem", ("range", Any, -4))) or (
                fname is None and match(fdata, ("mem", ("range", Any, 0)))
            ):
                # (no data)
                line = line[:-2] + (None, None)

            res.append(line)

        elif m := match(line, ("setmem", ":mem_idx", ":mem_val")):
            mem_idx, mem_val = m.mem_idx, m.mem_val

            # find all the future occurences of var and replace if possible
            if affects(line, mem_val):
                remaining_trace = trace[idx + 1 :]
            else:
                # The following line is the cause for O(N^2) complexity,
                # as here we iterate over the trace, and below we call `replace_mem`
                # over everything that's left in the trace.
                remaining_trace = replace_mem(trace[idx + 1 :], mem_idx, mem_val)

            if trace_uses_mem(remaining_trace + used_after, mem_idx):
                res.append(line)

            res.extend(cleanup_mems(remaining_trace, used_after))

            break

        elif opcode(line) == "while":
            _, cond, path, *rest = line
            # a path of the body that ends leaves the loop (see whiles.py):
            # what follows the loop may read what it wrote - solc's return
            # data, written on each way out of the loop
            path = cleanup_mems(path, trace[idx + 1 :] + used_after)
            res.append(
                (
                    "while",
                    cond,
                    path,
                )
                + tuple(rest)
            )

        elif opcode(line) == "if":
            _, cond, if_true, if_false = line
            after = trace[idx + 1 :] + used_after
            if_true = cleanup_mems(if_true, after)
            if_false = cleanup_mems(if_false, after)
            res.append(("if", cond, if_true, if_false))

        else:
            res.append(line)

    return res


@cached
def replace_mem_exp(exp, mem_idx, mem_val):
    if type(exp) != tuple:
        return exp

    res = tuple(
        replace_mem_exp(e, mem_idx, mem_val) if type(e) == tuple else e for e in exp
    )

    if opcode(mem_val) not in ("mem", "var", "data"):
        if m := match(
            res, ("delegatecall", ":gas", ":addr", ("mem", ":func"), ("mem", ":args"))
        ):
            op, f_begin, f_len = m.func
            assert op == "range"
            op, a_begin, a_len = m.args
            assert op == "range"
            if f_len == 4 and sub_op(add_op(f_begin, f_len), a_begin) == 0:
                # we have a situation when inside memory is sth like: (range 96 4) (100 ...)
                # let's merge those two memories, and try to replace with mem exp
                res_range = simplify_exp(("range", f_begin, add_op(f_len, a_len)))
                if res_range == mem_idx:
                    res = ("delegatecall", m.gas, m.addr, None, mem_val)

        if m := match(
            res, ("call", ":gas", ":addr", ":value", ("mem", ":func"), ("mem", ":args"))
        ):
            op, f_begin, f_len = m.func
            assert op == "range"
            op, a_begin, a_len = m.args
            assert op == "range"
            if f_len == 4 and sub_op(add_op(f_begin, f_len), a_begin) == 0:
                # we have a situation when inside memory is sth like: (range 96 4) (100 ...)
                # let's merge those two memories, and try to replace with mem exp
                res_range = simplify_exp(("range", f_begin, add_op(f_len, a_len)))
                if res_range == mem_idx:
                    res = ("call", m.gas, m.addr, m.value, None, mem_val)

    if res != exp:
        res = simplify_exp(res)

    if opcode(res) == "mem":
        assert len(res) == 2
        res = fill_mem(res, mem_idx, mem_val)

    return res


def replace_mem(trace, mem_idx, mem_val):
    """

    replaces any reference to mem_idx in the trace
    with a value of mem_val, up until a point of that mem being
    overwritten

    mem[64] = 'X'
    log mem[64]
    mem[65] = 'Y'
    log mem[64 len 1]
    log mem[65]
    mem[63] = 'Z'
    ...

    into

    mem[64] = 'X'
    log 'X'
    mem[65] = 'Y'
    log mask(1, 'X')
    log mem[65]
    ... (the rest unchanged)

    """
    mem_idx = simplify_exp(mem_idx)
    mem_val = simplify_exp(mem_val)
    mem_id = ("mem", mem_idx)

    if type(mem_val) is tuple and opcode(mem_val) != "mem":
        mem_val = arithmetic.eval(mem_val)

    res = []
    # the variables set on the way, and to what: a condition on one of them
    # is on what it was set to (see apply_constraint)
    defined = {}

    for idx, line in enumerate(trace):
        if m := match(line, ("setmem", ":memloc", Any)):
            # replace in val, and in where it's written - that is known
            # before the line changes the memory
            line = replace_mem_exp(line, mem_idx, mem_val)
            res.append(line)
            memloc = simplify_exp(line[1])
            # (None: it may overlap)
            if range_overlaps(memloc, mem_idx) is not False and affects(line, mem_val):
                # what the value was read from changes too: the rest of it,
                # not overwritten, isn't what it was read as any more
                res.extend(copy(trace[idx + 1 :]))
                return res
            if range_overlaps(memloc, mem_idx) is not False:
                split = splits_mem(mem_idx, memloc, mem_val)
                res2 = trace[idx + 1 :]
                for s in split:
                    res2 = replace_mem(res2, s[0], s[1])

                res.extend(res2)
                return res
            if affects(line, mem_val) or affects(line, mem_id):
                # (mem_id: where mem_idx is may depend on the memory the
                # line writes)
                res.extend(copy(trace[idx + 1 :]))
                return res

        elif opcode(line) == "if":
            _, cond, if_true, if_false = line
            cond = replace_mem_exp(cond, mem_idx, mem_val)
            known = replace_f(cond, lambda e: defined.get(e, e)) if defined else cond
            mem_idx_true = apply_constraint(mem_idx, known)
            mem_val_true = apply_constraint(mem_val, known)
            mem_idx_false = apply_constraint(mem_idx, is_zero(known))
            mem_val_false = apply_constraint(mem_val, is_zero(known))

            if_true = replace_mem(if_true, mem_idx_true, mem_val_true)
            if_false = replace_mem(if_false, mem_idx_false, mem_val_false)

            res.append(("if", cond, if_true, if_false))

            if affects(line, mem_val) or affects(line, mem_id):
                # one of the branches may have changed the memory, so
                # what comes after the if (if anything) is left alone
                res.extend(copy(trace[idx + 1 :]))
                return res

            # what comes after is run after the branch that doesn't end the
            # execution, if one does: where its condition holds (`require
            # return_data.size > 31`, then the call wrote 32 bytes)
            if trace_ends_execution(if_true) and not trace_ends_execution(if_false):
                mem_idx, mem_val = mem_idx_false, mem_val_false
            elif trace_ends_execution(if_false) and not trace_ends_execution(if_true):
                mem_idx, mem_val = mem_idx_true, mem_val_true
            mem_id = ("mem", mem_idx)
            forget_vars(defined, assigned_vars(if_true + if_false))

        elif opcode(line) == "while" and not (
            affects(line, mem_val, breaks=False) or affects(line, mem_id, breaks=False)
        ):
            # (one whose iterations don't change it)
            _, cond, path, jds, vars = line
            vars = [replace_mem_exp(v, mem_idx, mem_val) for v in vars]
            cond = replace_mem_exp(cond, mem_idx, mem_val)
            path = replace_mem(path, mem_idx, mem_val)

            res.append(("while", cond, path, jds, vars))
            if affects(line, mem_val) or affects(line, mem_id):
                # it writes there on its way out, by a break: what follows it
                # reads what it wrote
                res.extend(copy(trace[idx + 1 :]))
                return res
            defined.clear()

        elif affects(line, mem_val) or affects(line, mem_id):
            if opcode(line) in (
                "call",
                "staticcall",
                "delegatecall",
                "callcode",
                "internal",
            ):
                # what it's called with is read before it runs, and changes
                # anything
                res.append(replace_mem_exp(line, mem_idx, mem_val))
                res.extend(copy(trace[idx + 1 :]))
                return res
            res.extend(copy(trace[idx:]))
            return res

        else:
            # speed
            test = "mem" in str(line)
            if (
                test
                and (m := match(mem_idx, ("add", Any, ("var", ":num"))))
                and str(("var", m.num)) not in str(line)
            ):
                test = False
            # / speed

            if test:
                l = replace_mem_exp(line, mem_idx, mem_val)
            else:
                l = line

            res.append(l)

            if m := match(l, ("setvar", ":name", ":val")):
                forget_vars(defined, [m.name])
                if not contains(m.val, ("var", m.name)):
                    defined[("var", m.name)] = m.val
            for k, v in list(defined.items()):
                if affects(l, v):
                    del defined[k]

    return res


def assigned_vars(trace):
    """the names of the variables the trace sets"""
    res = []
    for line in trace:
        if opcode(line) == "setvar":
            res.append(line[1])
        elif opcode(line) == "if" and len(line) == 4:
            res += assigned_vars(line[2]) + assigned_vars(line[3])
        elif opcode(line) == "while":
            res += assigned_vars(line[2]) + [
                v[1] for v in line[4] if opcode(v) == "setvar"
            ]
    return res


def forget_vars(defined, names):
    """what's known of the variables, once the ones named are set again"""
    for name in names:
        defined.pop(("var", name), None)
        for k, v in list(defined.items()):
            if contains(v, ("var", name)):
                del defined[k]


"""

    Variables cleanup

"""


def cleanup_vars(trace, required_after=None):
    required_after = required_after or []
    """
        for every var = mem declaration, replace future
        occurences of it, if possible

        var1 = mem[64]
        log var1
        mem[65] = 'Y'
        log var1

        into

        var1 = mem[64]
        log mem[64]
        mem[65] = 'Y'
        log var1

        for var declarations that are no longer in use, remove them

    """

    res = []

    for idx, line in enumerate(trace):
        if opcode(line) == "setvar":
            _, var_idx, var_val = line
            # find all the future occurences of var and replace if possible

            if exp_size(var_val) > MAX_EXP_SIZE:
                # the vm gave a name to this expression because it's huge,
                # inlining it would make everything else huge as well
                res.append(line)
                res.extend(
                    cleanup_vars(trace[idx + 1 :], required_after=required_after)
                )
                return res

            remaining_trace = replace_var(trace[idx + 1 :], var_idx, var_val)
            if (
                contains(remaining_trace, ("var", var_idx))
                or ("var", var_idx) in required_after
            ):
                res.append(line)

            res.extend(cleanup_vars(remaining_trace, required_after=required_after))
            return res

        elif opcode(line) == "while":
            _, cond, path, *rest = line
            # the next iteration may use what this one sets: the condition
            # is evaluated again, and the body runs again from its start
            path = cleanup_vars(
                path,
                required_after=required_after
                + find_op_list(trace[idx + 1 :], "var")
                + find_op_list(cond, "var")
                + find_op_list(path, "var"),
            )
            res.append(
                (
                    "while",
                    cond,
                    path,
                )
                + tuple(rest)
            )

            a = parse_counters(line)

            if "endvars" in a:
                remaining_trace = trace[idx + 1 :]
                for var_idx, var_val in a["endvars"].items():
                    remaining_trace = replace_var(remaining_trace, var_idx, var_val)

                res.extend(
                    cleanup_vars(
                        remaining_trace,
                        required_after=required_after
                        + find_op_list(trace[idx + 1 :], "var"),
                    )
                )
                return res

        elif opcode(line) == "if":
            _, cond, if_true, if_false = line
            # a variable set in a branch may be used after the if, when the
            # branches merge again
            required = required_after + find_op_list(trace[idx + 1 :], "var")
            if_true = cleanup_vars(
                if_true,
                required_after=(
                    required_after if trace_ends_execution(if_true) else required
                ),
            )
            if_false = cleanup_vars(
                if_false,
                required_after=(
                    required_after if trace_ends_execution(if_false) else required
                ),
            )
            res.append(("if", cond, if_true, if_false))
        else:
            res.append(line)

    return res


def sets_var(trace, var_idx):
    """True if the trace (or a loop in it) sets the var."""

    def is_set(exp):
        return [exp] if match(exp, ("setvar", var_idx, Any)) else []

    return bool(find_f_list(trace, is_set))


def replace_var(trace, var_idx, var_val):
    """
    replace occurences of var, if possible

    """

    var_id = ("var", var_idx)
    res = []

    for idx, line in enumerate(trace):
        if m := match(line, ("setmem", ":mem_idx", Any)):
            # this all seems incorrect, (plus 'affects' checks below, needs to be revisited)
            # memloc = ("mem", m.mem_idx)
            # replace in val
            res.append(replace(line, var_id, var_val))

            if affects(line, var_val):
                res.extend(copy(trace[idx + 1 :]))
                return res
            else:
                continue

        if opcode(line) == "while":
            _, cond, path, jd, setvars = line
            setvars = replace(setvars, var_id, var_val)

            if sets_var(path, var_idx):
                # the loop sets the var again: it has the value only until
                # then, not in the next iterations, nor after the loop
                res.append(("while", cond, path, jd, setvars))
                res.extend(copy(trace[idx + 1 :]))
                return res

            if not affects(line, var_val, breaks=False):
                # (what its iterations don't change: but for what it writes
                # on its way out, by a break - which what follows it reads)
                cond = replace(cond, var_id, var_val)
                path = replace_var(path, var_idx, var_val)

            line = ("while", cond, path, jd, setvars)

        if opcode(line) == "if":
            _, cond, if_true, if_false = line
            cond = replace(cond, var_id, var_val)
            if_true = replace_var(if_true, var_idx, var_val)
            if_false = replace_var(if_false, var_idx, var_val)
            res.append(("if", cond, if_true, if_false))

            if affects(line, var_val):
                # one of the branches may have changed the memory the value
                # depends on, so what comes after the if is left alone
                res.extend(copy(trace[idx + 1 :]))
                return res

        elif affects(line, var_val):
            # what the line itself reads is read before it changes anything
            if opcode(line) not in ("while", "if"):
                line = replace(line, var_id, var_val)
            res.append(line)
            res.extend(copy(trace[idx + 1 :]))
            return res

        elif opcode(line) == "while":
            assert not affects(line, var_val)
            res.append(line)  # could replace vars inside of while, skipping for now

        else:
            res.append(replace(line, var_id, var_val))

    return res


"""

    loop parsing

"""


def find_conts(trace):
    def check(line):
        if opcode(line) == "continue":
            return [line]
        else:
            return []

    return find_f_list(trace, check)


def swap_cond(cond):
    replacement = {
        "lt": "gt",
        "le": "ge",
        "gt": "lt",
        "ge": "le",
    }

    return (replacement[cond[0]], cond[2], cond[1])


def move_right(left, right, exp):
    assert type(right) != list
    assert type(left) != list
    if left == exp:
        return right

    if opcode(left) == "add":
        terms = left[1:]
        if exp not in terms:
            return None  # deep embedding unsupported
        for e in terms:
            if e != exp:
                if type(e) == int:
                    e = to_real_int(e)
                right = sub_op(right, e)

        return right

    if opcode(left) == "mul":
        terms = left[1:]
        if exp not in terms:
            return None  # deep embedding unsupported
        for e in terms:
            if e != exp:
                assert type(e) != list
                assert type(right) != list
                right = div_op(right, e)

        return right


def normalize(cond):
    if type(cond) == tuple:
        cond = tuple(cleanup_mul_1(cond))

    if opcode(cond) not in ("lt", "le", "gt", "ge"):
        cond = ("lt", 0, cond)
        return normalize(cond)

    left, right = cond[1], cond[2]
    vars_left = find_op_list(left, "var")
    vars_right = find_op_list(right, "var")

    left_vars = tuple(
        [e for e in vars_left if match(e, ("var", int))]
    )  # int = loop vars
    right_vars = tuple([e for e in vars_right if match(e, ("var", int))])

    if len(left_vars) + len(right_vars) != 1:
        return None

    if len(right_vars) == 1:
        return normalize(swap_cond(cond))

    assert len(left_vars) == 1 and len(right_vars) == 0, cond

    var = left_vars[0]

    if opcode(left) != "var":
        assert type(right) != list
        assert type(left) != list
        right = move_right(left, right, var)
        if right is None:
            return None
        left = var
        cond = (cond[0], left, right)

    if m := match(cond, ("lt", ":left", ":right")):
        cond = ("le", m.left, sub_op(m.right, 1))

    if m := match(cond, ("gt", ":left", ":right")):
        cond = ("ge", m.left, add_op(m.right, 1))

    return cond  # we end up with (gt/lt (var int) sth)


def find_setmems(trace):
    def check(line):
        if m := match(line, ("while", Any, ":path", ...)):
            sm = find_setmems(m.path)
            if len(sm) == 0:
                return []

            # for s in sm:
            #     s_idx = s[1]
            #                if 'var' in str(s_idx):
            #                    print(s_idx)
            #                    assert False

            return sm

        elif opcode(line) == "setmem":
            return [line]

        else:
            return []

    return walk_trace(trace, check)


def memloc_left(setmem):
    assert opcode(setmem) in ("setmem", "mem")
    memloc = setmem[1]
    op, loc, _ = memloc
    assert op == "range"
    return loc


def memloc_right(setmem):
    assert opcode(setmem) in ("setmem", "mem")
    memloc = setmem[1]

    op, loc, rlen = memloc
    assert op == "range"
    return add_op(loc, rlen)


def make_range(left, right):
    r_len = sub_op(right, left)

    # (empty if right is before left - as words, not wrapping around 2**256)
    if words(left, right) and mem_ge_zero(r_len) is False:
        return ("range", left, 0)
    else:
        return ("range", left, r_len)


def extract_paths(while_exp):
    op, _, trace, jd, setvars = while_exp
    assert op == "while"

    def f(trace, jd, so_far):
        # extract all the paths leading up to jd
        if len(trace) == 0:
            return []

        line = trace[0]

        # assert opcode(line) != 'while'

        if opcode(line) == "if":
            _, cond, if_true, if_false = line
            rest = trace[1:]  # non-empty when the branches merge again
            res_true = f(if_true + rest, jd, so_far + [("require", cond)])
            res_false = f(if_false + rest, jd, so_far + [("require", is_zero(cond))])
            return res_true + res_false

        if len(trace) == 1:
            if opcode(line) == "continue":
                return [so_far]
            else:
                return []

        return f(trace[1:], jd, so_far + [line])

    return f(trace, jd, [])


def extract_setmems(while_exp, breaks=True):
    """
    The setmems of the loop body that are on a path to a `continue`, or to
    its end - a break, after which what follows the loop runs (the ones on
    the paths that end the execution don't matter to what comes after it).
    Without breaks, those on a path to a continue only: what the loop's next
    iterations may read.

    Same as collecting them from extract_paths, without enumerating the paths,
    which are exponential in the number of ifs in the body.
    """
    op, _, trace, jd, setvars = while_exp
    assert op == "while"

    def f(trace, after):
        # `after`: whether a continue, or the end of the body, is reachable
        # from the end of `trace`. Returns the setmems on the paths through
        # `trace` that reach one, and whether there are such paths.
        res = []
        reach = after

        for line in reversed(trace):
            if opcode(line) == "continue":
                reach = True

            elif opcode(line) in ENDS_EXECUTION:
                reach = False

            elif opcode(line) == "if":
                res_true, reach_true = f(line[2], reach)
                res_false, reach_false = f(line[3], reach)
                res = res_true + res_false + res
                reach = reach_true or reach_false

            elif reach and opcode(line) in ("setmem", "while"):
                res = find_setmems([line]) + res

        return res, reach

    res, _ = f(trace, breaks)

    return list(dict.fromkeys(res))


def extract_mems(while_exp):
    paths = extract_paths(while_exp)
    res = []
    for p in paths:
        res += find_mems(p)
    return res


#        mems = extract_mems(path)


def while_touches_mem(line, mem_idx, breaks=True):
    a = parse_counters(line)
    op, cond, path, jds, setvars = line
    assert op == "while"

    #    try:
    setmems = extract_setmems(line, breaks)
    #    setmems = find_setmems(path)
    #    except Exception:
    #        return True

    if len(setmems) == 0:
        return False

    setmems_begin = setmems_end = setmems

    if "lastvars" not in a:
        for (
            s
        ) in (
            setmems
        ):  # if no endvars, comparing just with a 'var' assumes 'var' is any natural number
            if range_overlaps(mem_idx, s[1]) is not False:
                return True

        return False

    # (each all at once, see replace_vars)
    setmems_begin = replace_vars(setmems, {v[1]: v[2] for v in a["setvars"]})
    setmems_end = replace_vars(
        setmems, {v[1]: a["lastvars"][v[1]] for v in a["setvars"]}
    )

    for idx, _ in enumerate(setmems):
        r_begin = memloc_left(setmems_begin[idx])
        r_end = memloc_right(setmems_end[idx])

        r = make_range(r_begin, r_end)
        if range_overlaps(mem_idx, r) is not False:
            return True

        r_begin = memloc_left(setmems_end[idx])
        r_end = memloc_right(setmems_begin[idx])

        r = make_range(r_begin, r_end)
        if range_overlaps(mem_idx, r) is not False:
            return True

    return False


def while_uses_mem(line, mem_idx):
    op, cond, path, jds, setvars = line
    assert op == "while"
    a = parse_counters(line)

    mems = list(find_mems(line))

    #    mems = extract_mems(line)

    if len(mems) == 0:
        return False

    mems_begin = mems_end = mems

    if "lastvars" not in a:
        for s in mems:
            if range_overlaps(mem_idx, s[1]) is not False:
                return True

        return False

    # (each all at once, see replace_vars)
    mems_begin = replace_vars(mems, {v[1]: v[2] for v in a["setvars"]})
    mems_end = replace_vars(mems, {v[1]: a["lastvars"][v[1]] for v in a["setvars"]})

    for idx, _ in enumerate(mems):
        r_begin = memloc_left(mems_begin[idx])
        r_end = memloc_right(mems_end[idx])

        r = make_range(r_begin, r_end)
        if range_overlaps(mem_idx, r) is not False:
            return True

        r_begin = memloc_left(mems_end[idx])
        r_end = memloc_right(mems_begin[idx])

        r = make_range(r_begin, r_end)
        if range_overlaps(mem_idx, r) is not False:
            return True

    return False


def exp_uses_mem(exp, mem_idx):
    mems = find_mems([exp])

    for m in mems:
        op, m_idx = m
        assert op == "mem"
        if range_overlaps(m_idx, mem_idx) is not False:
            return True

    return False


# what can have a different value in every iteration of a loop: its
# variables, and what reads the state
LOOP_VARIANT = (
    "var",
    "storage",
    "tload",
    "mem",
    "msize",
    "gas",
    "balance",
    "extcodesize",
    "extcodehash",
    "returndatasize",
    "ext_call",
    "return_data",
    "return_code",
    "new_address",
    ".result",
)


def is_loop_invariant(exp):
    """True if exp is sure to have the same value in every iteration of a loop."""
    if type(exp) is int:
        return True

    if type(exp) is str:
        return not any(name in exp for name in LOOP_VARIANT)

    if type(exp) is not tuple:
        return False

    return all(is_loop_invariant(e) for e in exp)


def parse_counters(line):
    a = {}
    op, cond, path, jds, setvars = line
    assert op == "while"

    a["setvars"] = setvars
    a["jds"] = jds

    # the continues of this loop, not of the ones nested in it
    conts = [c for c in find_conts(path) if c[1] == jds]

    startvars = {}
    for v in setvars:
        assert (m := match(v, ("setvar", ":vidx", ":vval")))
        startvars[m.vidx] = m.vval

    cond = normalize(cond)
    if cond is None or not conts:
        return {}

    cont = conts[0]

    stepvars = {}
    for v in cont[2]:
        var_idx, var_val = v[1], v[2]
        stepvars[var_idx] = var_val

    a["stepvars"] = stepvars

    counter = cond[1][1]
    counter_stop = cond[2]
    if counter not in startvars:
        logger.warning("counter %s not in %s", counter, startvars)
        return {}
    counter_start = startvars[counter]
    a["counter"] = counter
    a["start"] = counter_start
    a["stop"] = counter_stop
    if counter not in stepvars:
        logger.warning("counter not in stepvars")
        counter_diff = 0
    else:
        counter_diff = stepvars[counter]

    if len(conts) > 1:
        return a

    if opcode(counter_diff) != "add" or len(counter_diff) != 3:
        return a

    if type(counter_diff[1]) != int:
        # not the counter plus a number
        return a

    counter_diff = (counter_diff[0], to_real_int(counter_diff[1]), counter_diff[2])

    # counter_diff[2] ~ ('mul', 1, X) -> counter_diff[2] = X
    if opcode(counter_diff[2]) == "mul" and counter_diff[2][1] == 1:
        counter_diff = (counter_diff[0], counter_diff[1], counter_diff[2][2])

    if counter_diff[2] != ("var", counter):
        # the counter set from another variable: not a step of it
        return a

    counter_step = to_real_int(counter_diff[1])
    a["step"] = counter_step

    num_loops = div_op(
        add_op(sub_op(counter_stop, counter_start), counter_step), counter_step
    )

    if opcode(num_loops) == "div":  # so, no obvious divider
        a["counter_stop"] = counter_stop
        a["counter_start"] = counter_start
        a["counter_step"] = counter_step
        return a

    a["num_loops"] = num_loops

    lastvars = {}
    for v in setvars:
        var_idx, var_val = v[1], to_real_int(v[2])
        step = stepvars.get(var_idx)
        if not (
            (
                match(step, ("add", ":diff", ("var", var_idx)))
                or match(step, ("add", ":diff", ("mul", 1, ("var", var_idx))))
            )
            and is_loop_invariant(step[1])
        ):
            # not a var that gets incremented by the same amount on every
            # iteration (e.g. one that is set in a branch, or `s += i * i`),
            # we can't tell its value after the loop
            return a
        var_diff = to_real_int(step[1])
        assert type(num_loops) != list
        var_stop = add_op(var_val, mul_op(var_diff, num_loops))
        lastvars[var_idx] = var_stop

    # the values after the last iteration if there is one, which is enough
    # to tell what memory the loop touches
    a["lastvars"] = lastvars

    # num_loops is the number of iterations only if the counter doesn't start
    # beyond where it stops - otherwise there are none, and it's negative
    if cond[0] == "le" and counter_step > 0:
        starts_before_stop = mem_le_op(
            counter_start, add_op(counter_stop, counter_step)
        )
    elif cond[0] == "ge" and counter_step < 0:
        starts_before_stop = mem_le_op(
            add_op(counter_stop, counter_step), counter_start
        )
    else:
        starts_before_stop = False

    if (
        starts_before_stop is True
        and not leaves_loop(path)
        and not any(changes_value(line, v) for line in path for v in lastvars.values())
    ):
        # the values after the loop - but where a break leaves it before, or
        # where what they're computed from (where the counter stops: the
        # length of an array the body pops) changes in the body
        a["endvars"] = lastvars

    return a


def changes_value(line, exp):
    """
    Whether the line changes what exp reads: of the state, or of memory it
    writes over for sure (where the parts of it may overlap is how memory
    is, a copy's source and destination: taken as apart, as elsewhere).
    """
    if changes_reads(line, exp):
        return True
    if opcode(line) == "setmem":
        return any(range_overlaps(line[1], m[1]) is True for m in find_mems(exp))
    if opcode(line) == "if":
        return any(changes_value(l, exp) for l in line[2] + line[3])
    if opcode(line) == "while":
        return any(changes_value(l, exp) for l in line[2])
    return False


def leaves_loop(path):
    """
    Whether a path of the body of a loop gets to its end: it leaves the loop
    there (a break, see prettify.add_breaks) for what follows it, rather than
    continue it or end the execution.
    """
    if not path:
        return True
    last = path[-1]
    if opcode(last) == "if":
        return len(last) < 4 or leaves_loop(last[2]) or leaves_loop(last[3])
    return opcode(last) not in ENDS_EXECUTION + (
        "continue",
        "undefined",
        "leave",
        "goto",
    )
