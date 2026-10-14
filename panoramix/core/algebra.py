#
#  Algebra handles symbolic operations and comparisons
#
#  E.g. if you want to figure out if ('ADD', 2, 'x') is bigger than 'x'.,
#       you call lt_op(add_op(2,'x'), 'x')
#
#  The code here is crazy-fragile and inconsistent.
#  Modifying it is akin to playing kal-toh ( https://www.youtube.com/watch?v=8i2idMe142s )
#
#  Should be refactored together with arithmetic.py
#  I considered replacing it with a generic SMT-solver, but I'm quite sure a generic
#  solution would be way slower.
#
#  A better option would be to use a system similar to ACL2.
#


import numbers
import logging

from panoramix.core import variants
from panoramix.matcher import Any, match
from panoramix.utils.helpers import (
    CACHES,
    EasyCopy,
    all_concrete,
    cached,
    clear_caches,
    opcode,
    to_exp2,
)

logger = logging.getLogger(__name__)


class CannotCompare(Exception):
    pass


# copied from other modules, need to rework module structures
# to avoid circular dependencies
def _clamp_bits(k):
    """
    masks and shifts never exceed 256 bits, but a variant of an expression
    (see variants.py) can put 2^230 there, and 2 ** that would exhaust the
    memory
    """
    return max(min(k, 4096), -4096)


def _pow2(k):
    return 2 ** _clamp_bits(k)


def mask_to_int(size, offset):
    size = _clamp_bits(size)
    offset = _clamp_bits(offset)

    if offset < 0:
        size = size + offset
        if size < 1:
            return 0
        return 2**size - 1

    if size < 0:
        return 0

    return (2**size - 1) * (2**offset)


def may_be_wide(exp):
    """
    True if exp may have more than 256 bits: a value from memory can be
    longer than a word (a string, the arguments of a call), and a mask of
    256 bits of it doesn't leave it as it is.
    """
    if type(exp) == int:
        return exp >= 2**256

    op = opcode(exp)
    if op == "data":
        return True

    if op == "bytes":
        return not (type(exp[1]) == int and exp[1] <= 32)

    if op == "mem" or op in (
        "call.data",
        "code.data",
        "ext_call.return_data",
        "delegate.return_data",
        "callcode.return_data",
        "staticcall.return_data",
    ):
        if op == "mem":
            rng = exp[1]
            length = rng[2] if opcode(rng) == "range" and len(rng) == 3 else None
        else:
            length = exp[2] if len(exp) == 3 else None
        return not (type(length) == int and length <= 32)

    return False


# what no execution can make large: sizes of what gas pays for
BOUNDED_SYMBOLS = {
    "calldatasize": 64,
    "returndatasize": 64,
    "codesize": 64,
    "msize": 64,
    "gas": 64,
}

WORD_TOP = 2**256 - 1

# the highest the words memory addresses and sizes are made of can be: gas
# keeps them far below 2**256 (see memory_range)
MEMORY_TOP = 2**64 - 1


def value_range(exp, bounds=None, top=WORD_TOP):
    """
    (lowest, highest): what exp can be, as an integer - for any value of the
    words it's made of (between 0 and top - 2**256 - 1, or MEMORY_TOP, see
    memory_range - less for BOUNDED_SYMBOLS, and for what bounds says: {x:
    (lowest, highest)}).

    A sum or a product (add, mul) is the integer it is here: a sum of words
    may be negative, or above 2**256. That's how the fields of a mask are
    read (its size, offset and shift, see mask_op) - a word from the EVM is
    one of them only when it's the same integer (see shl_op). Anything else
    is a word.
    """
    return _value_range(exp, bounds, top)


def memory_range(exp):
    """
    value_range of a memory address or size (or what's computed from them):
    the words it's made of are below 2**64 - no execution can pay for the
    memory of a larger one (the documented assumption of the memory model,
    see memloc) - and it's the integer it is, not the word modulo 2**256.
    """
    return value_range(exp, top=MEMORY_TOP)


@cached
def _cached_value_range(exp, top):
    return _value_range_of(exp, None, top)


def _linear(exp, bounds, masks=True, top=WORD_TOP):
    """
    exp as a sum: ({term: coefficient}, constant, (lowest, highest) of a
    number added to them). masks: floor32(x)... as x minus a number - or,
    "exact", only 2 * x... (no number subtracted: two of them that are the
    same mask stay the same term, and cancel each other).
    """
    if type(exp) in (int, bool):
        return {}, int(exp), (0, 0)

    op = opcode(exp)
    if op == "add":
        terms, const, lo, hi = {}, 0, 0, 0
        for e in exp[1:]:
            t, c, (d_lo, d_hi) = _linear(e, bounds, masks, top)
            const, lo, hi = const + c, lo + d_lo, hi + d_hi
            for k, v in t.items():
                terms[k] = terms.get(k, 0) + v
        return terms, const, (lo, hi)

    if op == "mul" and len(exp) >= 3 and type(exp[1]) is int:
        rest = exp[2] if len(exp) == 3 else ("mul",) + exp[2:]
        k = exp[1]
        t, c, (d_lo, d_hi) = _linear(rest, bounds, masks, top)
        d = (k * d_lo, k * d_hi) if k >= 0 else (k * d_hi, k * d_lo)
        return {x: v * k for x, v in t.items()}, c * k, d

    if (
        masks
        and (m := match(exp, ("mask_shl", ":int:size", ":int:off", ":int:shl", ":x")))
        and 0 < m.size
        and 0 <= m.off <= 16
        and (masks is True or m.off == 0)
        and 0 <= m.shl
    ):
        # x with its lowest off bits cleared (floor32(x); ceil32(y) is
        # floor32(y + 31)), times 2**shl (8 * x, a mask moved left by 3):
        # 2**shl * (x - d), d its bits cleared, from 0 to 2**off - 1 - the
        # same d for the same mask, wherever it is (2 * ceil32(x) - ceil32(x)
        # is ceil32(x)), when the mask doesn't cut the top of x, nor the
        # word the top of what it's moved to (32 * (i + 1), from Vyper)
        lo, hi = _value_range(m.x, bounds, top)
        if 0 <= lo and hi < 2 ** (m.off + m.size) and hi << m.shl <= WORD_TOP:
            t, c, (d_lo, d_hi) = _linear(m.x, bounds, masks, top)
            k = 2**m.shl
            t = {x: v * k for x, v in t.items()}
            if m.off:
                t[("cleared", exp)] = t.get(("cleared", exp), 0) - k
            return t, c * k, (k * d_lo, k * d_hi)

    return {exp: 1}, 0, (0, 0)


def _value_range(exp, bounds, top=WORD_TOP):
    # (what's known of an expression is known of it wherever it is: each of
    # its words, each of its terms is looked at once - not again in each of
    # the passes of each sum it's in, an exponential number of times)
    if not bounds:
        return _cached_value_range(exp, top)
    return _value_range_of(exp, bounds, top)


def _value_range_of(exp, bounds, top=WORD_TOP):
    # (the masks as numbers minus others know sums such as ceil32(x) - x,
    # as what they are that each of them is a word, and with only the exact
    # ones as numbers 2 * (ceil32(x) + 32) - ceil32(x): all of them hold)
    ranges = [_sum_range(exp, bounds, m, top) for m in (True, "exact", False)]
    return max(r[0] for r in ranges), min(r[1] for r in ranges)


def _sum_range(exp, bounds, masks, top=WORD_TOP):
    terms, const, (lo, hi) = _linear(exp, bounds, masks, top)
    lo, hi = lo + const, hi + const
    for t, c in terms.items():
        if c == 0:
            continue
        t_lo, t_hi = _term_range(t, bounds, top)
        if c > 0:
            lo, hi = lo + c * t_lo, hi + c * t_hi
        else:
            lo, hi = lo + c * t_hi, hi + c * t_lo
    return lo, hi


def _word_top(exp, bounds, top=WORD_TOP):
    """The highest value of exp as a word."""
    lo, hi = _value_range(exp, bounds, top)
    return hi if 0 <= lo and hi <= WORD_TOP else WORD_TOP


def _term_range(t, bounds, top=WORD_TOP):
    if not bounds:
        return _cached_term_range(t, top)
    return _term_range_of(t, bounds, top)


@cached
def _cached_term_range(t, top):
    return _term_range_of(t, None, top)


def _term_range_of(t, bounds, top=WORD_TOP):
    """
    value_range of what isn't a sum, nor a number times something. With a
    top below WORD_TOP (see memory_range), a term of a memory address or
    size is below it - what it's made of isn't: the top byte of a word, as
    Solady's LibString.unpackTwo reads a length, is the byte it is. Nor is
    a word made of a number below 0: 8 * (32 - x % 32) as 256 + Mask(253, 0,
    3, -(x % 32)) is small only modulo 2**256, its mask is the large number
    it is. Nor is one made of what the contract is given, unchecked (see
    _of_unchecked) - unless it's a memory size itself, or checked to be
    small (see set_variables).
    """
    lo, hi = _whole_term_range(t, bounds, top)
    if (
        top < WORD_TOP
        and lo <= top
        and (t in _SMALL or not (_of_wrapped(t, bounds) or _of_unchecked(t)))
    ):
        hi = min(hi, top)
    return lo, hi


def _of_wrapped(t, bounds):
    """Whether t is a word made of what may not be one: a mask of -x..."""
    if opcode(t) == "mask_shl" and len(t) == 5:
        args = t[4:]
    elif opcode(t) in ("div", "mod", "and", "or", "xor", "min", "max"):
        args = t[1:]
    else:
        return False
    for a in args:
        lo, hi = (a, a) if type(a) is int else _value_range(a, bounds)
        if lo < 0 or hi > WORD_TOP:
            return True
    return False


# the words a contract is given - of its calldata, of what another one
# returned - that may be anything: a memory address or size made of them may
# wrap around 2**256 (see _of_unchecked), unless the contract checks them
INPUTS = ("cd", "call.data", "ext_call.return_data")


@cached
def _of_unchecked(t):
    """
    Whether t is made of a word the contract is given (see unchecked): `p +
    160`, with p a param added to a pointer, unchecked (Solady's
    LibBytes.load), may wrap around 2**256 - mem[159] for a p of 2**256 - 1.
    """
    return unchecked(t, _VARIABLES, _SMALL)


# what a word is computed from, its value made of theirs - not what's read
# where they say, in storage, in memory...
COMPUTED = (
    ("add", "mul", "div", "sdiv", "mod", "smod", "exp", "addmod", "mulmod")
    + ("signextend", "and", "or", "xor", "not", "mask_shl", "shl", "shr", "sar")
    + ("byte", "clz", "min", "max", "bytes", "data")
)


def unchecked(exp, values, small):
    """
    Whether exp is made of a word the contract is given (INPUTS) that isn't
    known to be small - in small (see set_variables) - computed from it
    (COMPUTED), or a variable set to it: one of the values it's set to
    (values, {name: [values]}).
    """
    seen = set()

    def f(e):
        if type(e) is not tuple or not e or e in small:
            return False
        if opcode(e) in INPUTS:
            return True
        if opcode(e) == "var" and len(e) == 2:
            if e[1] in values and e[1] not in seen:
                seen.add(e[1])
                return any(f(v) for v in values[e[1]])
            return False
        return opcode(e) in COMPUTED and any(f(x) for x in e[1:])

    return f(exp)


# what the variables of the trace being simplified are set to (see
# set_variables), the ones a value of which is made of them (a loop's
# counter), and the ones whose values are being looked at - and the words
# of it known to be small
_VARIABLES = {}
_CYCLIC = set()
_VISITING = set()
_SMALL = frozenset()


def set_variables(values, small=()):
    """
    {name: [values]}: what each variable of the trace being simplified is set
    to, anywhere in it - its value is one of these (see simplify_trace). And
    small: the words of it below 2**64 wherever they're in a memory address
    or size, as no other word the contract is given is (see
    simplify.small_words). What was decided with others is forgotten
    (clear_caches).
    """
    global _VARIABLES, _CYCLIC, _SMALL
    _VARIABLES = values
    _SMALL = frozenset(small)
    _CYCLIC = set(
        name
        for name, vals in values.items()
        if any(mentions_var(v, name) for v in vals)
    )
    clear_caches()


def _whole_term_range(t, bounds, top=WORD_TOP):
    if bounds and t in bounds:
        return bounds[t]

    if opcode(t) == "cleared":
        # the bits a mask clears (see _linear)
        return 0, 2 ** t[1][2] - 1

    if (
        opcode(t) == "var"
        and len(t) == 2
        and t[1] in _VARIABLES
        and t[1] not in _CYCLIC
        and t[1] not in _VISITING
    ):
        # one of the values it's set to - a snapshot of the free memory
        # pointer is one, at least 0x60 - unless one of them is made of it (a
        # loop's counter): then any word
        _VISITING.add(t[1])
        try:
            ranges = [_value_range(v, bounds, top) for v in _VARIABLES[t[1]]]
        finally:
            _VISITING.discard(t[1])
        lo, hi = min(r[0] for r in ranges), max(r[1] for r in ranges)
        if 0 <= lo and hi <= WORD_TOP:
            return lo, hi
        return 0, WORD_TOP

    if type(t) in (int, bool):
        return int(t), int(t)

    if type(t) is str:
        return 0, 2 ** BOUNDED_SYMBOLS.get(t, 256) - 1

    op = opcode(t)

    if op in BOOL_OPS:
        return 0, 1

    if op == "mul":
        # a product of what isn't numbers
        lo = hi = 1
        for f in t[1:]:
            f_lo, f_hi = _value_range(f, bounds)
            cands = (lo * f_lo, lo * f_hi, hi * f_lo, hi * f_hi)
            lo, hi = min(cands), max(cands)
        return lo, hi

    if op == "clz":
        # the leading zero bits of a word
        return 0, 256

    # the rest are words
    if (m := match(t, ("mod", ":x", ":int:c"))) and 0 < m.c:
        return 0, min(m.c - 1, _word_top(m.x, bounds))

    if (m := match(t, ("div", ":x", ":int:c"))) and 0 < m.c:
        return 0, _word_top(m.x, bounds) // m.c

    if m := match(t, ("mod", Any, ":y")):
        # below what it's divided by (0 by 0): an index, i % array.length
        return 0, max(_word_top(m.y, bounds, top) - 1, 0)

    if m := match(t, ("mask_shl", ":int:size", ":int:off", ":int:shl", ":x")):
        high_bit = _clamp_bits(m.off + m.size)
        if not may_be_wide(m.x):
            high_bit = min(high_bit, _word_top(m.x, bounds).bit_length())
        bottom = max(m.off, 0)
        if high_bit <= bottom:
            return 0, 0
        high = 2**high_bit - 2**bottom
        high = high << _clamp_bits(m.shl) if m.shl >= 0 else high >> -m.shl
        return 0, min(high, WORD_TOP)

    if (m := match(t, ("storage", ":int:size", ":int:off", Any))) and m.off >= 0:
        return 0, 2 ** max(0, min(m.size, 256)) - 1

    if t == ("mem", ("range", 64, 32)) and variants.FREE_MEMORY_POINTER:
        # solidity's free memory pointer: 0x80 from the start on (0x60 before
        # 0.4.22), and above as it's moved
        return 0x60, WORD_TOP

    if op == "and" and len(t) > 1:
        return 0, min(_word_top(e, bounds) for e in t[1:])

    if op in ("or", "xor") and len(t) > 1:
        bits = max(_word_top(e, bounds).bit_length() for e in t[1:])
        return 0, 2**bits - 1

    if op in ("min", "max") and len(t) > 1:
        ranges = [_value_range(e, bounds) for e in t[1:]]
        if all(0 <= lo and hi <= WORD_TOP for lo, hi in ranges):
            pick = min if op == "min" else max
            return pick(r[0] for r in ranges), pick(r[1] for r in ranges)

    return 0, WORD_TOP


def mentions_var(exp, name):
    """Whether exp reads the variable, or one whose values do (see set_variables)."""
    seen = set()

    def f(e):
        if type(e) is not tuple:
            return False
        if opcode(e) == "var" and len(e) == 2:
            if e[1] == name:
                return True
            if e[1] in _VARIABLES and e[1] not in seen:
                seen.add(e[1])
                return any(f(v) for v in _VARIABLES[e[1]])
            return False
        return any(f(x) for x in e[1:])

    return f(exp)


def is_word(exp, bounds=None):
    """Whether exp, as an integer (see value_range), is a word."""
    lo, hi = value_range(exp, bounds)
    return 0 <= lo and hi <= WORD_TOP


def shift_sign(exp):
    """
    1 if exp, as an integer (see value_range), is a word (a shift left by
    it), -1 if minus it is (a shift right), 0 if it's 0, None if it's not
    known which.
    """
    if exp == 0:
        return 0
    lo, hi = value_range(exp)
    if 0 <= lo and hi <= WORD_TOP:
        return 1
    if -WORD_TOP <= lo and hi <= 0:
        return -1
    return None


def readable_mask(size, offset, shl):
    """
    Whether a mask with these can be printed as what it is: its size and
    offset words, and its shift one way or the other (see shift_sign).
    """
    return is_word(size) and is_word(offset) and shift_sign(shl) is not None


@cached
def simplify(exp):
    if opcode(exp) == "max":
        terms = exp[1:]
        els = [simplify(e) for e in terms]
        res = -(2**256)
        for e in els:
            try:
                res = max_op(res, e)
            except Exception:
                return ("max",) + tuple(els)
        return res

    if m := match(exp, ("mask_shl", ":size", ":offset", ":shl", ":val")):
        size, offset, shl, val = (
            simplify(m.size),
            simplify(m.offset),
            simplify(m.shl),
            simplify(m.val),
        )

        if all_concrete(size, offset, shl, val):
            return apply_mask(val, size, offset, shl)

        if (size, offset, shl) == (256, 0, 0) and not may_be_wide(val):
            return val

    if opcode(exp) == "add":
        res = 0
        for e in exp[1:]:
            res = add_op(res, simplify(e))

        assert type(res) != list
        return res

    if opcode(exp) == "mul":
        res = 1
        for e in exp[1:]:
            res = mul_op(res, simplify(e))
        assert type(res) != list
        return res

    return exp


def calc_max(exp):
    if type(exp) != tuple:
        return exp

    exp = (opcode(exp),) + tuple(calc_max(e) for e in exp[1:])

    if opcode(exp) == "max":
        m = -(2**256)
        for e in exp[1:]:
            if type(e) != int:
                break
            m = max(m, e)
        else:
            return m

    return exp


def minus_op(exp):
    return mul_op(-1, exp)


def sub_op(left, right):
    if (type(left), type(right)) == (int, int):
        return left - right  # optimisation

    if left == 0:
        return minus_op(right)

    if right == 0:
        return left

    return add_op(left, minus_op(right))


def flatten_adds(exp):
    res = exp

    while any(opcode(a) == "add" for a in res):
        exp = []
        for r in res:
            if opcode(r) == "add":
                assert len(r[1:]) > 1
                exp += r[1:]
            else:
                exp.append(r)

        res = exp

    return res


def max_to_add(exp):
    if opcode(exp) != "max":
        return exp

    exp = exp[1:]

    for e in exp:
        if opcode(e) != "add" and type(e) != int:
            return simplify_max(("max",) + exp)

    for e in exp:
        if type(e) == int:
            m = min(
                x
                if type(x) == int
                else (
                    x[1] if type(x) == tuple and len(x) > 1 and type(x[1]) == int else 0
                )
                for x in exp
            )
            # used to be x[1] but 0x0000136DAE58AFCF1EDd2071973d4a7a6fbe98A5 didn't work
            res = ("max", e - m)
            for e2 in exp:
                if e2 != e:
                    res += (sub_op(e2, m),)

            return ("add", m, res)

    m = 10**20
    for e in exp:
        if type(e[1]) != int:
            m = 0
            break
        else:
            m = min(m, e[1])

    common = []
    first = exp[0]
    for f in first:
        if all(f in e[1:] for e in exp[1:]):
            common.append(f)

    if len(common) > 0:
        a = add_op(m, *common)
    else:
        a = m

    res = []
    for e in exp:
        res.append(sub_op(e, a))

    if type(a) == int:
        prefix = (a,)
    else:
        prefix = a[1:]

    return ("add",) + prefix + (simplify_max(("max",) + tuple(res)),)


@cached
def add_op(*args):
    if len(args) == 1:
        return args[0]
    elif len(args) == 0:
        return 0

    assert len(args) > 1
    assert (
        "mul" not in args
    )  # some old bug, it's ok for ['mul'..] to be in args, but not 'mul' directly

    # speed optimisation
    real = 0
    for r in args:
        if type(r) in (int, float):
            real += r
        else:
            break
    else:
        return real
    # / speed

    res = flatten_adds(list(args))

    for idx, r in enumerate(res):
        if opcode(r) != "mul":
            res[idx] = mul_op(1, r)

    real = 0
    symbolic = []

    for r in res:
        assert opcode(r) != "add"

        if type(r) in (int, float):
            real += r
            continue

        assert opcode(r) == "mul"

        # look at all the previously found symbolic expressions
        # perhaps you can add to the previous one - if so, do it
        # else, add this as a new symbolic exp
        for idx, rr in enumerate(symbolic):
            tried = try_add(r, rr) or try_add(rr, r)

            if tried:
                if opcode(tried) == "mul":
                    symbolic[idx] = tried

                elif m := match(tried, ("mask_shl", ":int:size", 0, ":osize", ":val")):
                    assert m.osize == 256 - m.size
                    symbolic[idx] = ("mul", 2**m.osize, m.val)

                else:
                    m = match(tried, ("add", ":int:num", ":term"))
                    assert m
                    symbolic[idx] = m.term
                    real += m.num

                break

        else:
            symbolic.append(r)

    symbolic = tuple(s for s in symbolic if s[1] != 0)

    # rem mul_1
    symbolic = tuple(
        s[2] if opcode(s) == "mul" and len(s) == 3 and s[1] == 1 else s
        for s in symbolic
    )

    if real == 0:
        res = symbolic
    else:
        if real > 0:
            real = real % (2**256)
        res = (real,) + symbolic

    if len(res) == 0:
        return 0

    if len(res) == 1:
        return res[0]

    return ("add",) + res


def bits(exp):
    """The number of bits in exp bytes."""
    if opcode(exp) == "add":
        # a size, too small to overflow: the bits of the terms add up
        return add_op(*[bits(e) for e in exp[1:]])

    if (m := match(exp, ("mul", ":int:k", ":x"))) and m.k < 0:
        # a term subtracted: minus the bits of what's subtracted - not 8 times
        # the word of a negative number (a mask of it: 256 + Mask(253, 0, 3,
        # -x) for 32 - x, which is 256 - 8 * x only modulo 2**256)
        return minus_op(bits(m.x if m.k == -1 else mul_op(-m.k, m.x)))

    return mul_op(exp, 8)


def mul_op(*args):
    # super common
    if len(args) == 1:
        return args[0]

    if match(args, (int, int)):
        return args[0] * args[1]

    for a in args:
        assert type(a) != list

        if p := to_exp2(a):
            rest = list(args)
            rest.remove(a)

            exp = mul_op(*rest)

            assert type(exp) != list, exp
            return mask_op(exp, size=256 - p, shl=p)

    # flatten muls
    res = tuple()
    for a in args:
        assert type(a) != list
        if opcode(a) == "mul":
            res += a[1:]
        else:
            res += (a,)

    # convert (mul (add x y) z) into (add (mul x z) (mul y z))
    # bc we're trying to keep a flat ordered hierarchy

    add_list = tuple(a for a in res if opcode(a) == "add")
    if len(add_list) > 0:
        el = add_list[0]
        assert opcode(el) == "add"

        without = list(res)
        without.remove(el)

        ret = tuple(mul_op(x, *without) for x in el[1:])
        return add_op(*ret)

    # multiply real numbers, add symbolic ones to output
    real = 1
    symbolic = tuple()
    for r in res:
        assert opcode(r) != "add"

        if r == 0:
            return 0
        elif type(r) in (int, float):
            real = int(real * r)  # arithmetic, or regular?
        else:
            symbolic += (r,)

    assert len(symbolic) == 0 or symbolic[0] != "mul"  # some old bug

    if len(symbolic) == 0:
        return real
    else:
        return (
            "mul",
            real,
        ) + symbolic


def get_sign(exp, top=WORD_TOP):
    """
    1 if exp > 0, -1 if exp < 0, 0 if it's 0 - as the integer it is, for any
    value of what it's made of (see value_range, and top there) - None if
    that depends on them.
    """
    if exp == 0:
        return 0

    lo, hi = value_range(exp, top=top)
    if lo > 0:
        return 1
    if hi < 0:
        return -1
    if lo == hi == 0:
        return 0
    return None


def safe_gt_zero(exp, top=WORD_TOP):
    return safe_ge_zero(sub_op(exp, 1), top)


def safe_ge_zero(exp, top=WORD_TOP):
    try:
        return ge_zero(exp, top)
    except CannotCompare:
        return None


def to_bytes(exp):
    """
    exp bits, as (bytes, bits left over): the bits left over are None if
    it isn't known whether exp is a whole number of bytes.
    """
    if type(exp) == int:
        return (exp + 7) // 8, exp % 8

    if type(exp) == tuple and exp[:4] == ("mask_shl", 253, 0, 3):
        return exp[4], 0

    if (
        m := match(exp, ("mask_shl", ":int:size", ":int:offset", ":int:shl", ":val"))
    ) and m.offset + m.shl >= 3:
        # the lowest 3 bits are 0
        return ("mask_shl", m.size, m.offset, m.shl - 3, m.val), 0

    if opcode(exp) == "mul" and len(exp) == 3 and type(exp[1]) == int:
        if exp[1] % 8 == 0:
            return ("mul", exp[1] // 8, exp[2]), 0

    if opcode(exp) == "add":
        res = []
        for e in exp[1:]:
            if (
                opcode(e) == "mul"
                and len(e) == 3
                and type(e[1]) == int
                and opcode(e[2]) == "mask_shl"
                and e[2][:4] == ("mask_shl", 253, 0, 3)
            ):
                by, bi = ("mul", e[1], e[2][4]), 0
            else:
                by, bi = to_bytes(e)

            if bi != 0:
                return mask_op(exp, shr=3), None
            res.append(by)

        return ("add",) + tuple(res), 0

    return mask_op(exp, shr=3), None


def divisible_bytes(exp):
    # returns true if an expression can be divided by 8 (into bytes) without exceptions raised etc
    try:
        return True if to_bytes(exp)[1] == 0 else False
    except Exception:
        return False


assert to_bytes(
    (
        "add",
        ("mask_shl", 253, 0, 3, ("cd", ("add", 4, ("cd", 36)))),
        ("mul", -1, ("mask_shl", 253, 0, 3, ("add", 36, ("cd", 36)))),
    )
) == (("add", ("cd", ("add", 4, ("cd", 36))), ("mul", -1, ("add", 36, ("cd", 36)))), 0)


def ge_zero(exp, top=WORD_TOP):
    """
    True if exp >= 0, False if exp < 0 - as the integer it is, for any value
    of what it's made of (see value_range; top: MEMORY_TOP for memory
    addresses and sizes, see memory_range) - CannotCompare if that depends on
    them.

    The comparisons (le_op, lt_op, max_op...) are decided so, and nothing
    else: a sign found by trying values - 0 and 2**230 for each word, as it
    was - is of these values only (the top byte of a word is 0 in both).
    """
    if type(exp) in (int, float):
        return exp >= 0

    lo, hi = value_range(exp, top=top)
    if lo >= 0:
        return True
    if hi < 0:
        return False
    raise CannotCompare


@cached
def lt_op(left, right, top=WORD_TOP):  # left < right
    """True if left < right, False if not, CannotCompare if it depends (see ge_zero)."""
    if type(left) == int and type(right) == int:
        return left < right

    if (m := match(left, ("add", ":int:num", ":max"))) and opcode(m.max) == "max":
        terms = m.max[1:]
        left = ("max",) + tuple(add_op(t, m.num) for t in terms)

    if (m := match(right, ("add", ":int:num", ":max"))) and opcode(m.max) == "max":
        terms = m.max[1:]
        right = ("max",) + tuple(add_op(t, m.num) for t in terms)

    if opcode(left) == "max":
        # below the largest: below all of them
        results = [safe_lt_op(t, right, top) for t in left[1:]]
        if all(r is True for r in results):
            return True

        if any(r is False for r in results):
            return False
        raise CannotCompare

    if opcode(right) == "max":
        # below the largest: below one of them
        results = [safe_lt_op(left, t, top) for t in right[1:]]
        if any(r is True for r in results):
            return True
        if all(r is False for r in results):
            return False
        raise CannotCompare

    lo, hi = value_range(sub_op(right, left), top=top)
    if lo > 0:
        return True
    if hi <= 0:
        return False
    raise CannotCompare


def safe_lt_op(left, right, top=WORD_TOP):
    try:
        return lt_op(left, right, top)
    except CannotCompare:
        return None


def safe_le_op(left, right, top=WORD_TOP):
    try:
        return le_op(left, right, top)
    except CannotCompare:
        return None


def simplify_max(exp):
    if opcode(exp) != "max":
        return exp

    res = ("max",)
    for e in exp[1:]:
        if opcode(e) == "max":
            res += e[1:]
        else:
            res += (e,)

    return res


@cached
def le_op(left, right, top=WORD_TOP):  # left <= right
    """True if left <= right, False if not, CannotCompare if it depends (see ge_zero)."""
    if opcode(left) == "max":
        left = max_to_add(left)

    if opcode(right) == "max":
        right = max_to_add(right)

    if type(left) in (int, float) and type(right) in (int, float):
        return left <= right

    return ge_zero(sub_op(right, left), top)


def max_op(left, right, top=WORD_TOP):
    try:
        if le_op(left, right, top):
            return right
        else:
            return left
    except CannotCompare:
        if le_op(right, left, top):
            return left
        else:
            return right


def safe_max_op(left, right, top=WORD_TOP):
    try:
        return max_op(left, right, top)
    except CannotCompare:
        return None


def _max_op(base, what):
    # compares base with what, different from algebra's max because it can return (max, x,y,z)
    if opcode(base) != "max":
        r = safe_lt_op(what, base)
        if r is True:
            return base
        elif r is False:
            return what
        return ("max", base, what)

    res = []
    for b in base[1:]:
        cmp = safe_lt_op(what, b)

        if cmp is True:
            return base
        if cmp is False:
            res.append(what)
        if cmp is None:
            res.append(b)

    res.append(what)

    # dedupe, keeping the order (a set would order the terms by their
    # hashes, i.e. differently from one run to the next)
    res = tuple(dict.fromkeys(res))
    if len(res) > 1:
        return ("max",) + res
    return res[0]


assert _max_op(("max", 128, "unknown"), 200) == ("max", 200, "unknown")
assert _max_op(("max", 128, "unknown"), 64) == ("max", 128, "unknown")


def div_op(a, b):
    assert type(a) != list
    assert type(b) != list
    if b == 1:
        return a

    if type(a) != int and type(b) == int:
        if b < 0:
            a = mul_op(-1, a)
            b = -b

        if shift := to_exp2(b):
            return mask_op(a, size=256 - shift, offset=shift, shr=shift)

    if type(a) != int or type(b) != int:
        #        return None
        return ("div", a, b)

    else:
        return a // b


def safe_min_op(left, right, top=WORD_TOP):
    try:
        return min_op(left, right, top)
    except CannotCompare:
        return None


def min_op(left, right, top=WORD_TOP):
    try:
        if le_op(left, right, top):
            return left
        else:
            return right
    except CannotCompare:
        if le_op(right, left, top):
            return right
        else:
            return left


def or_op(*args):
    if len(args) == 1:
        return args[0]
    #    assert len(args) > 1

    res = tuple()

    for r in args:
        if r == 0:
            pass

        elif opcode(r) == "or":
            terms = r[1:]
            assert len(terms) > 1
            res += terms

        elif r not in res:
            res += (r,)

    if len(res) == 0:
        return 0

    if len(res) == 1:
        return res[0]

    assert len(res) > 1

    return ("or",) + res


def neg_mask_op(exp, size, offset):
    exp1 = mask_op(
        exp, size=sub_op(256, add_op(size, offset)), offset=add_op(offset, size)
    )
    exp2 = mask_op(exp, size=offset, offset=0)

    return or_op(exp1, exp2)


def strategy_concrete(size, offset, shl, exp_size, exp_offset, exp_shl, exp):
    """
    This is an optimised version of strategy_1, the program would
    work correctly without it, but much slower, since concrete values
    for masks are very common
    """

    outer_left = offset + size
    outer_right = offset

    inner_left = exp_offset + exp_size + exp_shl
    inner_right = exp_offset + exp_shl

    left, right = min(outer_left, inner_left), max(outer_right, inner_right)

    if inner_left <= inner_right:
        return 0
    if inner_left <= outer_right:
        return 0

    new_offset = right - exp_shl
    new_size = left - right
    new_shl = shl + exp_shl

    if new_size > 0:
        return mask_op(exp, size=new_size, offset=new_offset, shl=new_shl)
    else:
        return 0


def strategy_0(size, offset, shl, exp_size, exp_offset, exp_shl, exp):
    return 0 if exp == 0 else None


def strategy_1(size, offset, shl, exp_size, exp_offset, exp_shl, exp):
    # default one

    outer_left = add_op(offset, size)
    outer_right = offset

    inner_left = add_op(exp_offset, exp_size, exp_shl)
    inner_right = add_op(exp_offset, exp_shl)

    left, right = (
        safe_min_op(outer_left, inner_left),
        safe_max_op(outer_right, inner_right),
    )

    if safe_le_op(inner_left, inner_right) is True:
        return 0
    if safe_le_op(inner_left, outer_right) is True:
        return 0

    if None not in (left, right):
        new_offset = sub_op(right, exp_shl)
        new_size = sub_op(left, right)
        new_shl = add_op(shl, exp_shl)

        gezero = safe_ge_zero(new_size)

        if gezero is not False and new_size != 0:
            return mask_op(exp, size=new_size, offset=new_offset, shl=new_shl)

        elif gezero is False or new_size == 0:
            return 0


def strategy_2(size, offset, shl, exp_size, exp_offset, exp_shl, exp):
    # move inner left by size, apply mask, and move back

    return strategy_1(
        size,
        sub_op(offset, exp_size),
        add_op(shl, exp_size),
        exp_size,
        exp_offset,
        sub_op(exp_shl, exp_size),
        exp,
    )


def strategy_3(size, offset, shl, exp_size, exp_offset, exp_shl, exp):
    # move inner left by it's shl, apply mask, move back

    return strategy_1(
        size,
        sub_op(offset, exp_shl),
        add_op(shl, exp_shl),
        exp_size,
        exp_offset,
        0,
        exp,
    )


def strategy_final(size, offset, shl, exp_size, exp_offset, exp_shl, exp):
    return (
        "mask_shl",
        size,
        offset,
        shl,
        ("mask_shl", exp_size, exp_offset, exp_shl, exp),
    )


def proven_le(left, right):
    """left <= right for any value of what they're made of (see value_range)"""
    return value_range(sub_op(right, left))[0] >= 0


def _proven_pick(cands, first):
    """the one of cands that first(one, other) proves to be first, else None"""
    for c in cands:
        if all(c == o or first(c, o) for o in cands):
            return c
    return None


def strategy_proven(size, offset, shl, exp_size, exp_offset, exp_shl, exp):
    """
    A mask of a mask, as strategy_1 does it, but only as far as it's proven
    (see value_range), and when what it makes can be printed as it is (see
    readable_mask): else it stays a mask of a mask.
    """
    # where the bits of the inner mask end up (the ones of its word), and
    # the bits the outer one keeps
    inner_right = add_op(exp_offset, exp_shl)
    inner_left = add_op(exp_offset, exp_size, exp_shl)
    outer_right = offset
    outer_left = add_op(offset, size)

    if (
        proven_le(inner_left, inner_right)
        or proven_le(inner_left, outer_right)
        or proven_le(outer_left, inner_right)
        or proven_le(inner_left, 0)
        or proven_le(256, inner_right)
    ):
        return 0

    left = _proven_pick((outer_left, inner_left, 256), proven_le)
    right = _proven_pick((outer_right, inner_right, 0), lambda a, b: proven_le(b, a))
    final = strategy_final(size, offset, shl, exp_size, exp_offset, exp_shl, exp)
    if left is None or right is None:
        return final

    new_size = sub_op(left, right)
    new_offset = sub_op(right, exp_shl)
    if proven_le(new_size, 0):
        return 0
    if not is_word(new_size):
        return final
    if not all_concrete(new_size, new_offset) and all_concrete(
        size, offset, exp_size, exp_offset
    ):
        # a size or an offset made of a shift (uint8(x << n), 2 * (1 << n)):
        # the two masks read better
        return final

    res = mask_op(exp, size=new_size, offset=new_offset, shl=add_op(shl, exp_shl))
    if opcode(res) == "mask_shl" and not readable_mask(*res[1:4]):
        return final
    return res


def mask_mask_op(size, offset, shl, exp_size, exp_offset, exp_shl, exp):
    if all_concrete(offset, shl, exp_offset, exp_shl, exp_size, size):
        return strategy_concrete(size, offset, shl, exp_size, exp_offset, exp_shl, exp)

    if readable_mask(size, offset, shl) and readable_mask(
        exp_size, exp_offset, exp_shl
    ):
        # (a mask whose shift isn't known to be a left or a right one would
        # be read as neither)
        return strategy_proven(size, offset, shl, exp_size, exp_offset, exp_shl, exp)

    strategies = (strategy_0, strategy_1, strategy_2, strategy_3, strategy_final)

    for s in strategies:
        res = s(size, offset, shl, exp_size, exp_offset, exp_shl, exp)
        if res is not None:
            return res

    assert False


# (what a mask makes of an expression may rest on what's known of the
# variables in it, see value_range: forgotten with the rest)
mask_dict = {}
CACHES.append(mask_dict)

# the operations whose result is 0 or 1
BOOL_OPS = ("bool", "iszero", "lt", "gt", "le", "ge", "eq", "slt", "sgt", "sle", "sge")


def mask_op(exp, size=256, offset=0, shl=0, shr=0):
    if size == 0:
        return 0

    idx = size, offset, shl, shr, exp
    if idx in mask_dict:
        return mask_dict[idx]

    ret = _mask_op(exp, size, offset, shl, shr)
    mask_dict[idx] = ret
    return ret


def _mask_op(exp, size=256, offset=0, shl=0, shr=0):
    if size == 0 or exp == 0:
        return 0
    #    if (size, offset, shl, shr) == (256, 0, 0, 0):
    #        return exp

    if m := match(exp, ("div", ":num", 1)):
        exp = m.num  # should be done somewhere else, but it's 0:37 at night

    shl = sub_op(shl, shr)
    shr = 0

    if opcode(exp) in BOOL_OPS and all_concrete(size, offset, shl):
        # 0 or 1: a mask keeps it as it is, or drops it
        if offset > 0 or size <= 0:
            return 0
        if shl == 0:
            return exp

    if (
        m := match(exp, ("storage", ":stor_size", ":int:stor_offset", ":stor_idx"))
    ) and m.stor_offset < 0:
        # a field moved left (see apply_mask_to_storage): a mask of the field,
        # moved
        field = ("storage", m.stor_size, 0, m.stor_idx)
        return mask_op(
            ("mask_shl", m.stor_size, 0, -m.stor_offset, field), size, offset, shl
        )

    if m := match(exp, ("storage", ":stor_size", ":stor_offset", ":stor_idx")):
        # trimming the storage inside

        # if safe_le_op(offset, minus_op(shl)):
        #    offset = minus_op(shl)

        # for shl > 0, we are either dealing with multiplication (e.g. store * 32 - happens often)
        # or with trimming the storage and moving around (store << 96)
        # e.g. 0xfF18DBc487b4c2E3222d115952bABfDa8BA52F5F, setupToken
        # below heuristics handle all this, and deliver good results in practice
        # but may be incorrect in some unusual cases

        if type(shl) == int and (shl > 0 and shl < 8):
            pass

        elif (
            type(shl) == int
            and shl >= 8
            and size == 256
            and (new_exp := apply_mask_to_storage(exp, size - shl, offset, shl))
            is not None
        ):
            return new_exp

        elif (new_exp := apply_mask_to_storage(exp, size, offset, shl)) is not None:
            # (0 when the mask is above the bits read: Mask(56, 200, uint200(x)))
            return new_exp

    if opcode(exp) == "or":
        rest = exp[1:]
        return or_op(*[mask_op(e, size, offset, shl, shr) for e in rest])

    if (
        (m := match(exp, ("signextend", ":int:b", ":val")))
        and all_concrete(size, offset)
        and offset + size <= 8 * (m.b + 1)
    ):
        # bits that signextend doesn't change
        return mask_op(m.val, size, offset, shl)

    if opcode(exp) == "mask_shl":
        params = exp[1:]
        shl = sub_op(shl, shr)
        double_mask = mask_mask_op(size, offset, shl, *params)

        return double_mask

    if type(size) != int or size > 0:
        return ("mask_shl", size, offset, sub_op(shl, shr), exp)
    else:
        return 0


def apply_mask_to_storage(exp, size, offset, shl):
    m = match(exp, ("storage", ":stor_size", ":stor_offset", ":stor_idx"))
    assert m
    stor_size, stor_offset, stor_idx = m.stor_size, m.stor_offset, m.stor_idx

    #    shr = minus_op(shl)

    stor_offset = add_op(stor_offset, offset)
    stor_size = sub_op(stor_size, offset)
    shl = add_op(shl, offset)
    offset = 0

    if safe_le_op(size, stor_size) is True:
        stor_size = size
    elif safe_le_op(stor_size, size) is not True:
        # which one is narrower isn't known (e.g. uint8 of a field of
        # 256 - 8 * i bits): the mask has to stay
        return None

    if safe_le_op(stor_size, 0) is True:
        return 0

    res = ("storage", stor_size, stor_offset, stor_idx)

    if shl == 0:
        return res

    if type(shl) is not int:
        # (a field moved left, below, is by a number of bits known)
        return None

    if shl < 0:
        # moved right: its bits from -shl on, at 0
        new_size = add_op(stor_size, shl)
        if type(new_size) is not int:
            return None
        if new_size <= 0:
            return 0
        return ("storage", new_size, add_op(stor_offset, -shl), stor_idx)

    if (m := match(res, ("storage", size, 0, ":stor_idx"))) and offset == 0:
        # a field moved left (see _mask_op)
        return ("storage", size, -shl, m.stor_idx)


def apply_mask(val, size, offset=0, shl=0):
    assert all_concrete(val, size, offset, shl)

    # a word (-1 is 2**256 - 1, see simplify_exp) is its 256 bits, and what
    # the mask makes of it a word: the bits moved past its top are gone - x
    # << 128 of shl_op is all of x moved. (A number of more bytes, of a long
    # memory range, is as wide as it is.)
    word = -(2**256) < val < 2**256
    if word:
        val %= 2**256

    mask = mask_to_int(size, offset)
    val = val & mask

    if shl >= 256 or shl <= -256:
        # shifted out of the word entirely
        return 0

    if shl > 0:
        val = val << shl

    if shl < 0:
        val = val >> -shl

    return val % 2**256 if word else val


def shr_op(exp, off):
    """
    exp >> off: the bits [off, 256) of exp, moved down by off, the same mask
    as a division by 2**off (see Stack.simplify).

    A shift by a symbolic amount is left as ("shr", off, exp): its mask would
    have a symbolic size too, Mask(256 - off, off, exp) >> off, which reads
    worse and doesn't simplify any better. simplify_exp turns it into the mask
    once the amount is known.
    """
    if not isinstance(off, int):
        return ("shr", off, exp)

    if off >= 256:
        return 0

    return mask_op(exp, size=256 - off, offset=off, shr=off)


def shl_op(exp, off):
    """
    exp << off, off a word: a mask of exp moved left by off - when that's
    what it is, off being the integer it's made of (see value_range: a word
    such as 159 - x isn't) and the mask one that can be printed as it is
    (see readable_mask). Else it's left as it is, ("shl", off, exp).
    """
    if exp == 0:
        # (moved by anything)
        return 0

    if type(off) is int:
        return 0 if off >= 256 else mask_op(exp, shl=off)

    if is_word(off) and value_range(off)[1] < 256:
        # (one that may be 256 or more shifts it all out, but known it may
        # be a word written negative, -8: a mask would move it right)
        res = mask_op(exp, shl=off)
        if _readable(res):
            return res

    return ("shl", off, exp)


def _readable(exp):
    """whether a mask made by mask_op can be printed as it is"""
    if type(exp) is int:
        return True
    if m := match(exp, ("mask_shl", ":size", ":off", ":shl", Any)):
        return readable_mask(m.size, m.off, m.shl)
    if opcode(exp) == "or":
        return all(_readable(e) for e in exp[1:])
    if m := match(exp, ("storage", ":size", ":off", Any)):
        return type(m.size) is int and type(m.off) is int
    return False


def signextend_op(b, val):
    """
    ("signextend", b, val): the lowest 8 * (b + 1) bits of val as a signed
    number, the bits above them copies of the highest of them.
    """
    if type(b) is not int:
        return ("signextend", b, val)

    # (the word of what it was made of)
    b %= 2**256
    bits = 8 * (b + 1)
    if bits >= 256:
        return val

    if type(val) is int:
        low = val & (2**bits - 1)
        if low >> (bits - 1):
            return low | (2**256 - 2**bits)
        return low

    if m := match(val, ("signextend", ":int:c", ":inner")):
        return signextend_op(min(b, m.c), m.inner)

    # a number made of some bits of another one: only its lowest bits count,
    # and if it has less than them the highest one is 0
    if (
        (m := match(val, ("mask_shl", ":int:size", ":int:off", ":int:shl", ":inner")))
        and m.shl == -m.off
        and m.off >= 0
    ):
        if m.size < bits:
            return val
        if m.off == 0:
            return signextend_op(b, m.inner)
        if m.size > bits:
            return ("signextend", b, ("mask_shl", bits, m.off, -m.off, m.inner))

    if (m := match(val, ("storage", ":int:size", ":int:off", ":loc"))) and m.off >= 0:
        if m.size < bits:
            return val
        if m.size > bits:
            return ("signextend", b, ("storage", bits, m.off, m.loc))

    return ("signextend", b, val)


def try_add(self, other):
    if (res := _try_add(self, other)) is not None:
        return res

    return __try_add(self, other)


def __try_add(self, other):
    return _try_add(_unshift(self), _unshift(other))


def _unshift(term):
    """
    num * Mask(size, off, shl, val) is num * 2**shl * Mask(size, off, 0, val),
    and when no bit of the mask stays under 256 - shl, the mask can take all
    the bits from off: e.g. 8 * x is Mask(253, 0, 3, x), which becomes
    8 * Mask(256, 0, 0, x), and adds up with x.
    """
    if (
        m := match(
            term,
            ("mul", ":num", ("mask_shl", ":int:size", ":int:off", ":int:shl", ":val")),
        )
    ) and m.shl > 0:
        size = m.size + m.shl if m.off + m.size + m.shl >= 256 else m.size
        return ("mul", m.num * _pow2(m.shl), ("mask_shl", size, m.off, 0, m.val))

    return term


def _try_add(self, other):
    # tries to add (mul a x) (mul b y)
    # 'self' name to be refactored

    #   so proud of this /s

    if not match(self, ("mul", int, Any)) or not match(other, ("mul", int, Any)):
        return None

    if (
        (ms := match(self, ("mul", -1, ":val")))
        and (
            mo := match(
                other,
                ("mul", ":mul", ("mask_shl", ":int:other_size", 0, ":int:shl", ms.val)),
            )
        )
        and mo.other_size == 256 - mo.shl
    ):
        # mul * 2**shl * val - val
        return mul_op(mo.mul * _pow2(mo.shl) - 1, ms.val)

    #    if self, other == mul(x, exp), mul(y, exp)
    #                   => mul(x+y, exp)

    if (ms := match(self, ("mul", ":int:x", ":exp"))) and (
        mo := match(other, ("mul", ":int:y", ms.exp))
    ):
        return ("mul", ms.x + mo.y, ms.exp)

    #    if self, other == mul(x, mask_shl(256-y, y, 0, exp)),
    #                      mul(x, mask_shl(y, 0, 0, exp))
    #                   => mul(x, (mask_shl, 256, 0, 0, exp))

    if (ms := match(self, ("mul", ":x", ":mask"))) and (
        mo := match(other, ("mul", ms.x, ":mask"))
    ):
        self_mask, other_mask = ms.mask, mo.mask

        if (
            opcode(self_mask) == "mask_shl"
            and opcode(other_mask) == "mask_shl"
            and isinstance(self_mask[1], numbers.Number)
            and isinstance(self_mask[2], numbers.Number)
            and self_mask[1] + self_mask[2] == 256
            and self_mask[2] == other_mask[1]
            and other_mask[2] == 0
            and self_mask[3] == other_mask[3]
            and self_mask[4] == other_mask[4]
        ):
            return mul_op(
                self[1], mask_op(self_mask[4], size=256, offset=0, shl=self_mask[3])
            )

    #   if self, other == mul(x, mask_shl(256-y, y, 0, ADD(2**y - 1, mul(1, exp)))),
    #                     mul(-x, exp)
    #                  => mul(x, 2**y - mask_op(exp, size=y))

    """
    to be tested:

    if other ~ ('mul', :x, :exp) and \
       self ~ ('mul', -x, ('mask_shl', 256-y, int:y, 0, ('add', 2**y-1, ('mul', 1, exp)))):
           return mul_op(-x, sub_op(2**y, mask_op(x, size=y)))
    """

    if (
        opcode(self[2]) == "mask_shl"
        and opcode(other[2]) != "mask_shl"
        and self[1] == minus_op(other[1])
    ):
        x = other[2]
        for y in [3, 4, 5, 6, 7, 8, 16, 32, 64, 128]:
            m = (
                "mask_shl",
                256 - y,
                y,
                0,
                ("add", 2**y - 1, ("mul", 1, x)),
            )  # - x #== 2**y-1 - Mask(y,0,0, x)
            if self[2] == m:
                return mul_op(self[1], sub_op(2**y, mask_op(x, size=y)))

    #   if self, other == mul(-x, mask_shl(256-y, y, 0, exp),
    #                     mul(x, exp)
    #                  => mul(x, mask_op(exp, size=y))

    if (
        opcode(self[2]) == "mask_shl"
        and opcode(other[2]) != "mask_shl"
        and self[1] == minus_op(other[1])
    ):
        x = other[2]
        for y in [3, 4, 5, 6, 7, 8, 16, 32, 64, 128]:
            m = ("mask_shl", 256 - y, y, 0, x)  # - x #== 2**y-1 - Mask(y,0,0, x)
            if self[2] == m:
                return mul_op(other[1], mask_op(x, size=y))

    # other

    if m := match(self, ("mul", ":num", ("mask_shl", 256, 0, 0, ":exp"))):
        self = ("mul", m.num, m.exp)

    if m := match(other, ("mul", ":num", ("mask_shl", 256, 0, 0, ":exp"))):
        other = ("mul", m.num, m.exp)

    assert (m := match(self, ("mul", ":int:num", ":exp")))
    num1, exp1 = m.num, m.exp
    assert (m := match(other, ("mul", ":int:num", ":exp")))
    num2, exp2 = m.num, m.exp

    if exp1 == exp2:
        return mul_op(num1 + num2, exp1)

    # mask 256,0,0,x - 6,0,0,x == 250,6,0,x

    if num1 == -num2 and (m := match(exp2, ("mask_shl", ":int:size", 0, 0, exp1))):
        return ("mul", num1, ("mask_shl", 256 - m.size, m.size, 0, exp1))

    if (
        num1 == -num2
        and (ms := match(exp1, ("mask_shl", ":int:size1", 0, 0, ":x")))
        and (mo := match(exp2, ("mask_shl", ":int:size2", 0, 0, ms.x)))
        and mo.size2 < ms.size1
    ):
        # the bits [size2, size1) of x
        return ("mul", num1, ("mask_shl", ms.size1 - mo.size2, mo.size2, 0, ms.x))

    return None


assert max_to_add(("max", 480, ("add", 356, ("cd", ("add", 4, ("cd", 36)))))) == (
    "add",
    356,
    ("max", 124, ("cd", ("add", 4, ("cd", 36)))),
)
assert add_op(64, ("var", 4)) == ("add", 64, ("var", 4))

l = ("add", 128, ("cd", ("add", 4, ("cd", 36))))
r = ("add", 128, ("mask_shl", 251, 5, 0, ("add", 31, ("cd", ("add", 4, ("cd", 36))))))
# (a length below 2**64, as a memory size is: then ceil32 doesn't wrap)
set_variables({}, {("cd", ("add", 4, ("cd", 36)))})
assert le_op(l, r, MEMORY_TOP) is True
set_variables({})
assert safe_le_op(l, r, MEMORY_TOP) is None
assert safe_le_op(l, r) is None
