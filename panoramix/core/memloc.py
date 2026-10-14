import logging
import sys
from functools import partial

from panoramix.matcher import Any, match
from panoramix.utils.helpers import (
    cached,
    contains,
    find_f_list,
    is_array,
    opcode,
    replace,
)

from panoramix.core.algebra import (
    BOUNDED_SYMBOLS,
    CannotCompare,
    add_op,
    all_concrete,
    apply_mask,
    apply_mask_to_storage,
    bits,
    calc_max,
    divisible_bytes,
    flatten_adds,
    mask_op,
    max_to_add,
    minus_op,
    mul_op,
    neg_mask_op,
    or_op,
    simplify,
    simplify_max,
    sub_op,
    to_bytes,
    try_add,
    value_range,
)
from panoramix.core import algebra
from panoramix.core.algebra import MEMORY_TOP
from panoramix.core.masks import find_mask

logger = logging.getLogger(__name__)


# The comparisons of the memory model: of addresses and sizes, whose words are
# below 2**64 - gas keeps them there (see algebra.memory_range)
ge_zero = partial(algebra.ge_zero, top=MEMORY_TOP)
get_sign = partial(algebra.get_sign, top=MEMORY_TOP)
le_op = partial(algebra.le_op, top=MEMORY_TOP)
lt_op = partial(algebra.lt_op, top=MEMORY_TOP)
max_op = partial(algebra.max_op, top=MEMORY_TOP)
min_op = partial(algebra.min_op, top=MEMORY_TOP)
safe_ge_zero = partial(algebra.safe_ge_zero, top=MEMORY_TOP)
safe_gt_zero = partial(algebra.safe_gt_zero, top=MEMORY_TOP)
safe_le_op = partial(algebra.safe_le_op, top=MEMORY_TOP)
safe_lt_op = partial(algebra.safe_lt_op, top=MEMORY_TOP)
safe_max_op = partial(algebra.safe_max_op, top=MEMORY_TOP)
safe_min_op = partial(algebra.safe_min_op, top=MEMORY_TOP)


def words(*exps):
    """
    Whether each of exps is a memory address that doesn't wrap around
    2**256 - the integer it's compared as, the word it is (see
    algebra.memory_range): `p + 160` with a p that may be large, unchecked,
    may be anywhere.
    """
    for e in exps:
        lo, hi = (e, e) if type(e) is int else algebra.memory_range(e)
        if lo < 0 or hi > algebra.WORD_TOP:
            return False
    return True


def apply_mask_to_range(memloc, size, offset):
    op, range_pos, range_len = memloc
    assert op == "range"

    size_bytes, size_bits = to_bytes(size)
    offset_bytes, offset_bits = to_bytes(offset)

    assert offset_bits == size_bits == 0, (offset_bits, size_bits)  # for now
    assert safe_le_op(add_op(size_bytes, offset_bytes), range_len) is True, (
        size_bytes,
        offset_bytes,
        range_len,
    )  # otherwise we need to learn to handle that

    range_pos = add_op(range_pos, sub_op(range_len, add_op(size_bytes, offset_bytes)))
    range_len = size_bytes  # sub_op(range_len, add_op(offset_bytes, size_bytes))

    return ("range", range_pos, range_len)


assert apply_mask_to_range(("range", 212, 32), 160, 0) == ("range", 224, 20)
assert apply_mask_to_range(("range", 212, 32), 160, 96) == ("range", 212, 20)


def cmp_to_key(mycomp):
    class K:
        def __init__(self, obj):
            self.obj = obj

        def __lt__(self, other):
            return lt_op(self.obj, other.obj)

    return K(mycomp)


def value_bits(exp):
    """How many of the lowest bits of exp may not be 0, as far as it's known."""
    if type(exp) == int and exp >= 0:
        return exp.bit_length()

    if opcode(exp) in ("bool", "iszero", "eq", "lt", "gt", "le", "ge"):
        return 1
    if opcode(exp) in ("slt", "sgt", "sle", "sge"):
        return 1

    if m := match(exp, ("mask_shl", ":int:size", ":int:off", ":int:shl", ":x")):
        top = min(m.off + m.size, value_bits(m.x))
        if top <= m.off:
            return 0
        return max(0, min(top + m.shl, 256))

    if (m := match(exp, ("storage", ":int:size", ":int:off", Any))) and m.off >= 0:
        return m.size

    if algebra.may_be_wide(exp):
        # more than a word (a long range of memory, a data): as many bits
        w = width_of(exp)
        return w if type(w) is int and w > 256 else 2**64

    return 256


def max_value_bits(exp, bounds=None):
    """
    An upper bound of the bit length of exp. Numbers well below 2**256 add
    up like integers do, without wrapping around (see the comparisons of
    sums in simplify_exp). bounds: the bit lengths known of some of what
    exp is made of (a variable that holds a size, say).
    """
    if bounds and type(exp) in (tuple, str) and exp in bounds:
        return bounds[exp]

    def bits(e):
        return max_value_bits(e, bounds)

    if type(exp) == int:
        return exp.bit_length() if 0 <= exp < 2**256 else 256

    if type(exp) == str:
        return BOUNDED_SYMBOLS.get(exp, 256)

    op = opcode(exp)

    if op == "add" and len(exp) > 1:
        terms = exp[1:]
        top = max(bits(t) for t in terms)
        return min(256, top + (len(terms) - 1).bit_length())

    if op == "mul" and len(exp) > 1:
        return min(256, sum(bits(t) for t in exp[1:]))

    if op == "div" and len(exp) == 3:
        return bits(exp[1])

    if op == "mod" and len(exp) == 3:
        return min(bits(exp[1]), bits(exp[2]))

    if op == "and" and len(exp) > 1:
        return min(bits(t) for t in exp[1:])

    if op in ("or", "xor") and len(exp) > 1:
        return max(bits(t) for t in exp[1:])

    if op == "min" and len(exp) > 1:
        return min(bits(t) for t in exp[1:])

    if m := match(exp, ("mask_shl", ":int:size", ":int:off", ":int:shl", ":x")):
        top = min(m.off + m.size, bits(m.x))
        if top <= m.off:
            return 0
        return max(0, min(top + m.shl, 256))

    return value_bits(exp)


def max_value(exp):
    """An upper bound of the value of exp, as an unsigned word."""
    top = 2**256 - 1

    if type(exp) in (int, bool):
        return int(exp) % 2**256

    op = opcode(exp)

    if op in ("bool", "iszero", "eq", "lt", "gt", "le", "ge", "slt", "sgt", "sle", "sge"):
        return 1

    if (m := match(exp, ("mod", Any, ":int:c"))) and 0 < m.c < 2**256:
        return m.c - 1

    if (m := match(exp, ("div", ":x", ":int:c"))) and 0 < m.c < 2**256:
        return max_value(m.x) // m.c

    if op == "and" and len(exp) > 1:
        return min(max_value(e) for e in exp[1:])

    if m := match(exp, ("mask_shl", ":int:size", ":int:off", ":int:shl", ":x")):
        if m.size <= 0:
            return 0
        if m.off < 0:
            return top
        # x & mask is at most x, and at most the mask (not x's bound & mask:
        # x <= 4 has x & 3 up to 3, 4 & 3 is 0) - of an x of more than a
        # word, the mask
        bits = ((1 << m.size) - 1) << m.off
        if not algebra.may_be_wide(m.x):
            bits = min(max_value(m.x), bits)
        bits = bits << m.shl if m.shl >= 0 else bits >> -m.shl
        return min(bits, top)

    if (m := match(exp, ("storage", ":int:size", ":int:off", Any))) and m.off >= 0:
        return (1 << m.size) - 1 if m.size < 256 else top

    return top


def split_or(value):
    orig_value = value

    if opcode(value) not in ("or", "mask_shl"):
        return [(256, 0, value)]

    if opcode(value) == "mask_shl":
        value = ("or", value)

    opcode_, *terms = value
    assert opcode_ == "or"

    ret_rows = []

    for row in terms:
        if m := match(row, ("bool", ":arg")):
            row = (
                "mask_shl",
                8,
                0,
                0,
                ("bool", m.arg),
            )  # does weird things if size == 1, in loops.activateSafeMode

        if row == "caller":
            row = (
                "mask_shl",
                160,
                0,
                0,
                "caller",
            )  # does weird things if size == 1, in loops.activateSafeMode

        if m := match(row, ("mul", 1, ":val")):
            row = m.val

        if row == 0 or (
            opcode(row) == "mask_shl"
            and type(row[2]) == int
            and value_bits(row[4]) <= row[2]
        ):
            # nothing: bits of a value above its top (Mask(56, 200,
            # uint200(x))) are 0
            continue

        if opcode(row) == "mask_shl" and all_concrete(*row[1:]):
            row = apply_mask(row[4], row[1], row[2], row[3])

        if type(row) in [int, float]:
            size, offset = find_mask(row)
            shl = 0
            row = ("mask_shl", size, offset, 0, row)

        if m := match(row, ("mem", ":mem_idx")):
            if opcode(m.mem_idx) != "range":
                m.mem_idx = ("range", m.mem_idx, 32)

            # mem_begin = m.mem_idx[1]
            mem_len = m.mem_idx[2]
            ret_rows.append((bits(mem_len), 0, row))
            continue

        if m := match(row, ("storage", ":size", ":off", ":idx")):
            if type(m.off) == int and m.off < 0:
                # a field moved left (see algebra.apply_mask_to_storage)
                ret_rows.append((m.size, -m.off, ("storage", m.size, 0, m.idx)))
            else:
                ret_rows.append((m.size, 0, row))
            continue

        if opcode(row) != "mask_shl" and value_bits(row) < 256:
            # a truth value (x < y, iszero(x)...): its bits from the bit 0
            ret_rows.append((value_bits(row), 0, row))
            continue

        if opcode(row) != "mask_shl":
            return [(256, 0, orig_value)]

        assert opcode(row) == "mask_shl"
        _, size, offset, shl, value = row

        if all_concrete(size, offset, shl) and size + offset + shl > 256:
            # a part that goes past the top of the word: x << 128 is all of x
            # moved, as the code shifted it (see shl_op), not 256 bits of the
            # word - the word is written whole, as it's computed
            return [(256, 0, orig_value)]

        stor_size = size
        stor_offset = add_op(offset, shl)
        shl = sub_op(shl, stor_offset)
        if type(value) == int and all_concrete(size, offset, shl):
            value = apply_mask(value, size, offset, shl)

        elif (m := match(value, ("mem", ":idx"))) and add_op(offset, shl) == 0:
            try:
                new_memloc = apply_mask_to_range(m.idx, size, offset)
                value = ("mem", new_memloc)
            except AssertionError:
                # e.g. a size we can't compare with the range length (yet)
                value = mask_op(value, size=size, offset=offset, shl=shl)

        else:
            value = mask_op(value, size=size, offset=offset, shl=shl)

        ret_rows.append(
            (
                stor_size,
                stor_offset,
                value,
            )
        )

    if len(ret_rows) == 2:
        """
        a special case where rows are symbolic and complimentary. happens often
        (('mask_shl', 5, 0, 3, ('cd', ('add', 4, ('cd', 68)))), 0, ('mem', ('range', ('add', 160, ('mask_shl', 251, 5, 0, ('cd', ('add', 4, ('cd', 68)))), ('mul', -1, ('mask_shl', 5, 0, 0, ('cd', ('add', 4, ('cd', 68)))))), ('mask_shl', 5, 0, 0, ('cd', ('add', 4, ('cd', 68)))))))
        (('mask_shl', 253, 0, 3, ('add', 32, ('mul', -1, ('mask_shl', 5, 0, 0, ('cd', ('add', 4, ('cd', 68))))))), ('add', 256, ('mul', -1, ('mask_shl', 253, 0, 3, ('add', 32, ('mul', -1, ('mask_shl', 5, 0, 0, ('cd', ('add', 4, ('cd', 68))))))))), ('mem', ('range', ('add', 185, ('mask_shl', 251, 5, 0, ('add', 31, ('cd', ('add', 4, ('cd', 68))))), ('mask_shl', 251, 5, 0, ('add', 31, ('cd', ('add', 4, ('cd', 164))))), ('mask_shl', 251, 5, 0, ('cd', ('add', 4, ('cd', 68)))), ('mask_shl', 251, 0, 5, 1)), ('add', 32, ('mul', -1, ('mask_shl', 5, 0, 0, ('cd', ('add', 4, ('cd', 68)))))))))
        """
        first = ret_rows[0]
        second = ret_rows[1]
        if first[1] != 0 and second[1] == 0:
            second, first = first, second

        f_size, f_off, f_val = first
        s_size, s_off, s_val = second

        try:
            if f_off == 0 and s_off == (
                "add",
                256,
                (
                    "mul",
                    -1,
                    (
                        "mask_shl",
                        253,
                        0,
                        3,
                        ("add", 32, ("mul", -1, ("mask_shl", 5, 0, 0, f_size[4]))),
                    ),
                ),
            ):
                assert match(
                    s_size, ("mask_shl", Any, Any, Any, ("add", 32, ("mul", -1, ...)))
                )
                return ret_rows
        except (TypeError, IndexError):
            pass

    try:
        ret_rows.sort(key=lambda row: cmp_to_key(row[1]))  # sort by offsets, descending
    except Exception:
        return [(256, 0, orig_value)]

    # insert zeroes into empty spaces

    result = []

    pos = 0

    for idx, r in enumerate(ret_rows):
        if type(r[1]) != int or type(r[0]) != int:
            return [(256, 0, orig_value)]
        if r[1] < pos:
            # the parts overlap: their bits are or-ed
            return [(256, 0, orig_value)]
        if r[1] > pos:
            result.append((r[1] - pos, pos, 0))

        size, offset, value = r
        if idx + 1 < len(ret_rows) and type(ret_rows[idx + 1][1]) == int:
            room = ret_rows[idx + 1][1] - offset
            if room < size and value_bits(value) <= room:
                # a mask wider than the value in it, up to the next part:
                # e.g. a bool shifted left, Mask(32, 0, bool) << 224, next
                # to what's above its byte
                size = room

        result.append((size, offset, value))
        pos = offset + size

    if pos < 256:
        result.append((256 - pos, pos, 0))

    return result


"""

    The width of what's in memory.

    A value written to memory is as wide as the memory it's written to - the
    32 bytes of an mstore, the 1 byte of an mstore8... - and as a part of a
    ("data", ...), what's read back is as wide as sizeof says: the top bit of
    a mask, 256 bits for a word. They differ for a narrow value written to a
    word (an address, say: sizeof is 160 bits), a number that isn't a word
    (the 4 bytes of a selector), zeroes...

    ("bytes", size, exp) is exp, as `size` bytes: the lowest ones of its
    value. The memory model gives it to what it reads back when the width
    of the value isn't the width of the memory it's in. As a number, it's
    the value of exp.

"""


def sizeof(exp):  # returns size of expression in *bits*
    if m := match(exp, ("bytes", ":size", Any)):
        return bits(m.size)

    if opcode(exp) == "data":
        return add_op(*[sizeof(e) for e in exp[1:]]) if len(exp) > 1 else 0

    if m := match(exp, ("storage", ":size", ...)):
        return m.size

    if m := match(exp, ("mask_shl", ":size", ":off", ":shl", Any)):
        return add_op(m.size, m.off, m.shl)

    if (m := match(exp, (":op", Any, ":size_bytes"))) and is_array(m.op):
        return bits(m.size_bytes)

    if m := match(exp, ("mem", ("range", Any, ":size_bytes"))):
        return bits(m.size_bytes)

    if m := match(exp, ("extcodecopy", Any, ("range", Any, ":size_bytes"))):
        return bits(m.size_bytes)

    assert not match(exp, ("mem", ":idx"))
    assert not match(exp, ("arr", ":l", Any))

    if type(exp) == int and exp > 2**256:
        return bits(
            ((exp).bit_length() + 7) // 8
        )  # number of bytes needed to contain the number, rounded up

    return 256
    return None


def low_zero_bits(exp):
    """How many of the lowest bits of exp are sure to be 0."""
    if type(exp) is int:
        exp %= 2**256
        return 256 if exp == 0 else (exp & -exp).bit_length() - 1
    if m := match(exp, ("mask_shl", Any, ":int:off", ":int:shl", Any)):
        return max(0, min(256, m.off + m.shl))
    if opcode(exp) == "mul":
        return min(256, sum(low_zero_bits(e) for e in exp[1:]))
    if opcode(exp) == "add":
        return min(low_zero_bits(e) for e in exp[1:])
    return 0


def byte_elements(exp):
    """
    The positions in exp of its elements of bytes: of a data, a sha3, an
    array, and the data (or the one value) of a return, a revert, a log, the
    params of a call... As many bytes as they're written (see sizeof), not
    as they're worth: address(x) there is 20 bytes, x a word, whatever x is.
    Not a setmem's value - it's as wide as the range it's written to.
    """
    op = opcode(exp)
    if op in ("sha3", "data"):
        return range(1, len(exp))
    if op == "arr":
        return range(2, len(exp))
    positions = BYTES_OPERANDS.get(op, ())
    return tuple(i for i in positions if i < len(exp) and exp[i] is not None)


# the positions of the operands that are bytes, of the other operations
BYTES_OPERANDS = {
    "return": (1,),
    "revert": (1,),
    "log": (1,),
    # the selector and the params
    "call": (4, 5),
    "staticcall": (4, 5),
    "callcode": (4, 5),
    "delegatecall": (3, 4),
    # the code
    "create": (2,),
    "create2": (2,),
    "precompiled": (3,),
}


def sized(exp):
    """
    Whether exp, as an element of bytes, says how many bytes it is - a data,
    a Bytes(n, v), an ABI array, a range of memory or of calldata, of the
    code of an account... - rather than by how it's written (see sizeof).
    """
    op = opcode(exp)
    return op in ("bytes", "data", "arr", "mem", "sall", "extcodecopy") or is_array(op)


def width_of(exp):
    """
    The bits exp is as an element of bytes (see sizeof), None if unknown -
    an ABI-encoded array in it, whose offset goes before the rest.
    """
    if opcode(exp) == "arr" or (
        opcode(exp) == "data" and any(width_of(e) is None for e in exp[1:])
    ):
        return None
    try:
        return sizeof(exp)
    except AssertionError:
        return None


def implicit(exp, width):
    """
    Whether exp, as `width` bits of bytes, needs no ("bytes", ...) to say how
    wide it is: it's a word, or it says how many bytes it is (a range...).
    Otherwise its width goes with it: a rewrite of a value keeps what it's
    worth, not how it's written - address(x) is 20 bytes, x once it's known
    to be an address a word.
    """
    if sized(exp):
        w = width_of(exp)
        return w is None or sub_op(w, width) == 0
    return width == 256 and width_of(exp) == 256


def keep_width(old, new):
    """
    new, that is worth what old is, where old is an element of bytes: as
    wide as old is. What says how many bytes it is (a range, a "bytes"...)
    is as it's rewritten; a number that isn't a word says it.
    """
    if new == old or sized(new):
        return new
    width = width_of(old)
    if width is None or (width == 256 and width_of(new) == 256):
        return new
    if type(width) == int and width > 0 and width % 8 == 0:
        # the value, as that many bytes
        return ("bytes", width // 8, new)
    return old


def keep_widths(old, new):
    """
    new, the operation old is with its operands rewritten, with the ones
    that are bytes as wide as they were - and the value of a write to memory
    as wide as the range it's written to.
    """
    if (
        type(old) is not tuple
        or type(new) is not tuple
        or new == old
        or len(old) != len(new)
        or opcode(old) != opcode(new)
    ):
        return new
    res = list(new)
    for i in byte_elements(new):
        if res[i] != old[i]:
            res[i] = keep_width(old[i], res[i])
    if opcode(new) == "setmem" and len(new) == 3 and opcode(new[1]) == "range":
        res[2] = keep_setmem_width(new[1][2], old[2], res[2])
    return tuple(res)


def keep_setmem_width(length, old, new):
    """
    new, that is worth what old is, written to `length` bytes of memory: a
    number takes that many bytes; what says how many bytes it is must be as
    many - or, bytes of another width, the number they make: fewer of them
    as that many bytes (see with_width), more their last bytes.
    """
    if new == old or not sized(new):
        return new
    width = width_of(new)
    if type(width) is not int or type(length) is not int or width == 8 * length:
        # (as many, or not known: as it's rewritten)
        return new
    if width > 8 * length and (res := resize_bytes(new, length)) is not None:
        return res
    if 0 < length <= 32 and 0 < width <= 256 and opcode(new) != "data":
        return ("bytes", length, new)
    return old


def resize_bytes(exp, size):
    """
    exp, bytes of a known width, as the number they make written to `size`
    bytes - Bytes(size, exp) - without a "bytes" of bytes of another width:
    zeroes before them, or their last `size` bytes. None if they can't be
    cut there.
    """
    width = width_of(exp)
    if type(width) is not int or type(size) is not int or width % 8:
        return None
    width //= 8
    if width == size:
        return exp
    if width < size:
        parts = exp[1:] if opcode(exp) == "data" else (exp,)
        return ("data", ("bytes", size - width, 0)) + parts
    res = slice_exp(exp, width - size, width)
    if res is None or (opcode(res) == "bytes" and sized(res[2])):
        return None
    return res


def with_width(exp, size):
    """
    exp as the `size` bytes of memory it's in (see "bytes"): the number it
    is, as that many bytes - of bytes of another width, the number they
    make (see resize_bytes for them as bytes).
    """
    if opcode(exp) == "bytes":
        exp = exp[2]

    if opcode(exp) == "mask_shl" and all_concrete(*exp[1:]):
        # a number, and as such a word
        exp = apply_mask(exp[4], exp[1], exp[2], exp[3])

    if implicit(exp, bits(size)):
        return exp

    return ("bytes", size, exp)


def split_setmem(line):
    if opcode(line) != "setmem":
        return [line]

    _, mem_idx, mem_val = line

    if opcode(mem_val) != "or":
        return [line]

    post_split = split_or(mem_val)

    res = []
    for size, offset, split_val in post_split:
        if not (divisible_bytes(size) and divisible_bytes(offset)):
            # parts of bytes: memory is written byte by byte
            return [line]
        try:
            split_idx = apply_mask_to_range(mem_idx, size, offset)
        except Exception:
            logger.exception("problem with split_setmem")
            return [line]
        res.append(("setmem", split_idx, split_val))

    # (written all at once: see in_order)
    ordered = in_order(res, setmem_reads_written)
    return [line] if ordered is None else ordered


def in_order(writes, reads_written):
    """
    The parts a write of a whole word was split in - all written at once,
    with values of what was there before - one after the other, in an order
    where none reads what one before it wrote: as they are, unless one
    reads what one before it writes (then that one goes first). None when
    there's no such order: two that read what the other writes, a swap.

    reads_written(a, b): whether the value a writes may read what b writes.
    """
    n = len(writes)
    after = [
        [j for j in range(n) if j != i and reads_written(writes[i], writes[j])]
        for i in range(n)
    ]
    waiting = [0] * n
    for i in range(n):
        for j in after[i]:
            waiting[j] += 1
    res, done = [], [False] * n
    while len(res) < n:
        ready = [i for i in range(n) if not done[i] and not waiting[i]]
        if not ready:
            return None
        i = ready[0]
        done[i] = True
        res.append(writes[i])
        for j in after[i]:
            waiting[j] -= 1
    return res


def setmem_reads_written(a, b):
    """whether the value of the setmem a may read memory the setmem b writes"""
    if contains(a[2], "msize"):
        return True
    for m in find_f_list(a[2], lambda e: [e] if opcode(e) == "mem" else []):
        if opcode(m[1]) != "range" or range_overlaps(m[1], b[1]) is not False:
            return True
    return False


def store_reads_written(a, b):
    """
    whether the value of the store a may read bits the store b writes: of
    its slot, or of one that may be it
    """
    _, size, off, idx, _ = b
    reads = find_f_list(
        a[4], lambda e: [e] if opcode(e) == "storage" and len(e) == 4 else []
    )
    for _, r_size, r_off, r_idx in reads:
        diff = sub_op(r_idx, idx)
        if type(diff) is int and diff % 2**256 != 0:
            # another slot
            continue
        if not all_concrete(r_size, r_off, size, off):
            return True
        if r_off < 0:
            # a field moved left (see algebra.apply_mask_to_storage): of the
            # bits from 0
            r_off = 0
        if r_off < off + size and off < r_off + r_size:
            return True
    return False


def byte_field_store(line):
    """
    The store of a field at a byte computed at runtime - an element of a
    packed array - as the compiler writes it, the whole slot:

        stor[idx] = v * 256^k or not(mask * 256^k) and stor[idx]

    is the field of the bits of mask at the byte k set to v:
    store(size, 8 * k, idx, v). None if the line isn't that.
    """
    m = match(line, ("store", 256, 0, ":idx", ("or", ":a", ":b")))
    if not m:
        return None

    def times_byte(exp):
        # (x, k) if exp is x * 256^k
        if opcode(exp) != "mul":
            return None
        powers = [f for f in exp[1:] if match(f, ("exp", 256, Any))]
        others = [f for f in exp[1:] if f not in powers and f != 1]
        if len(powers) != 1 or len(others) != 1:
            return None
        return others[0], powers[0][2]

    for put, keep in ((m.a, m.b), (m.b, m.a)):
        if not (mk := match(keep, ("and", ":x", ":y"))):
            continue
        for cleared, stor in ((mk.x, mk.y), (mk.y, mk.x)):
            if stor != ("storage", 256, 0, m.idx) or opcode(cleared) != "not":
                continue
            mask = times_byte(cleared[1])
            val = times_byte(put)
            if not (mask and val) or mask[1] != val[1] or type(mask[0]) != int:
                continue
            size = mask[0].bit_length()
            if mask[0] != 2**size - 1 or value_bits(val[0]) > size:
                continue
            lo, hi = value_range(val[1])
            if lo < 0 or 8 * hi + size > 256:
                # (256^k is 0 for a k of 32 or more: then nothing changes,
                # not a field past the slot)
                continue
            return [("store", size, mul_op(8, val[1]), m.idx, val[0])]

    return None


def split_store(line):
    logger.debug("split_store %s", line)

    if (res := byte_field_store(line)) is not None:
        return res

    if (
        m := match(
            line,
            (
                "store",
                256,
                0,
                ":int:idx",
                ("mask_shl", ":int:size", ":int:off", 0, ("storage", 256, 0, ":idx")),
            ),
        )
    ) and m.size < 256:
        off, idx, size = m.off, m.idx, m.size

        lines = []
        if off > 0:
            lines.append(("store", off, 0, idx, 0))
        #        lines.append(('store', size, off, idx, ('storage', size, off, idx)))
        if size + off < 256:
            lines.append(("store", (256 - size - off), size + off, idx, 0))

        return lines

    if m := match(line, ("store", 256, 0, ":idx", ":val")):
        idx, val = m.idx, m.val
        splitted = split_or(val)

        if not all(all_concrete(s_size, s_off) for s_size, s_off, _ in splitted):
            # where the parts are isn't known: nor what the rest of the
            # word is set to
            return [line]

        splitted = sorted(
            (part for part in splitted if part[0] > 0), key=lambda part: part[1]
        )
        pos = 0
        for s_size, s_off, s_val in splitted:
            if s_off < pos or s_off + s_size > 256:
                # parts over each other (or out of the word): that's an or
                # of them, not stores of one then the other
                logger.warning("unusual store")
                return [line]
            pos = s_off + s_size

        same = [
            part for part in splitted if part[2] == ("storage", part[0], part[1], idx)
        ]
        values = [part for part in splitted if part[2] != 0 and part not in same]
        if not same and len(values) < 2:
            # one value: the word it makes, rather than it at its bits and
            # zeroes around them (but a store that keeps some bits of the
            # slot is one of the others)
            return [line]

        res = []
        # the word is written whole: the bits of no part are set to 0
        pos = 0
        for s_size, s_off, s_val in splitted:
            if s_off > pos:
                res.append(("store", s_off - pos, pos, idx, 0))
            if s_val != (
                "storage",
                s_size,
                s_off,
                idx,
            ):  # ignore writing the same to the same storage
                res.append(("store", s_size, s_off, idx, s_val))
            pos = s_off + s_size
        if pos < 256:
            res.append(("store", 256 - pos, pos, idx, 0))

        # (written all at once: see in_order)
        ordered = in_order(merge_zeros(res), store_reads_written)
        return [line] if ordered is None else ordered
    else:
        return [line]


def merge_zeros(stores):
    """
    A value that fits in its part with the 0 bits after it, up to a whole
    byte: one store of both - a bool of the bits 224-231 rather than the
    bit 224 and 0 in the 7 bits above it.
    """
    res = []
    for store in stores:
        if (
            res
            and (m := match(res[-1], ("store", ":int:size", ":int:off", ":idx", ":val")))
            and store[4] == 0
            and store[2] == m.off + m.size
            and m.size % 8
            and value_bits(m.val) <= m.size
        ):
            grow = min(store[1], 8 - m.size % 8)
            res[-1] = ("store", m.size + grow, m.off, m.idx, m.val)
            if grow < store[1]:
                res.append(("store", store[1] - grow, store[2] + grow, m.idx, 0))
            continue
        res.append(store)
    return res


def memloc_overwrite(memloc, split):
    # returns mem ranges excluding the ones that are *for sure* overwritten by 'split'
    # e.g. overwrites(('range', 64, 32), ('range', 70, 10)) -> [('range', 64, 6), (range, 80, 16)]
    # e.g. overwrites(('range', 64, 32), ('range', 70, 'unknown')) -> [('range', 64, 32)], bc. 'unknown' can be 0

    op, m_left, m_len = memloc
    assert op == "range"
    op, s_left, s_len = split
    assert op == "range"

    m_right = add_op(m_left, m_len)
    s_right = add_op(s_left, s_len)

    if not words(m_left, m_right, s_left, s_right):
        return [memloc]

    if safe_le_op(m_right, s_left) is True:  # split after memory - no overlap
        return [memloc]
    if safe_le_op(s_right, m_left) is True:  # split before memory - no overlap
        return [memloc]

    left_len = sub_op(s_left, m_left)
    right_len = sub_op(m_right, s_right)

    range_left = ("range", m_left, left_len)
    range_right = ("range", s_right, right_len)

    left_ge_zero, right_ge_zero = safe_ge_zero(left_len), safe_ge_zero(right_len)

    if left_ge_zero is None or right_ge_zero is None:
        # we can't compare some numbers, conservatively return whole range
        return [memloc]

    res = []

    if safe_ge_zero(left_len) is True and left_len != 0:
        res.append(range_left)

    if safe_ge_zero(right_len) is True and right_len != 0:
        res.append(range_right)

    return res


assert memloc_overwrite(("range", 64, 32), ("range", 70, 10)) == [
    ("range", 64, 6),
    ("range", 80, 16),
]
assert memloc_overwrite(("range", 64, 32), ("range", 70, add_op("unknown", 100))) == [
    ("range", 64, 6)
]
assert memloc_overwrite(("range", 64, "x"), ("range", 70, add_op("unknown", 100))) == [
    ("range", 64, "x")
]


def slice_exp(exp, left, right, width=None):
    """
    Bytes left to right of exp, `width` bits wide (the width of the memory
    it's in, sizeof by default).
    """
    size = sub_op(right, left)

    logger.debug("slicing %s, offset %i bytes, until %i bytes", exp, left, right)
    # e.g. mem[32 len 10], 2, 4 == mem[34,2]

    if opcode(exp) == "bytes" and sized(exp[2]):
        # bytes, as the number they make (see with_width)
        if width is None:
            width = bits(exp[1])
        exp = exp[2]

    if (
        sized(exp)
        and opcode(exp) != "bytes"
        and type(width) is int
        and (w := width_of(exp)) != width
    ):
        # bytes in memory of another width: the number they make there -
        # zeroes and them, or their last bytes (see keep_setmem_width)
        if type(w) is not int or width % 8:
            return None
        exp = resize_bytes(exp, width // 8)
        if exp is None:
            return None

    if m := match(exp, ("mem", ("range", ":rleft", ":rlen"))):
        rleft, rlen = m.rleft, m.rlen
        if safe_le_op(add_op(left, size), rlen):
            return ("mem", ("range", add_op(rleft, left), size))
        else:
            return None

    if (m := match(exp, (":op", ":rleft", ":rlen"))) and is_array(m.op):
        if safe_le_op(add_op(left, size), m.rlen):  # , (rleft, rlen, left, size, right)
            return (m.op, add_op(m.rleft, left), size)
        else:
            return None

    if m := match(exp, ("extcodecopy", ":addr", ("range", ":rleft", ":rlen"))):
        if safe_le_op(add_op(left, size), m.rlen):
            return ("extcodecopy", m.addr, ("range", add_op(m.rleft, left), size))
        return None

    if opcode(exp) == "data" and all_concrete(left, right) and left < right:
        # the parts of the data between left and right: a mask of it would
        # be a mask of a value of more than a word
        sizes = [sizeof(e) for e in exp[1:]]
        if all(type(s) == int and s % 8 == 0 for s in sizes) and (
            width is None or width == sum(sizes)
        ):
            res = []
            pos = 0
            for part, part_size in zip(exp[1:], sizes):
                part_size //= 8
                lo, hi = max(left, pos), min(right, pos + part_size)
                if lo < hi:
                    if (lo, hi) == (pos, pos + part_size):
                        piece = part
                    else:
                        piece = slice_exp(part, lo - pos, hi - pos)
                    if piece is None:
                        return None
                    res.append(piece)
                pos += part_size

            if right <= pos:
                return res[0] if len(res) == 1 else ("data",) + tuple(res)

    if opcode(exp) == "bytes":
        if width is None:
            width = bits(exp[1])
        exp = exp[2]

    if sized(exp) and opcode(exp) != "bytes":
        w = width_of(exp)
        if type(w) is not int or w > 256:
            # bytes of more than a word (or of a width not known) cut where
            # it's not known: not a number, to take bits of
            return None

    if width is None:
        width = sizeof(exp)

    off = sub_op(width, bits(right))
    logger.debug("applying mask, size 8*%s, offset %s", size, off)

    m = mask_op(exp, size=bits(size), offset=off, shr=off)
    logger.debug("result %s", m)
    return with_width(m, size)


assert slice_exp(("mem", ("range", 32, 10)), 2, 4) == ("mem", ("range", 34, 2))
# (a part that isn't a word says how many bytes it is)
assert slice_exp(("mask_shl", 32, 0, 0, ("cd", 0)), 0, 4) == (
    "bytes",
    4,
    ("mask_shl", 32, 0, 0, ("cd", 0)),
)
assert slice_exp(("mask_shl", 32, 0, 0, ("cd", 0)), 2, 4) == (
    "bytes",
    2,
    ("mask_shl", 16, 0, 0, ("cd", 0)),
)
assert slice_exp(("mask_shl", 32, 0, 0, ("cd", 0)), 0, 2) == (
    "bytes",
    2,
    ("mask_shl", 16, 16, -16, ("cd", 0)),
)


def splits_mem(memloc, split, memval, split_val=None):
    # returns memory values we can be confident of, after overwriting the split part of memory

    op, m_left, m_len = memloc
    assert op == "range"
    op, s_left, s_len = split
    assert op == "range"

    m_right = add_op(m_left, m_len)
    s_right = add_op(s_left, s_len)

    logger.debug(f"applying split [{s_left} (len {s_len}) {s_right}]")
    logger.debug(f"            to [{m_left} (len {m_len}) {m_right}]")

    if not words(m_left, m_right, s_left, s_right):
        return []

    if not safe_ge_zero(s_len):
        s_len = "undefined"
        s_right = add_op(s_left, s_len)

    if safe_le_op(m_right, s_left) is True:  # split after memory - no overlap
        return [(memloc, memval)]

    if safe_le_op(s_right, m_left) is True:  # split before memory - no overlap
        return [(memloc, memval)]

    left = safe_max_op(s_left, m_left)
    right = safe_min_op(s_right, m_right)

    logger.debug(f"split overwrites memory from {left} to {right}")

    # left/right relative to beginning of memory location
    in_left = sub_op(left, m_left)
    in_right = sub_op(right, m_left)

    logger.debug(f"that is, relative to memloc {in_left} to {in_right}")
    if safe_le_op(in_left, m_len) is not True or left is None:
        logger.debug(
            f"we are not sure that m_len: {m_len} is bigger than beginning of split, returning []"
        )
        return []

    assert in_left == 0 if safe_le_op(right, m_left) else True

    val_left = (
        slice_exp(memval, 0, in_left, width=bits(m_len)) if left is not None else None
    )
    val_right = (
        slice_exp(memval, in_right, sub_op(m_right, m_left), width=bits(m_len))
        if right is not None
        else None
    )
    res = []

    left_len = sub_op(left, m_left)  # sizeof(val_left)
    right_len = sub_op(m_right, right)

    if safe_ge_zero(left_len) is True and left_len != 0 and val_left is not None:
        res.append((("range", m_left, left_len), val_left))

    if split_val is not None:
        center_left = safe_max_op(m_left, s_left)
        center_right = safe_min_op(m_right, s_right)

        center_len = sub_op(center_right, center_left)

        if is_array(opcode(split_val)):  # in ARRAY_OPCODES:
            # mem[a len b] = calldata[x len b]
            # log mem[c len d]
            # -> calldata[x+ c - a, center_len]
            arr_offset, arr_len = split_val[1:]
            center_offset = add_op(arr_offset, sub_op(center_left, s_left))
            center_val = (opcode(split_val), center_offset, center_len)

        else:
            # its bytes there, of the split_val the split's s_len bytes are
            center_val = slice_exp(
                split_val,
                sub_op(center_left, s_left),
                sub_op(center_right, s_left),
                width=bits(s_len),
            )

        center_range = ("range", center_left, center_len)

        if safe_ge_zero(center_len) and center_len != 0 and center_val is not None:
            res.append((center_range, center_val))

    if (
        safe_ge_zero(right_len) is True
        and right_len != 0
        and val_right is not None
        # (the split ends in the memory, not before it: then all of it is
        # left, not a part from the split's end on)
        and safe_le_op(m_left, right) is True
    ):
        res.append((("range", right, right_len), val_right))

    return res


# (the bytes of a word left: as many bytes as they are)
assert splits_mem(("range", 66, 32), ("range", 65, 32), "a") == [
    (("range", 97, 1), ("bytes", 1, ("mask_shl", 8, 0, 0, "a")))
], splits_mem(("range", 66, 32), ("range", 65, 32), "a")
assert splits_mem(("range", 64, 32), ("range", 65, 32), "a") == [
    (("range", 64, 1), ("bytes", 1, ("mask_shl", 8, 248, -248, "a")))
], splits_mem(("range", 64, 32), ("range", 65, 32), "a")
assert splits_mem(("range", 4, 32), ("range", 65, 32), "a") == [(("range", 4, 32), "a")]
assert splits_mem(("range", 104, 32), ("range", 65, 32), "a") == [
    (("range", 104, 32), "a")
]
assert splits_mem(("range", 64, 32), ("range", 65, 30), "a") == [
    (("range", 64, 1), ("bytes", 1, ("mask_shl", 8, 248, -248, "a"))),
    (("range", 95, 1), ("bytes", 1, ("mask_shl", 8, 0, 0, "a"))),
]

assert (
    splits_mem(("range", 64, 32), ("range", "x", 32), "a") == []
)  # not sure means return empty
assert splits_mem(("range", 64, 32), ("range", 65, "x"), "a") == [
    (("range", 64, 1), ("bytes", 1, ("mask_shl", 8, 248, -248, "a")))
]
assert (
    splits_mem(("range", 64, "x"), ("range", 65, sub_op("x", 2)), "a") == []
), splits_mem(
    ("range", 64, "x"), ("range", 65, sub_op("x", 2)), "a"
)  # because it's either '1' if x>=1 or '0' if x == 0

assert splits_mem(("range", 64, 32), ("range", 65, 30), "a", "b") == [
    (("range", 64, 1), ("bytes", 1, ("mask_shl", 8, 248, -248, "a"))),
    (("range", 65, 30), ("bytes", 30, ("mask_shl", 240, 0, 0, "b"))),
    (("range", 95, 1), ("bytes", 1, ("mask_shl", 8, 0, 0, "a"))),
]

assert splits_mem(("range", "x", 32), ("range", "y", 32), "a", "b") == []

assert splits_mem(("range", 64, 32), ("range", 65, 32), "a", "b") == [
    (("range", 64, 1), ("bytes", 1, ("mask_shl", 8, 248, -248, "a"))),
    (("range", 65, 31), ("bytes", 31, ("mask_shl", 248, 8, -8, "b"))),
]
assert splits_mem(("range", 64, 32), ("range", 63, 32), "a", "b") == [
    (("range", 64, 31), ("bytes", 31, ("mask_shl", 248, 0, 0, "b"))),
    (("range", 95, 1), ("bytes", 1, ("mask_shl", 8, 0, 0, "a"))),
]
assert splits_mem(("range", 64, 32), ("range", 630, 32), "a", "b") == [
    (("range", 64, 32), "a")
]
assert splits_mem(("range", 64, 32), ("range", 1, 32), "a", "b") == [
    (("range", 64, 32), "a")
]

# assert splits_mem(('range', 64, 'x'), ('range', 64, 'x'), 'a', ('array', 10, 'x')) == [(('range', 64, 'x'), ('array', 10, 'x'))]

"""
#assert splits_mem(('range', 64, 32), ('add', 10, 'x')), ('range', 70, ('add', 10, 'x')), 'a', ('array', 10, ('add', 10, 'x'))) == [(('range', 64, 6), ('mask_shl', 48, ('mask_shl', 253, 0, 3, ('add', 4, 'x')), ('mul', -1, ('mask_shl', 253, 0, 3, ('add', 4, 'x'))), 'a')), (('range', 70, ('add', 4, 'x')), ('array', 10, ('add', 4, 'x')))]
"""


def splits_len(split_list):
    sum_range = 0
    for el in split_list:
        el_range = el[0]
        sum_range = add_op(sum_range, el_range[2])

    return sum_range


"""
test_s = splits_mem(('range', 64, 32), ('range', 65, 32), 'a','b')
assert splits_len(test_s) == 32

test_s = splits_mem(('range', 64, 32), ('range', 65, 30), 'a','b')
assert splits_len(test_s) == 32, test_s
"""


def replace_max_with_MAX(exp):
    if opcode(exp) != "max":
        return exp, None

    exp = max_to_add(exp)

    res = exp

    for e in exp:
        if opcode(e) == "max":
            res = e

    exp = replace(exp, res, "MAX")
    exp = simplify(exp)
    return exp, res


def fill_mem(exp, split, split_val):
    if exp == ("mem", split):
        return with_width(split_val, split[2])

    op, memloc = exp
    assert op == "mem"
    op, m_left, m_len = memloc
    assert op == "range"
    op, s_left, s_len = split
    assert op == "range"

    m_right = add_op(m_left, m_len)
    s_right = add_op(s_left, s_len)

    if safe_ge_zero(m_len) is not False and range_contains(split, memloc) is True:
        # (a read of a negative length, its end before its start, isn't in
        # anything)
        inside = slice_exp(
            split_val,
            sub_op(m_left, s_left),
            sub_op(m_right, s_left),
            width=bits(s_len),
        )
        if inside is not None:
            # all of it was written: that part of the value (maybe none of
            # it, for a length that may be 0 - the padding of an array to
            # whole words)
            return inside

    logger.debug(f"orig memloc: {m_left} len {m_len} right {m_right}")
    logger.debug(f"split memloc: {s_left} len {s_len} right {s_right}")

    if (
        not words(m_left, m_right, s_left, s_right)
        or safe_ge_zero(s_len) is not True
        or safe_ge_zero(m_len) is not True
        or safe_le_op(s_left, m_right) is not True
        or safe_le_op(m_left, s_right) is not True
    ):
        # the split may be all after the memory read, or all before it (or
        # it's not known): not replacing. Else the read is the memory before
        # the split, the split's part, and the memory after it - each of a
        # size >= 0, maybe 0 (a read of a length that may be 0)
        logger.debug("split may be before or after memory - not replacing")
        return exp

    left = safe_max_op(s_left, m_left)
    right = safe_min_op(s_right, m_right)

    logger.debug(f"split begins at {left} ends at {right}")

    if left is None or right is None:
        return exp  # if we can't figure out which one is smaller/larger, we're not replacing

    if safe_le_op(right, left) is True:
        # the read has none of it
        return exp

    memloc, memloc_max = replace_max_with_MAX(memloc)
    split, split_max = replace_max_with_MAX(split)
    # 'max' op tends to mess up with all the algebra stuff, so we're replacing
    # it with a variable 'MAX' for the time being

    if split_max != memloc_max:
        logger.warning("different maxes")
        return exp

    # by now we know:
    # - the split overlaps memory for sure
    # - we know the boundaries of split
    # - so we now return data (before_split, split_val, after_split)

    res_left = slice_exp(exp, 0, sub_op(left, m_left))
    if res_left is None:
        return exp
    logger.debug(f"value left untouched on left: {res_left}")

    res_right = slice_exp(exp, sub_op(right, m_left), sub_op(m_right, m_left))
    if res_right is None:
        return exp

    logger.debug(f"value right untouched on right: {res_right}")

    res = []

    if safe_ge_zero(sizeof(res_left)) is True:
        if sizeof(res_left) != 0:
            res.append(res_left)
    else:
        return exp

    center_in_start = sub_op(left, s_left)
    center_in_len = sub_op(right, s_left)

    logger.debug(f"inserted value offset {center_in_start}, length {center_in_len}")
    logger.debug(f"cutting this out of {split_val}")

    res_center = slice_exp(split_val, center_in_start, center_in_len, width=bits(s_len))

    logger.debug(f"inserted value after slicing: {res_center}")

    if res_center is None:
        return exp

    if safe_ge_zero(sizeof(res_center)) is True:
        res.append(res_center)
    else:
        # we can't tell that the part of the split that's read is there
        return exp

    if safe_ge_zero(sizeof(res_right)) is True:
        if sizeof(res_right) != 0:
            res.append(res_right)
    else:
        return exp

    assert None not in res

    return ("data",) + tuple(res)


@cached
def range_overlaps(range1, range2):
    op, r1_begin, r1_len = range1
    assert op == "range"
    op, r2_begin, r2_len = range2
    assert op == "range"

    r1_end = add_op(r1_begin, r1_len)
    r2_end = add_op(r2_begin, r2_len)

    if not words(r1_begin, r1_end, r2_begin, r2_end):
        return None

    try:
        if lt_op(r2_begin, r1_begin):
            r1_begin, r1_end, r2_begin, r2_end = r2_begin, r2_end, r1_begin, r1_end

        # r1 begins before r2 for sure now
        return le_op(r1_end, r2_begin) is not True

    except CannotCompare:
        return None


@cached
def range_contains(outer, inner):
    # checks if outer range *fully* contains inner range
    op, outer_begin, outer_len = outer
    assert op == "range"
    op, inner_begin, inner_len = inner
    assert op == "range"

    outer_end = add_op(outer_begin, outer_len)
    inner_end = add_op(inner_begin, inner_len)

    if not words(outer_begin, outer_end, inner_begin, inner_end):
        return None

    try:
        if not le_op(outer_begin, inner_begin):
            return False

        if not le_op(inner_end, outer_end):
            return False

        return True

    except CannotCompare:
        return None


assert (
    range_overlaps(
        (
            "range",
            ("add", 256, ("mask_shl", 246, 5, 0, ("ext_call.return_data", 128, 32))),
            32,
        ),
        ("range", 160, 96),
    )
    == False
)

assert range_overlaps(("range", 260, 32), ("range", 292, 32)) == False
assert range_overlaps(("range", 324, 32), ("range", 292, 32)) == False
assert range_overlaps(("range", 292, 32), ("range", 300, 32)) == True
assert range_overlaps(("range", 300, 32), ("range", 292, 32)) == True


assert range_contains(("range", 64, 10), ("range", 64, 32)) == False
assert range_contains(("range", 64, 32), ("range", 64, 10)) == True
assert range_contains(("range", 64, 32), ("range", 64, 32)) == True
assert range_contains(("range", 64, 32), ("range", ("var", 1), 32)) == None
assert range_contains(("range", 10, 32), ("range", 100, "x")) == False
