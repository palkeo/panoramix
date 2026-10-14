#
#  Originally taken from py-evm/eth/vm/logic/arithmetic.py
#
#  Then, comments removed to make it more in line with the rest of the codebase.
#
#  And then modified to do things it was never designed to do.
#
#  Now it's a part of the Panoramix code base.
#  It was assimilated, resistance was futile.
#
#  If you look closely, you'll still see the original peeking through
#  screaming inside: "what have you done to meeeeeee!"
#

import logging
from copy import copy

import panoramix.core.algebra as algebra
from panoramix.matcher import Any, match
from panoramix.utils.helpers import opcode

from panoramix.core.masks import get_bit

logger = logging.getLogger(__name__)

UINT_256_CEILING = 2**256
UINT_255_MAX = 2**255 - 1
UINT_256_MAX = 2**256 - 1


def to_real_int(exp):
    if type(exp) == int and get_bit(exp, 255):
        return -sub(0, exp)
    else:
        return exp


def unsigned_to_signed(value):
    if value <= UINT_255_MAX:
        return value
    else:
        return value - UINT_256_CEILING


# Symbols standing for values that can change while the contract runs, as
# opposed to e.g. calldata or block attributes. Panoramix uses the same
# expression for e.g. a storage slot before and after it's written to, or
# for the success of every external call, so two identical expressions
# mentioning one of these don't necessarily have the same value.
VOLATILE = (
    "storage",
    "balance",
    "ext_call",
    "returndatasize",
    "return_code",
    "new_address",
    "memcopy",
    ".result",
    "gas",
    "extcodesize",
    "extcodehash",
    "mem",
    "msize",
    "tload",
    "return_data",
)


def mentions(exp, names):
    if type(exp) == str:
        return any(name in exp for name in names)
    if type(exp) == tuple:
        return any(mentions(e, names) for e in exp)
    return False


def is_volatile(exp):
    return mentions(exp, VOLATILE)


# the results of the last external call (or creation, or precompile)
CALL_RESULTS = (
    "ext_call",
    "returndatasize",
    "return_code",
    "return_data",
    "new_address",
    "memcopy",
    ".result",
)


def state_read(exp):
    """
    What part of the state exp reads, if it's a read that an expression
    keeps making: "storage", "tload", "account" (a balance, a code),
    "call" (a result of the last call), None if it's none of these.
    """
    op = opcode(exp)
    if op in ("storage", "tload"):
        return op
    if op in ("balance", "extcodesize", "extcodehash"):
        return "account"
    if type(exp) is str and any(n in exp for n in CALL_RESULTS):
        return "call"
    if type(op) is str and any(n in op for n in CALL_RESULTS):
        return "call"
    return None


def may_alias(a, b):
    """False if the storage slots (or transient keys) a and b are sure to differ."""
    if a == b:
        return True

    if type(a) is int and type(b) is int:
        return False

    # a slot of a mapping or of a dynamic array (a hash) isn't a small one
    for x, y in ((a, b), (b, a)):
        if type(x) is int and x < 2**64 and type(y) is tuple:
            if opcode(y) == "sha3" or (
                opcode(y) == "add" and any(opcode(t) == "sha3" for t in y[1:])
            ):
                return False

    return True


def changed_by(exp, op, target=None):
    """
    True if exp, a read of the state, may have a different value after `op`
    (with `target` the slot or key it writes, for sstore/store and tstore).
    """
    kind = state_read(exp)
    if kind is None:
        return False
    if op in ("sstore", "store"):
        return kind == "storage" and may_alias(exp[3], target)
    if op == "tstore":
        return kind == "tload" and may_alias(exp[1], target)
    if op == "staticcall":
        return kind == "call"
    if op in ("call", "callcode", "delegatecall", "codecall", "create", "create2"):
        return True
    return False


def changed_reads(exp, op, target=None):
    """The reads of the state in exp that op may change, outermost first."""
    if changed_by(exp, op, target):
        return [exp]

    res = []
    if type(exp) is tuple:
        for e in exp:
            for r in changed_reads(e, op, target):
                if r not in res:
                    res.append(r)
    return res


def simplify_bool(exp):
    if opcode(exp) == "iszero":
        inside = simplify_bool(exp[1])

        if opcode(inside) == "iszero":
            return inside[1]
        else:
            # this had a bug and it went on unnoticed. does this check ever get executed?
            return is_zero(inside)

    if opcode(exp) == "bool":
        return exp[1]

    return exp


def and_op(*args):
    assert len(args) > 1
    if any(type(a) in (int, bool) and a == 0 for a in args):
        # (bitwise or not) 0 and anything is 0
        return 0
    left = args[0]

    if len(args) > 2:
        right = and_op(*args[1:])
    else:
        right = args[1]

    if type(left) == int and type(right) == int:
        return left & right

    res = tuple()

    if opcode(left) == "and":
        res += left[1:]
    else:
        res += (left,)

    if opcode(right) == "and":
        res += right[1:]
    else:
        res += (right,)

    return ("and",) + res


def is_bool(exp):
    """Whether exp is 0 or 1: its bits are then those of a truth value."""
    if type(exp) in (int, bool):
        return exp in (0, 1)

    op = opcode(exp)

    if op in ("bool", "iszero", "lt", "gt", "le", "ge", "eq", "slt", "sgt", "sle", "sge"):
        return True

    if op in ("and", "or", "xor"):
        return all(is_bool(e) for e in exp[1:])

    if op == "lor":
        # python's or: one of its operands
        return all(is_bool(e) for e in exp[1:])

    if op == "land":
        # python's and: its last operand, or one that is 0
        return is_bool(exp[-1])

    if m := match(exp, ("mask_shl", 1, ":int:off", ":int:shl", Any)):
        # a single bit, moved down to the lowest one
        return m.off + m.shl == 0

    if m := match(exp, ("storage", 1, ":int:off", Any)):
        return m.off >= 0

    return False


def comp_bool(left, right):
    if left == right:
        return True
    if left == ("bool", right):
        return True
    if ("bool", left) == right:
        return True
    return None


def is_zero(exp):
    if type(exp) in (int, bool):
        # (as a word)
        return exp % 2**256 == 0

    if type(exp) != tuple:
        return ("iszero", exp)

    if opcode(exp) == "iszero":
        if opcode(exp[1]) == "eq":
            return exp[1]
        elif opcode(exp[1]) == "iszero":
            return is_zero(exp[1][1])
        else:
            return ("bool", exp[1])

    if opcode(exp) == "bool":
        return is_zero(exp[1])

    if opcode(exp) == "or":
        # all of them are 0 - that of a number is a python bool (see above),
        # not something to put in an expression: a number that isn't 0
        # decides it, one that is adds nothing
        res = []
        for r in exp[1:]:
            z = is_zero(r)
            if z is False:
                return False
            if z is not True:
                res.append(z)
        if len(res) < 2:
            return res[0] if res else True
        return and_op(*res)

    if opcode(exp) == "and" and all(is_bool(r) for r in exp[1:]):
        # of truth values: one of them is false. Not of other numbers - of
        # 1 and 2, neither of which is 0, `and` is 0. (Likewise, a number
        # decides it or adds nothing.)
        res = []
        for r in exp[1:]:
            z = is_zero(r)
            if z is True:
                return True
            if z is not False:
                res.append(z)
        if not res:
            return False
        return algebra.or_op(*res)

    if opcode(exp) == "le":
        return ("gt", exp[1], exp[2])

    if opcode(exp) == "lt":
        return ("ge", exp[1], exp[2])

    if opcode(exp) == "ge":
        return ("lt", exp[1], exp[2])

    if opcode(exp) == "gt":
        return ("le", exp[1], exp[2])

    if opcode(exp) == "sle":
        return ("sgt", exp[1], exp[2])

    if opcode(exp) == "slt":
        return ("sge", exp[1], exp[2])

    if opcode(exp) == "sge":
        return ("slt", exp[1], exp[2])

    if opcode(exp) == "sgt":
        return ("sle", exp[1], exp[2])

    return ("iszero", exp)


def eval_bool(exp, known_true=True, symbolic=True):
    # ('bool', x) is true exactly when x is, so a known-true ('bool', x)
    # tells us as much as a known-true x. is_zero(('iszero', x)) yields
    # ('bool', x), so this is what the VM passes as the condition of the
    # false branch of `if iszero(x)` - e.g. the `iszero(success)` check
    # after every external call.
    if opcode(known_true) == "bool":
        known_true = known_true[1]

    if exp == known_true:
        return True

    if is_zero(exp) == known_true:
        return False

    if exp == is_zero(known_true):
        return False

    if exp is True or exp is False:
        # is_zero() of a number returns a python bool
        return exp

    if type(exp) == int:
        # a word: true when it isn't 0 (-1 is 2**256 - 1)
        return exp % 2**256 != 0

    if opcode(exp) == "bool":
        return eval_bool(exp[1], known_true=known_true, symbolic=symbolic)

    if opcode(exp) == "iszero":
        e = eval_bool(exp[1], known_true=known_true, symbolic=symbolic)
        if e is not None:
            return not e

    if opcode(exp) == "or":
        res = 0
        for e in exp[1:]:
            ev = eval_bool(e, known_true=known_true, symbolic=symbolic)
            if ev is None:
                return None
            res = res or ev
        return res

    if opcode(exp) == "and":
        # `and` can be bitwise as well, so we can only tell when an operand
        # is zero: (True and 2) is 2 & 1 == 0 for the EVM.
        for e in exp[1:]:
            if eval_bool(e, known_true=known_true, symbolic=symbolic) is False:
                return False

        #'ge', 'gt', 'eq' - tbd
    if opcode(exp) in ["le", "lt"] and opcode(exp) == opcode(known_true):
        if exp[1] == known_true[1]:
            # ('le', x, sth) while ('le', x, sth2) is known to be true: when
            # sth2 <= sth, as the words they are (x < a + 1 is no x < a + 2
            # where a + 2 wraps to 0)
            a, b = known_true[2], exp[2]
            if algebra.is_word(a) and algebra.is_word(b) and algebra.proven_le(a, b):
                return True

    if not symbolic:
        r = eval(exp)

        if type(r) == int:
            return r != 0

        return None

    if opcode(exp) == "le":
        left = eval(exp[1])
        right = eval(exp[2])

        if left == right:
            return True

        if type(left) == int and type(right) == int:
            return left % UINT_256_CEILING <= right % UINT_256_CEILING

        try:
            return algebra.le_op(left, right)
        except Exception:
            return None

    if opcode(exp) == "lt":
        left = eval(exp[1])
        right = eval(exp[2])

        if left == right:
            return False

        if type(left) == int and type(right) == int:
            return left % UINT_256_CEILING < right % UINT_256_CEILING

        try:
            return algebra.lt_op(left, right)
        except Exception:
            return None

    if opcode(exp) == "gt":
        left = eval(exp[1])
        right = eval(exp[2])

        if type(left) == int and type(right) == int:
            return left % UINT_256_CEILING > right % UINT_256_CEILING

        if left == right:
            return False

        try:  # a > b iff b < a
            return algebra.lt_op(right, left)
        except Exception:
            return None

    if opcode(exp) == "ge":
        left = eval(exp[1])
        right = eval(exp[2])

        if type(left) == int and type(right) == int:
            return left % UINT_256_CEILING >= right % UINT_256_CEILING

        if left == right:
            return True

        try:
            lt = algebra.lt_op(left, right)
            if lt == True:
                return False
            if lt == False:
                return True
            if lt is None:
                return None
        except Exception:
            pass

    if opcode(exp) == "eq":
        left = eval(exp[1])
        right = eval(exp[2])

        if left == right:
            return True

        if algebra.sub_op(left, right) == 0:
            return True

    return None


def add(*terms):
    return sum(terms) & UINT_256_MAX


def addmod(left, right, mod):
    if mod == 0:
        return 0
    else:
        return (left + right) % mod


def sub(left, right):
    if left == right:
        return 0
    else:
        return (left - right) & UINT_256_MAX


def mod(value, mod):
    if mod == 0:
        return 0
    else:
        return value % mod


def smod(value, mod):
    value, mod = map(
        unsigned_to_signed,
        (value, mod),
    )

    pos_or_neg = -1 if value < 0 else 1

    if mod == 0:
        return 0

    return (abs(value) % abs(mod) * pos_or_neg) & UINT_256_MAX


def mul(*factors):
    res = 1
    for f in factors:
        res = (res * f) & UINT_256_MAX
    return res


def mulmod(left, right, mod):
    if mod == 0:
        return 0
    return (left * right) % mod


def div(numerator, denominator):
    if numerator == 0:
        return 0
    if denominator == 0:
        return 0
    return (numerator // denominator) & UINT_256_MAX


def not_op(exp):
    return UINT_256_MAX - exp


def sdiv(numerator, denominator):
    numerator, denominator = map(
        unsigned_to_signed,
        (numerator, denominator),
    )

    pos_or_neg = -1 if numerator * denominator < 0 else 1

    if denominator == 0:
        return 0

    return pos_or_neg * (abs(numerator) // abs(denominator))


def exp(base, exponent):
    if exponent == 0:
        return 1
    elif base == 0:
        return 0
    else:
        return pow(base, exponent, UINT_256_CEILING)


def signextend(bits, value):
    if bits <= 31:
        testbit = bits * 8 + 7
        sign_bit = 1 << testbit
        if value & sign_bit:
            return value | (UINT_256_CEILING - sign_bit)
        else:
            return value & (sign_bit - 1)
    else:
        return value


def shl(shift_length, value):
    if shift_length >= 256:
        return 0
    else:
        return (value << shift_length) & UINT_256_MAX


def shr(shift_length, value):
    if shift_length >= 256:
        return 0
    else:
        return (value >> shift_length) & UINT_256_MAX


def sar(shift_length, value):
    value = unsigned_to_signed(value)

    if shift_length >= 256:
        return 0 if value >= 0 else UINT_256_MAX
    else:
        return (value >> shift_length) & UINT_256_MAX


def or_op(*terms):
    res = 0
    for t in terms:
        res |= t
    return res


def xor(*terms):
    res = 0
    for t in terms:
        res ^= t
    return res


def clz(value):
    # the leading zero bits of the word (Osaka's CLZ): 256 for 0
    return 256 - value.bit_length()


def byte_op(position, value):
    if position >= 32:
        return 0
    return (value // pow(256, 31 - position)) % 256


def lt(left, right):
    return 1 if left < right else 0


def gt(left, right):
    return 1 if left > right else 0


def le(left, right):
    return lt(left, right) | eq(left, right)


def ge(left, right):
    return gt(left, right) | eq(left, right)


def sle(left, right):
    return slt(left, right) | eq(left, right)


def sge(left, right):
    return sgt(left, right) | eq(left, right)


def slt(left, right):
    left = unsigned_to_signed(left)
    right = unsigned_to_signed(right)
    return 1 if left < right else 0


def sgt(left, right):
    left = unsigned_to_signed(left)
    right = unsigned_to_signed(right)
    return 1 if left > right else 0


def eq(left, right):
    return 1 if left == right else 0


def eval(exp):
    exp = copy(exp)

    if type(exp) != tuple:
        return exp

    for i, p in enumerate(exp[1:]):
        if opcode(p) in OPCODES:
            exp = exp[: i + 1] + (eval(p),) + exp[i + 2 :]

    for p in exp[1:]:
        if type(p) != int:
            return eval_symbolic(exp)

    if exp[0] in OPCODES:
        # of words: -1 is 2**256 - 1 (the operations, py-evm's, take them
        # from 0 up)
        return OPCODES[exp[0]](*(p % UINT_256_CEILING for p in exp[1:]))

    return exp


def eval_symbolic(exp):
    """
    Identities that hold regardless of the value of the symbolic operands.
    (in the same spirit as `mul` or `div` above returning 0 without looking
    at the other operand)
    """
    if len(exp) != 3:
        return exp

    op, left, right = exp

    if op in ("div", "sdiv", "mod", "smod") and left == 0:
        return 0

    if op == "mul" and 0 in (left, right):
        return 0

    if left == right and not is_volatile(left):
        if op in ("lt", "gt", "slt", "sgt"):
            return 0

        if op in ("le", "ge", "sle", "sge", "eq"):
            return 1

    # unsigned comparisons with zero
    if (op == "gt" and left == 0) or (op == "lt" and right == 0):
        return 0

    if (op == "le" and left == 0) or (op == "ge" and right == 0):
        return 1

    return exp


OPCODES = {
    "add": add,
    "addmod": addmod,
    "sub": sub,
    "mod": mod,
    "smod": smod,
    "mul": mul,
    "mulmod": mulmod,
    "div": div,
    "sdiv": sdiv,
    "exp": exp,
    "signextend": signextend,
    "shl": shl,
    "shr": shr,
    "sar": sar,
    "and": and_op,
    "or": or_op,
    "xor": xor,
    "not": not_op,
    "byte": byte_op,
    "clz": clz,
    "eq": eq,
    "lt": lt,
    "le": le,
    "gt": gt,
    "sgt": sgt,
    "slt": slt,
    "ge": ge,
    "gt": gt,
    "sge": sge,
    "sle": sle,
}
