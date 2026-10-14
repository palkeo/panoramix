"""

    Postprocess goes through a function trace, and removes all the unnecessary memcalls.

    Assumes that the trace is a tree structure (that is - needs to be run *before* the folder)

"""
from panoramix.core.algebra import minus_op
from panoramix.core.arithmetic import is_bool, is_zero
from panoramix.utils.helpers import keep_widths, opcode
from panoramix.matcher import match


def cleanup_mul_1(trace):
    def cleanup_exp(exp):
        if type(exp) != tuple:
            return exp

        # mask_shl storage -> storage
        if (
            opcode(exp) == "mask_shl"
            and opcode(exp[4]) == "storage"
            and exp[1] == exp[4][1]
            and type(exp[2]) == int
            and exp[2] == minus_op(exp[3])
            and exp[2] == exp[4][2]
        ):
            return cleanup_exp(exp[4])

        if exp[:4] == ("mask_shl", 160, 0, 0) and exp[4] in ("caller",):
            return exp[4]

        if opcode(exp) == "bool" and type(exp[1]) == int:
            return 1 if exp[1] != 0 else 0

        # mask_shl, 200, 56, 0, "'supportsInterface(bytes4)'" -> supportsInterface(bytes4)
        if (
            opcode(exp) == "mask_shl"
            and type(exp[1]) == int
            and type(exp[2]) == int
            and type(exp[3]) == int
            and type(exp[4]) == str
            and exp[1] + exp[2] == 256
            and exp[3] == 0
            and exp[4][0] == exp[4][-1] == "'"
        ):
            s = exp[4][1:-1]
            if len(s) * 8 == exp[1]:
                return s

        if exp[:4] == ("mask_shl", 256, 0, 0):
            e = cleanup_exp(exp[4])

            if type(e) == int and e < 0x100**32:
                return e

            if opcode(e) == "sha3":
                return e

                # ^ should be more generic

        #        if opcode(exp) == 'iszero' and \
        #            opcode(exp[1]) == 'eq':
        #             return ('Neq', ) + cleanup_exp(exp[1][1:])

        if opcode(exp) == "mul" and exp[1] == 1:
            if len(exp) == 3:
                return cleanup_exp(exp[2])
            else:
                assert len(exp) > 3, exp
                return ("mul",) + tuple(cleanup_exp(x) for x in exp[2:])

        return keep_widths(exp, tuple(cleanup_exp(x) for x in exp))

    res = []

    for line in trace:
        if opcode(line) == "if":
            cond, if_true, if_false = line[1:]
            res.append(
                (
                    "if",
                    cleanup_exp(cond),
                    cleanup_mul_1(if_true),
                    cleanup_mul_1(if_false),
                )
            )

        elif opcode(line) == "while":
            cond, tr, jd, setvars = line[1], line[2], line[3], line[4]
            res.append(
                (
                    "while",
                    cleanup_exp(cond),
                    cleanup_mul_1(tr),
                    jd,
                    cleanup_exp(setvars),
                )
            )

        elif opcode(line) == "LOOP":
            tr, jd = line[1:]
            res.append(("LOOP", cleanup_mul_1(tr), jd))

        else:
            res.append(cleanup_exp(line))

    return res


def logical(cond, if_true, if_false):
    """
    The value `if cond: v = if_true else: v = if_false` sets v to, as a
    python `and` / `or` of cond (OUTPUT.md), where a branch sets it to what
    the condition decides there; None where it doesn't. Where cond holds,
    it isn't 0, where it doesn't, it's 0:

        if c: v = c (or 1, c a truth value) else: v = f     c or f
        if c: v = t else: v = 0 (or c)                      c and t
        if c: v = 0 else: v = f                             not c and f
        if c: v = t else: v = 1                             not c or t
    """
    if if_true == cond or (if_true == 1 and is_bool(cond)):
        res = ("lor", cond, if_false)
    elif if_false in (0, cond):
        res = ("land", cond, if_true)
    elif if_true == 0:
        res = ("land", is_zero(cond), if_false)
    elif if_false == 1:
        res = ("lor", is_zero(cond), if_true)
    else:
        return None

    if res[0] == "lor" and res[2] == 0:
        # c or 0: c, which is 0 where it's false
        return res[1]
    if res[0] == "land" and res[2] == 1:
        # c and 1: 1 where c isn't 0
        return res[1] if is_bool(res[1]) else ("bool", res[1])
    return res


def short_circuits(trace):
    """
    The `&&` and `||` of the code, written as python's `and` and `or`: the
    paths of a short circuit merge again after it (vm.merge_branches), each
    setting the variable of the merge - to the left operand itself where it
    decides, the number it is there (vm.decided). An if whose branches only
    set the same variable to values that make it an and / or of the
    condition (see logical) is that assignment - inner ifs first, so that
    `a && (b || c)` is one.
    """
    res = []
    for line in trace:
        if opcode(line) == "if" and len(line) == 4:
            _, cond, if_true, if_false = line
            if_true, if_false = short_circuits(if_true), short_circuits(if_false)
            line = ("if", cond, if_true, if_false)
            if (
                len(if_true) == len(if_false) == 1
                and opcode(if_true[0]) == opcode(if_false[0]) == "setvar"
                and len(if_true[0]) == len(if_false[0]) == 3
                and if_true[0][1] == if_false[0][1]
            ):
                value = logical(cond, if_true[0][2], if_false[0][2])
                if value is not None:
                    line = ("setvar", if_true[0][1], value)
        elif opcode(line) == "while":
            line = line[:2] + (short_circuits(line[2]),) + line[3:]
        res.append(line)
    return res
