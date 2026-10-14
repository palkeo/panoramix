"""
SafeMath: the arithmetic a contract checks before it uses its result.

OpenZeppelin's SafeMath, ds-math and the like (solidity < 0.8: a
library's internal functions, inlined) and solidity's own checks since
0.8 (an overflow reverts with Panic(17), a division by zero with
Panic(18)) all compile to a check - an if that reverts, or runs an
invalid opcode - before the result is used: `c = a + b; require(c >= a,
"SafeMath: addition overflow")`. Where a function's folded ast has such
checks, and the result after them, the result is a call -
`SafeMath.add(a, b)` - and the checks are gone; the contract's `def
SafeMath:` has the functions as their code is: the checks as they are (a
buggy one as it is, the compiler's own check of a division the function
makes), and how they fail - with the message (`errorMessage`, a
parameter, for a check that fails with different ones), the panic, the
invalid opcode.

The checks stay where the result isn't used where they fail: after a
line that may fail too, or change what they read, or in a branch only -
the call there would fail after that line, or not on every path.
"""

from panoramix.core.algebra import add_op, mul_op
from panoramix.folder import changes_read
from panoramix.prettify import (
    ERROR,
    PANIC,
    data_bytes,
    pprint_logic,
    prettify,
    pretty_text,
)
from panoramix.utils.helpers import C, opcode

MAX = 2**256 - 1

# the parameters of the functions of the block
A, B = ("var", "a"), ("var", "b")

# the functions of the block, in its order: their results (inc and dec,
# solidity's a + 1 and a - 1 since 0.8, have one parameter)
RESULTS = {
    "add": ("add", A, B),
    "sub": ("add", A, ("mul", -1, B)),
    "mul": ("mul", A, B),
    "div": ("div", A, B),
    "mod": ("mod", A, B),
    "inc": ("add", 1, A),
    "dec": ("add", -1, A),
}

OVERFLOW = ("panic", 0x11)
DIVISION_BY_ZERO = ("panic", 0x12)


def failure(body):
    """
    How the branch of a check fails, if that's all it does: ("error",
    message) for a revert with a message (its bytes), ("panic", code),
    ("revert",) with no data, ("invalid",).
    """
    if type(body) != list or len(body) != 1:
        return None

    line = body[0]
    if line == ("invalid",):
        return ("invalid",)

    if line == ("revert", None):
        return ("revert",)

    if opcode(line) != "revert" or len(line) != 2 or opcode(line[1]) != "data":
        return None

    data = line[1][1:]
    if len(data) == 2 and data[0] == ("bytes", 4, PANIC) and type(data[1]) == int:
        return ("panic", data[1])

    if len(data) >= 3 and data[0] == ("bytes", 4, ERROR) and data[1] == 32:
        # the string: its length, then its bytes - what follows them in
        # the last word (memory, where the compiler didn't zero it) isn't
        # part of it
        length, b = data[2], b""
        if type(length) != int or length <= 0:
            return None
        for part in data[3:]:
            if len(b) >= length:
                break
            chunk = data_bytes(part)
            if chunk is None:
                return None
            b += chunk
        if len(b) >= length and pretty_text(b[:length]):
            return ("error", b[:length])

    return None


def is_check(line):
    """A one-sided if whose branch fails, all it does."""
    return opcode(line) == "if" and len(line) == 3 and failure(line[2]) is not None


def _negated(term):
    """What a term of a sum subtracts, if it's subtracted."""
    if type(term) == int and term < 0:
        return -term

    if opcode(term) == "mul" and len(term) >= 3 and type(term[1]) == int:
        if term[1] == -1:
            return term[2] if len(term) == 3 else ("mul",) + term[2:]
        if term[1] < 0:
            return ("mul", -term[1]) + term[2:]

    return None


def _sum(terms):
    if len(terms) == 0:
        return 0
    if len(terms) == 1:
        return terms[0]
    return ("add",) + tuple(terms)


def _product(terms):
    if len(terms) == 1:
        return terms[0]
    return ("mul",) + tuple(terms)


def match_check(cond, fail, res):
    """
    The operation a check (of cond, failing as fail) is of, if res is its
    result: (op, a, b), res being op(a, b) - b None for inc and dec.
    """
    op = opcode(res)

    if op == "add" and len(res) > 2:
        terms = res[1:]
        neg = [_negated(t) for t in terms if _negated(t) is not None]

        if neg:
            a = _sum([t for t in terms if _negated(t) is None])
            b = _sum(neg)
            # a - b: b compared with a (OpenZeppelin's require(b <= a, ..)
            # - not strictly: require(a > b) is a function's own, more often
            # than not), or a - b with a (ds-math's require((z = x - y) <=
            # x, ..), solidity's since 0.8.?)
            if cond in (("gt", b, a), ("lt", a, b), ("gt", res, a), ("lt", a, res)):
                return "sub", a, b
            # solidity's a - 1 since 0.8: a isn't 0
            if (
                b == 1
                and fail == OVERFLOW
                and cond in (("iszero", a), ("eq", a, 0), ("eq", 0, a))
            ):
                return "dec", a, None
            return None

        for i, a in enumerate(terms):
            b = _sum(terms[:i] + terms[i + 1 :])
            # a + b: the sum compared with a (require(c >= a, ..), strictly
            # too, solidity's since 0.8.?)
            if cond in (("lt", res, a), ("le", res, a), ("gt", a, res), ("ge", a, res)):
                return "add", a, b
            # a compared with ~b, the most b can be added to (solidity's
            # until 0.8.?), as the simplifier has it
            for nb in _complement(b):
                if cond in (("gt", a, nb), ("lt", nb, a)):
                    return "add", a, b
            # solidity's a + 1 since 0.8: a isn't the largest word
            if (
                b == 1
                and fail == OVERFLOW
                and cond
                in (("eq", a, -1), ("eq", -1, a), ("eq", a, MAX), ("eq", MAX, a))
            ):
                return "inc", a, None
        return None

    if op == "mul" and len(res) > 2:
        # a * b: divided by a, it's b (where a isn't 0)
        terms = res[1:]
        for i, a in enumerate(terms):
            b = _product(terms[:i] + terms[i + 1 :])
            for ne in (
                ("iszero", ("eq", ("div", res, a), b)),
                ("iszero", ("eq", b, ("div", res, a))),
            ):
                if cond in (ne, ("and", a, ne), ("and", ne, a)):
                    return "mul", a, b
            # b compared with the most a can be multiplied by, where a isn't
            # 0 (solidity's until 0.8.?)
            for most in (("div", -1, a), ("div", MAX, a)):
                for gt in (("gt", b, most), ("lt", most, b)):
                    if cond in (("and", a, gt), ("and", gt, a)):
                        return "mul", a, b
        return None

    if op in ("div", "mod") and len(res) == 3:
        # a / b: b isn't 0
        a, b = res[1], res[2]
        if zero_test(cond) == b:
            return op, a, b
        return None

    return None


def _complement(b):
    """~b, as the simplifier may have it."""
    nb = add_op(-1, mul_op(-1, b))
    res = [("not", b), nb]
    if type(nb) == int:
        res.append(nb % 2**256)
    return res


def zero_test(cond):
    """What cond tests to be 0, if that's what it does."""
    if opcode(cond) == "iszero" and len(cond) == 2:
        return cond[1]
    if opcode(cond) in ("eq", "le") and len(cond) == 3 and cond[2] == 0:
        return cond[1]
    if opcode(cond) == "eq" and len(cond) == 3 and cond[1] == 0:
        return cond[2]
    return None


def divides(d, exp):
    """Whether exp divides by d somewhere (a division, a modulo)."""
    if opcode(exp) in ("div", "mod") and len(exp) == 3 and exp[2] == d:
        return True
    return type(exp) in (tuple, list) and any(divides(d, e) for e in exp)


def candidates(exp, res=None):
    """
    The results of arithmetic in exp, the outer ones first - but a * b / a,
    which only a check of a * b computes.
    """
    if res is None:
        res = []

    if opcode(exp) in ("add", "mul", "div", "mod") and not (
        opcode(exp) == "div"
        and len(exp) == 3
        and opcode(exp[1]) == "mul"
        and exp[2] in exp[1][1:]
    ):
        res.append(exp)

    if type(exp) in (tuple, list):
        for e in exp:
            if type(e) in (tuple, list):
                candidates(e, res)

    return res


def replace(exp, old, new):
    """exp with old as new, wherever it is (into the branches too)."""
    if exp == old:
        return new

    if type(exp) == tuple:
        return tuple(replace(e, old, new) for e in exp)

    if type(exp) == list:
        return [replace(e, old, new) for e in exp]

    return exp


# What runs between the checks and the use of their result, as long as it
# doesn't change what they read: what can't fail.
TRANSPARENT = ("setvar", "setmem", "set", "store", "log")


def _uses(lines, res, call):
    """
    The lines (from the one the result is first used in) with the call
    in place of the result: until one changes what it reads - in that
    one, where it's read before that (a write's value, an if's
    condition); not in a loop that changes it.
    """
    out = []
    for k, line in enumerate(lines):
        changes = changes_read(line, res)
        if changes and opcode(line) == "while":
            return out + lines[k:]
        if changes and opcode(line) == "if":
            out.append((line[0], replace(line[1], res, call)) + line[2:])
            return out + lines[k + 1 :]
        out.append(replace(line, res, call))
        if changes:
            return out + lines[k + 1 :]
    return out


def _of(checks, res):
    """
    The operation all the checks are of, if res is its result: the last one
    of it, those before it too - of it as well, followed only by the
    compiler's (a library's check, then solidity's own), or the compiler's
    check of a division a later one makes by a parameter (OpenZeppelin's
    a * b / a, solidity < 0.8) - (op, a, b), or None.
    """
    last = checks[-1]
    m = match_check(last[1], failure(last[2]), res)
    if m is None:
        return None
    op, a, b = m

    for k, check in enumerate(checks[:-1]):
        fail = failure(check[2])
        later = checks[k + 1 :]
        if match_check(check[1], fail, res) == m and all(
            failure(c[2])[0] in ("invalid", "panic") for c in later
        ):
            continue
        d = zero_test(check[1])
        if (
            d is not None
            and fail in (("invalid",), DIVISION_BY_ZERO)
            and abstract(d, res, op, a, b) in (A, B)
            and any(
                divides(abstract(d, res, op, a, b), abstract(c[1], res, op, a, b))
                for c in later
            )
        ):
            continue
        return None

    return m


def _convert(lines, i, found):
    """
    The lines after the checks from i on, their result a call where it's
    used, if they're the checks of arithmetic used after them (None if
    not): the checks one after the other, the result used in the
    condition of the next check, or in the next line that may fail - only
    what can't is between them, and changes nothing they read.
    """
    end = i
    while end < len(lines) and is_check(lines[end]):
        end += 1

    # (the most checks first: a library's, then the compiler's of it)
    for e in range(end, i, -1):
        checks = lines[i:e]

        for j in range(e, len(lines)):
            line = lines[j]
            # where it's first used: what runs at once - an if's
            # condition, not a loop's (run again, the checks aren't)
            if opcode(line) == "if":
                now = line[1] if len(line) > 1 else None
            elif opcode(line) == "while":
                now = None
            else:
                now = line

            if now is not None:
                for res in candidates(now):
                    if m := _of(checks, res):
                        return lines[e:j] + _call(lines[j:], checks, res, m, found)

            if e < end or not (type(line) == str or opcode(line) in TRANSPARENT):
                break
            if any(changes_read(line, c[1]) for c in checks):
                break

    return None


def _call(lines, checks, res, m, found):
    """The lines, from the one the result is first used in, with the call
    of the checks' function instead (see _uses)."""
    op, a, b = m
    key = (op,) + tuple(
        (abstract(c[1], res, op, a, b), _kind(failure(c[2]))) for c in checks
    )
    messages = tuple(failure(c[2])[1] for c in checks if failure(c[2])[0] == "error")
    found.append((key, messages))
    call = ("safemath", len(found) - 1, a) + ((b,) if b is not None else ())
    return _uses(lines, res, call)


def _kind(fail):
    """How a check fails, but its message."""
    return ("error",) if fail[0] == "error" else fail


def abstract(cond, res, op, a, b):
    """The condition of a check of op(a, b), as the function's: of its
    parameters."""

    def subst(exp):
        if exp == res:
            return RESULTS[op]
        if exp == a:
            return A
        if b is not None and exp == b:
            return B
        if type(exp) == tuple:
            return tuple(subst(e) for e in exp)
        return exp

    return subst(cond)


def block(lines, found):
    """The lines, their checked arithmetic calls (see _convert)."""
    lines = list(lines)
    out = []
    i = 0
    while i < len(lines):
        if is_check(lines[i]):
            rest = _convert(lines, i, found)
            if rest is not None:
                lines = lines[:i] + rest
                continue

        out.append(inside(lines[i], found))
        i += 1

    return out


def inside(line, found):
    """The line, the checked arithmetic of its branches calls."""
    if opcode(line) == "if" and len(line) >= 2:
        return (line[0], line[1]) + tuple(
            block(branch, found) if type(branch) == list else branch
            for branch in line[2:]
        )

    if opcode(line) == "while" and len(line) >= 3 and type(line[2]) == list:
        return (line[0], line[1], block(line[2], found)) + line[3:]

    return line


def rewrite(functions):
    """
    The functions' asts, their checked arithmetic calls of the functions
    of the contract's SafeMath - returned: (name, key, varying), key the
    operation and its checks, varying the positions (among its messages)
    of those it's given - in the order of the block.
    """
    found = []
    asts = [
        block(func.ast, found) if func.ast is not None else None for func in functions
    ]

    # one function for each operation and its checks, numbered in the
    # order they're found - its messages parameters where they differ
    keys, messages = [], {}
    for key, msgs in found:
        if key not in messages:
            keys.append(key)
            messages[key] = []
        messages[key].append(msgs)

    names, defs = {}, []
    for op in RESULTS:
        n = 0
        for key in keys:
            if key[0] != op:
                continue
            n += 1
            names[key] = op if n == 1 else f"{op}{n}"
            first = messages[key][0]
            varying = tuple(
                k
                for k in range(len(first))
                if any(m[k] != first[k] for m in messages[key])
            )
            defs.append((names[key], key, varying, first))

    varying_of = {d[1]: d[2] for d in defs}

    def call(exp):
        if opcode(exp) == "safemath" and type(exp[1]) == int:
            key, msgs = found[exp[1]]
            return (
                ("safemath", names[key])
                + tuple(call(e) for e in exp[2:])
                + tuple(pretty_text(msgs[k]) for k in varying_of[key])
            )
        if type(exp) == tuple:
            return tuple(call(e) for e in exp)
        if type(exp) == list:
            return [call(e) for e in exp]
        return exp

    for func, ast in zip(functions, asts):
        if ast is not None:
            func.ast = call(ast)

    return defs


def pretty_defs(defs):
    """The lines of the block's functions."""
    for name, key, varying, first in defs:
        op, checks = key[0], key[1:]
        params = ["a"] if op in ("inc", "dec") else ["a", "b"]
        params += [
            "errorMessage" if n == 0 else f"errorMessage{n + 1}"
            for n in range(len(varying))
        ]
        panics = any(kind[0] == "panic" for cond, kind in checks)
        comment = f"  {C.gray}# solidity >= 0.8{C.end}" if panics else ""
        yield f"  {C.green}def {C.end}{name}({', '.join(params)}):{comment}"

        k = 0  # (the position of the check among those with messages)
        for cond, kind in checks:
            if kind == ("error",) and k in varying:
                n = varying.index(k)
                yield "      if " + prettify(
                    cond, add_color=True, parentheses=False, rem_bool=True
                ) + ":"
                yield "          revert with " + params[len(params) - len(varying) + n]
            else:
                message = first[k] if kind == ("error",) else None
                yield from pprint_logic([("if", cond, [fail_line(kind, message)])], 6)
            if kind == ("error",):
                k += 1

        yield from pprint_logic([("return", RESULTS[op])], 6)


def fail_line(kind, message):
    """The line a check fails with: its revert (a message's bytes as the
    words of an ABI-encoded string), its invalid."""
    if kind == ("invalid",):
        return ("invalid",)
    if kind == ("revert",):
        return ("revert", None)
    if kind[0] == "panic":
        return ("revert", ("data", ("bytes", 4, PANIC), kind[1]))
    padded = message + bytes(-len(message) % 32)
    words = tuple(
        int.from_bytes(padded[k : k + 32], "big") for k in range(0, len(padded), 32)
    )
    return ("revert", ("data", ("bytes", 4, ERROR), 32, len(message)) + words)
