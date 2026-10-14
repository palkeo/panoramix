import collections
import logging

from panoramix.core import arithmetic
from panoramix.core.algebra import (
    _max_op,
    add_op,
    apply_mask,
    apply_mask_to_storage,
    bits,
    calc_max,
    div_op,
    divisible_bytes,
    flatten_adds,
    ge_zero,
    get_sign,
    le_op,
    lt_op,
    mask_op,
    max_op,
    max_to_add,
    min_op,
    minus_op,
    mul_op,
    neg_mask_op,
    or_op,
    safe_ge_zero,
    safe_le_op,
    safe_lt_op,
    safe_max_op,
    safe_min_op,
    simplify,
    simplify_max,
    sub_op,
    to_bytes,
    try_add,
)
from panoramix.core.arithmetic import is_volatile, is_zero, to_real_int
from panoramix.core.masks import get_bit, to_mask, to_neg_mask
from panoramix.core.memloc import (
    apply_mask_to_range,
    fill_mem,
    memloc_overwrite,
    range_overlaps,
    split_setmem,
    split_store,
    splits_mem,
)
from panoramix.matcher import Any, match
from panoramix.prettify import (
    explain,
    pformat_trace,
    pprint_repr,
    pprint_trace,
    pretty_repr,
)
from panoramix.simplify import ENDS_EXECUTION, simplify_trace
from panoramix.utils.helpers import (
    C,
    contains,
    find_f_list,
    find_f_set,
    find_op_list,
    opcode,
    replace_f,
    replace_vars,
    replace_f_stop,
    rewrite_trace,
    rewrite_trace_full,
    rewrite_trace_ifs,
    rewrite_trace_multiline,
    to_exp2,
    walk_trace,
)

from panoramix.postprocess import cleanup_mul_1

logger = logging.getLogger(__name__)

"""

    Rube Goldberg would be proud.

"""


def make_whiles(trace, timeout=0):
    trace = make(trace)
    explain("Loops -> whiles", trace)

    # clean up jumpdests
    trace = rewrite_trace(
        trace, lambda line: [] if opcode(line) == "jumpdest" else [line]
    )
    trace = simplify_trace(trace, timeout=timeout)

    return trace


"""

    make whiles

"""


def make(trace):
    res = []

    for idx, line in enumerate(trace):
        if m := match(line, ("if", ":cond", ":if_true", ":if_false")):
            res.append(("if", m.cond, make(m.if_true), make(m.if_false)))

        elif m := match(line, ("label", ":jd", ":vars", ...)):
            jd, vars = m.jd, m.vars
            # (a loop that can't be made fails the function: without it, what
            # follows the label would read as run once)
            before, inside, remaining, cond = to_while(
                trace[idx + 1 :],
                jd,
                begin=[v_val for _, _, v_val in vars],
                loop_vars=[v_idx for _, v_idx, _ in vars],
            )

            inside = make(inside)
            remaining = make(remaining)

            before = replace_vars(before, {v_idx: v_val for _, v_idx, v_val in vars})
            before = make(before)

            res.extend(before)
            res.append(("while", cond, inside, repr(jd), vars))
            res.extend(remaining)

            return res

        elif m := match(line, ("goto", ":jd", ":setvars")):
            res.append(("continue", repr(m.jd), m.setvars))

        else:
            res.append(line)

    return res


def get_jds(line):
    if m := match(line, ("goto", ":jd", ...)):
        return [m.jd]
    return []


def is_revert(trace):
    if len(trace) != 1:
        return False

    return opcode(trace[0]) in ("revert", "invalid")


def falls_through(trace):
    """
    Whether a path of trace gets to its end: goes on with what follows it,
    rather than to a label (goto) or out of the call.
    """
    if not trace:
        return True
    last = trace[-1]
    if opcode(last) == "if":
        return falls_through(last[2]) or falls_through(last[3])
    return opcode(last) not in ("goto", "undefined", "leave") + ENDS_EXECUTION


# lines that change nothing an expression reads (but variables, see sets)
PURE_LINES = (
    "setvar",
    "if",
    "while",
    "label",
    "goto",
    "continue",
    "jump",
    "jumpdest",
    "undefined",
    "leave",
    "log",
) + ENDS_EXECUTION


def sets(trace):
    """The variables trace sets, at any depth."""
    return set(find_f_list(trace, lambda e: [e[1]] if opcode(e) == "setvar" else []))


def writes(trace):
    """Whether trace may change what an expression reads: memory, storage..."""
    for line in trace:
        if type(line) is list:
            if writes(line):
                return True
        elif opcode(line) == "if":
            if writes(line[2]) or writes(line[3]):
                return True
        elif opcode(line) == "while":
            if writes(line[2]):
                return True
        elif type(line) is tuple and opcode(line) not in PURE_LINES:
            return True
    return False


def reads(trace, name):
    """Whether trace reads the variable name."""
    return bool(
        find_f_list(
            trace,
            lambda e: (
                [e]
                if type(e) is tuple and len(e) == 2 and e[0] == "var" and e[1] == name
                else []
            ),
        )
    )


def exit_values(out, body, after, loop_vars):
    """
    `out` is the branch of a loop's exit condition, `body` the other one, and
    `after` what both go on with: the exit and the breaks of the body merged
    (see vm.merge_branches), each setting the variables of the merge for what
    differs on the stack - what a break leaves there, a local of the body in
    solc 0.4, which keeps its previous value on the exit.

    (body, after) for out written as nothing, None if it can't be: each
    variable it sets has to be one that `after` doesn't read, or one it sets
    to a loop variable, that `after` doesn't read otherwise - which stands
    for it then, the breaks setting it instead.
    """
    renames = {}
    for line in out:
        if opcode(line) != "setvar":
            return None
        _, name, value = line
        if not reads(after, name):
            continue
        if not (
            opcode(value) == "var"
            and len(value) == 2
            and value[1] in loop_vars
            and value[1] not in renames.values()
            and not reads(after, value[1])
            and value[1] not in sets(after)
        ):
            return None
        renames[name] = value[1]

    if not renames:
        return body, after

    def rename(e):
        if type(e) is tuple and len(e) == 3 and e[0] == "setvar" and e[1] in renames:
            return ("setvar", renames[e[1]], e[2])
        return e

    def renamed(trace):
        return replace_f(
            replace_vars(trace, {k: ("var", v) for k, v in renames.items()}), rename
        )

    return renamed(body), renamed(after)


def evaluated_after(path, values):
    """
    Whether the values - of loop variables - are the same evaluated after
    path as before it: none reads a variable path sets, nor what it may
    write.
    """
    names = sets(path)
    if find_f_list(
        values,
        lambda e: (
            [e]
            if type(e) is tuple and len(e) == 2 and e[0] == "var" and e[1] in names
            else []
        ),
    ):
        return False
    return not (writes(path) and any(is_volatile(v) for v in values))


def to_while(trace, jd, path=None, begin=(), loop_vars=()):
    """
    `trace` is what follows a loop label, `jd` the label, `begin` the values
    of the loop variables at the start (`loop_vars` their names). Returns
    (before, inside, remaining, cond) so that the loop can be written as:

        before
        while cond:
            inside
        remaining

    """
    path = path or []

    def add_path(line):
        # the lines preceding the exit condition are executed again after
        # the body, before the next iteration - that goes on at the label, not
        # at another one (a continue of an outer loop leaves this one)
        if m := match(line, ("goto", jd, ":svs")):
            # (the values of the next iteration, all at once)
            path2 = replace_vars(path, {v_idx: v_val for _, v_idx, v_val in m.svs})

            return path2 + [line]
        else:
            return [line]

    def rotates(body):
        """
        Whether the lines before the exit condition can be done before the
        loop, with the values the loop starts with, and at the end of each
        iteration that goes on (add_path), with the values of the next one:
        the variables are set after them then (the while's, a continue's).
        The values have to be the same evaluated there: `prev = index + n`
        with index read in the body, read again there, is the next one. And
        a continue in the lines would be before the loop.
        """
        nexts = [
            v
            for goto in find_f_list(
                body, lambda e: [e] if match(e, ("goto", jd, Any)) else []
            )
            for _, _, v in goto[2]
        ]
        return jd not in find_f_list(path, get_jds) and evaluated_after(
            path, nexts + list(begin)
        )

    while trace:
        line, *trace = trace

        if m := match(line, ("if", ":cond", ":if_true", ":if_false")):
            cond, if_true, if_false = m.cond, m.if_true, m.if_false

            # `trace` is what comes after the if - if its branches merge
            # again (see vm.merge_branches), that's what follows on the merged
            # path. Nothing otherwise.

            # a branch that reverts is a check on the way to the exit
            # condition, kept as it is: a require would drop the data of the
            # revert (a Panic code, a message), and tell an invalid from a
            # revert no more
            if is_revert(if_true):
                path.append(("if", cond, if_true, []))
                trace = if_false + trace
                continue
            if is_revert(if_false):
                path.append(("if", is_zero(cond), if_false, []))
                trace = if_true + trace
                continue

            jds_true = find_f_list(if_true, get_jds)
            jds_false = find_f_list(if_false, get_jds)

            if (
                trace
                and (jd in jds_true or jd in jds_false)
                and jd not in find_f_list(trace, get_jds)
            ):
                # The branches merge again, but the loop doesn't go on after
                # that: what follows the if is where the loop ends up when it
                # is left - by the exit condition, a break, a return from
                # inside it... The if is the body of the loop, which it leaves
                # when it doesn't continue, and that follows it.
                if (jd in jds_true) != (jd in jds_false):
                    body, out = (
                        (if_true, if_false) if jd in jds_true else (if_false, if_true)
                    )
                    exits = exit_values(out, body, trace, loop_vars)
                    if exits is not None and rotates(body):
                        # An exit condition, and the paths of the body that
                        # get to its end are breaks (see prettify.add_breaks):
                        # all go on with what follows the if - its other
                        # branch, written as nothing (see exit_values).
                        body, trace = exits
                        if jd not in jds_true:
                            cond = is_zero(cond)
                        return path, rewrite_trace(body, add_path), trace, cond
                return [], path + [line], trace, ("bool", 1)

            if trace or (jd not in jds_true and jd not in jds_false):
                # The branches merge again and the loop goes on after that:
                # a statement of the loop body (a goto inside it is a
                # `continue`), not the exit condition, which is the last
                # thing on its path.
                path.append(line)
                continue

            if jd in jds_true and jd in jds_false:
                # the loop goes on whichever way the if goes - if it can be
                # left at all, it's from inside the branches (a return, or
                # an if with an exit of its own)
                return [], path + [line], trace, ("bool", 1)

            body, out = (if_true, if_false) if jd in jds_true else (if_false, if_true)
            if out and falls_through(body):
                # A path of the body gets to its end: it leaves the loop -
                # a break, an early return (solc's return code, shared) - and
                # goes on with what follows the if, not with the exit
                # branch. After a `while cond:`, it would.
                return [], path + [line], trace, ("bool", 1)

            if not rotates(body):
                # the loop as it is: its exit inside
                return [], path + [line], trace, ("bool", 1)

            if jd in jds_true:
                if_true = rewrite_trace(if_true, add_path)
                return path, if_true, if_false + trace, cond
            else:
                if_false = rewrite_trace(if_false, add_path)
                return path, if_false, if_true + trace, is_zero(cond)

        elif match(line, ("goto", jd, ...)):
            # the path loops back unconditionally: the exits, if any, are
            # the reverts and returns along the way - the loop as it is (the
            # lines before done again after it would skip the first
            # iteration's)
            return [], path + [line], trace, ("bool", 1)

        elif opcode(line) == "label":
            # a loop in this one, before its exit condition (Vyper tests it
            # at the end): the rest of the trace is that loop, and what
            # follows it, where this one goes on or ends
            return [], path + [line] + trace, [], ("bool", 1)

        else:
            path.append(line)

    assert False, f"no if after label?{jd}"
