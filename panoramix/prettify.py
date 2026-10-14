"""

    This module displays expressions and traces in a human readable form.

    It went through very many iterations, so it's a mess by now.

    A lot of it can be easily refactored, so if you're looking for a place to contribute,
    this may be it :)

"""

import logging
import re
import sys
from copy import deepcopy
from functools import partial

import panoramix.core.arithmetic as arithmetic
from panoramix.core.algebra import (
    add_op,
    apply_mask,
    ge_zero,
    is_word,
    lt_op,
    may_be_wide,
    minus_op,
    safe_le_op,
    shift_sign,
    sub_op,
    to_bytes,
)
from panoramix.core.arithmetic import is_bool, is_zero, simplify_bool
from panoramix.core.masks import get_bit, mask_to_type
from panoramix.core.memloc import (
    byte_elements,
    resize_bytes,
    sized,
    sizeof,
    width_of,
)
from panoramix.loader import Loader
from panoramix.matcher import Any, match
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
    all_concrete,
    clean_color,
    colorize,
    contains,
    find_f_list,
    internal_name,
    is_array,
    opcode,
    padded_hex,
    precompiled,
    replace,
    replace_f,
    replace_lines,
    to_exp2,
)
from panoramix.utils.signatures import canonical_type, fix_input_names, get_param_name
from panoramix.utils.supplement import fetch_sig

logger = logging.getLogger(__name__)

# the selectors of Panic(uint256) and Error(string)
PANIC = 0x4E487B71
ERROR = 0x08C379A0

PANIC_CODES = {
    0x00: "Used for generic compiler inserted panics.",
    0x01: "If you call assert with an argument that evaluates to false.",
    0x11: "If an arithmetic operation results in underflow or overflow outside of an unchecked { ... } block.",
    0x12: "If you divide or modulo by zero (e.g. 5 / 0 or 23 % 0).",
    0x21: "If you convert a value that is too big or negative into an enum type.",
    0x22: "If you access a storage byte array that is incorrectly encoded.",
    0x31: "If you call .pop() on an empty array.",
    0x32: "If you access an array, bytesN or an array slice at an out-of-bounds or negative index (i.e. x[i] where i >= x.length or i < 0).",
    0x41: "If you allocate too much memory or create an array that is too large.",
    0x51: "If you call a zero-initialized variable of internal function type.",
}

prev_trace = None


def explain(title, trace):
    global prev_trace

    if "--explain" not in sys.argv:
        return

    if trace == prev_trace:
        return

    print("\n" + C.green_back + f" {title}: " + C.end + "\n")
    pprint_trace(trace)
    prev_trace = trace


def explain_text(title, params):
    global prev_trace

    if "--explain" not in sys.argv:
        return

    print("\n" + C.blue_back + f" {title}: " + C.end + "\n")

    for name, val in params:
        print(f" {C.gray}{name}{C.end}: {val}")
    print()


def make_ast(trace):
    def store_to_set(line):
        if m := match(line, ("store", ":size", ":off", ":idx", ":val")):
            return ("set", ("stor", m.size, m.off, m.idx), m.val)
        else:
            return line

    def mask_storage(exp):
        if m := match(exp, ("stor", ":size", ":off", ":idx")):
            return ("mask_shl", m.size, 0, 0, exp)
        else:
            return exp

    trace = replace_lines(trace, store_to_set)
    trace = replace_f(trace, mask_storage)

    return trace


def format_exp(exp):
    if type(exp) == str:
        return f'"{exp}"'
    if type(exp) == int:
        if exp > 10**6 and exp % 10**6 != 0:
            return hex(exp)
        else:
            return str(exp)
    elif type(exp) != list:
        return str(exp)
    else:
        if len(exp) == 0:
            return COLOR_GRAY + "[]" + ENDC
        if type(opcode(exp)) == list:
            return (
                COLOR_GRAY
                + "["
                + ENDC
                + f"{COLOR_GRAY}, {ENDC}".join([format_exp(e) for e in exp])
                + COLOR_GRAY
                + "]"
                + ENDC
            )
        else:
            return (
                COLOR_GRAY
                + "["
                + ENDC
                + f"{COLOR_GRAY}, {ENDC}".join(
                    [opcode(exp)] + [format_exp(e) for e in exp[1:]]
                )
                + COLOR_GRAY
                + "]"
                + ENDC
            )


def pprint_repr(trace, indent=0):
    for line in trace:
        if opcode(line) == "if":
            cond, if_true, if_false = line[1:]
            print(indent * " ", f"[if, {format_exp(cond)}, [")
            pprint_repr(if_true, indent + 2)
            print(indent * " ", "],[")
            pprint_repr(if_false, indent + 2)
            print(indent * " ", "] ")

        elif opcode(line) == "while":
            cond, tr = line[1], line[2]
            print(indent * " ", f"[while, {format_exp(cond)}, [")
            pprint_repr(tr, indent + 2)
            print(indent * " ", "], ")

        else:
            print(indent * " ", format_exp(line) + f"{COLOR_GRAY}, {ENDC}")


"""
def pprint_repr(exp):
    print(repr(exp))
    return
    print(pretty_repr(exp))
"""


def pretty_repr(exp, indent=0):
    if type(exp) not in (tuple, list):
        return repr(exp)
    elif type(exp) == list:
        res = ", \n".join([" " * indent + pretty_repr(e, indent) for e in exp])
        res = indent * " " + "[" + res[:-3] + "]"
        return res
    elif type(exp) == tuple:
        res = ", ".join(pretty_repr(e) for e in exp)
        if len(res) > 40 and len(exp) > 1:
            op, first, *rest = exp
            indent += len(pretty_repr(op) + ", ") + 1
            res = pretty_repr(op) + ", " + pretty_repr(first, indent) + ",\n"
            for r in rest:
                res += indent * " " + pretty_repr(r, indent) + ", \n"
            res = res[:-3]  # removes ', \n'

        return "(" + res + ")"
    elif type(exp) == list:
        res = (",\n" + " " * indent).join([pretty_repr(e, indent) for e in exp])
        return f"[{res}]"


# print(pretty_repr(('data', ('mem', ('range', ('add', 32, 'QQ', ('mask_shl', 251, 5, 0, ('add', 31, ('ext_call.return_data', 128, 32)))), 32)), 'yy', ('data', ('mem', ('range', ('add', 32, 'QQ'), ('ext_call.return_data', 128, 32))), ('mem', ('range', ('add', 96, 'QQ', ('mask_shl', 251, 5, 0, ('add', 31, ('ext_call.return_data', 128, 32))), ('ext_call.return_data', 128, 32)), 0))))))
# exit()


def pformat_trace(trace):
    return "\n".join(pprint_logic(trace)) + "\n\n"


def pprint_trace(trace):
    trace = make_ast(trace)
    pprint_ast(trace)


def pprint_ast(trace):
    empty = True

    for l in pprint_logic(trace):
        print(l)
        empty = False

    if empty:
        print("  stop")
    print()
    print()


def check_word(branch):
    """
    require if the branch only reverts, assert if it's only an invalid (an
    assert of solidity < 0.8 - it uses all the gas), None otherwise.
    """
    if len(branch) != 1:
        return None
    if branch[0] == ("revert", None):
        return "require"
    if opcode(branch[0]) == "invalid":
        return "assert"
    return None


# what a path ends with, not going on after it
ENDS_PATH = (
    "return",
    "stop",
    "selfdestruct",
    "invalid",
    "revert",
    "continue",
    "break",
    "undefined",
    "leave",
)


def add_breaks(path):
    """
    The body of a loop, with a break where it ends: the loop is left there
    (see make_whiles), not gone on with - it's only by a continue that it is.
    """
    if not path:
        return [("break",)]

    last = path[-1]
    if opcode(last) in ENDS_PATH:
        return path

    if m := match(last, ("if", ":cond", ":if_true", ":if_false")):
        return path[:-1] + [
            ("if", m.cond, add_breaks(m.if_true), add_breaks(m.if_false))
        ]

    return path + [("break",)]


def breaks(path):
    """Whether path ends with a break (see add_breaks)."""
    return bool(path) and opcode(path[-1]) == "break"


def continues_from_inside(path, jd):
    """Whether a loop inside path continues the loop jd."""

    def f(exp):
        if opcode(exp) == "while":
            inner = exp[2] if type(exp[2]) == list else exp[2].trace
            return find_f_list(inner, lambda e: [e] if match(e, ("continue", jd, Any)) else [])
        return []

    return bool(find_f_list(path, f))


def pprint_logic(exp, indent=2, loops=()):
    """
    The lines of exp, indented by indent. loops: the (jd, label) of the
    loops it's in, the innermost last - a continue of another one than the
    innermost names it.
    """
    INDENT_LEN = 4

    if opcode(exp) == "while":
        if len(exp) == 5:
            cond, path, jd, vars = exp[1], exp[2], exp[3], exp[4]
        else:
            cond, path = exp[1], exp[2]
            jd, vars = None, []

        for v in sequential_setvars(vars):
            yield " " * indent + list(
                pretty_line(("setvar", v[1], v[2]), add_color=True)
            )[0]

        if type(path) != list:
            path = path.trace

        label = None
        if jd is not None and continues_from_inside(path, jd):
            label = f"loop{len(loops) + 1}"

        if cond in (1, ("bool", 1)):
            cond_text = "True"
        else:
            cond_text = prettify(cond, add_color=True, parentheses=False, rem_bool=True)
        while_line = (
            (f"{label}: " if label else "")
            + COLOR_GREEN
            + "while "
            + ENDC
            + cond_text
            + COLOR_GREEN
            + ":"
            + ENDC
        )
        yield " " * indent + while_line

        for l in pprint_logic(add_breaks(path), indent + INDENT_LEN, loops + ((jd, label),)):
            yield l

    elif m := match(exp, ("continue", ":jd", ":setvars")):
        for v in sequential_setvars(m.setvars):
            yield " " * indent + str(list(pretty_line(v, add_color=True))[0])
        target = ""
        if loops and loops[-1][0] != m.jd:
            # of a loop the continue is in a loop in
            labels = [label for jd, label in loops if jd == m.jd]
            target = " " + (labels[0] if labels and labels[0] else "?")
        yield " " * indent + COLOR_GREEN + "continue" + ENDC + target

    elif opcode(exp) == "break":
        yield " " * indent + COLOR_GREEN + "break" + ENDC

    elif opcode(exp) == "require":
        _, cond = exp
        yield " " * indent + "require " + prettify(
            exp[1], add_color=True, parentheses=False, rem_bool=True
        ) + ""

    elif m := match(
        exp, ("if", ":cond", ":if_true")
    ):  # one-sided ifs, only after folding
        cond, if_true = m.cond, m.if_true
        if word := check_word(if_true):
            yield " " * indent + word + " " + prettify(
                is_zero(exp[1]), add_color=True, parentheses=False, rem_bool=True
            )
        else:
            yield " " * indent + "if " + prettify(
                exp[1], add_color=True, parentheses=False, rem_bool=True
            ) + ":"
            for l in pprint_logic(if_true, indent + INDENT_LEN, loops):
                yield l

    elif m := match(exp, ("if", ":cond", ":if_true", ":if_false")):
        cond, if_true, if_false = m.cond, m.if_true, m.if_false
        if word := check_word(if_false):
            yield " " * indent + word + " " + prettify(
                exp[1], add_color=True, parentheses=False, rem_bool=True
            )

            for l in pprint_logic(exp[2], indent, loops):
                yield l

        elif word := check_word(if_true):
            yield " " * indent + word + " " + prettify(
                is_zero(exp[1]), add_color=True, parentheses=False, rem_bool=True
            )

            for l in pprint_logic(exp[3], indent, loops):
                yield l

        elif if_false and (breaks(if_true) or breaks(if_false) and if_true):
            # a branch that leaves the loop (see add_breaks): the other one
            # follows the if rather than being its else
            if not breaks(if_true):
                cond, if_true, if_false = is_zero(cond), if_false, if_true
            yield " " * indent + "if " + prettify(
                cond, add_color=True, parentheses=False, rem_bool=True
            ) + ":"
            for l in pprint_logic(if_true, indent + INDENT_LEN, loops):
                yield l
            for l in pprint_logic(if_false, indent, loops):
                yield l

        else:
            yield " " * indent + "if " + prettify(
                exp[1], add_color=True, parentheses=False, rem_bool=True
            ) + ":"

            for l in pprint_logic(if_true, indent + INDENT_LEN, loops):
                yield l
            """
            while len(if_false) == 1 and opcode(if_false) == 'if' and len(if_false) == 4:
                first = if_false[0]
                assert first ~ ('if', :c, :i_t, :if_false)

                yield ' '*indent + 'elif ' + prettify(c, add_color=True, parentheses=False, rem_bool=True) + ':'

                for l in pprint_logic(i_t, indent + INDENT_LEN):
                    yield l"""

            yield " " * indent + "else:"
            for l in pprint_logic(if_false, indent + INDENT_LEN, loops):
                yield l

    elif type(exp) == list:
        for idx, line in enumerate(exp):
            if idx == len(exp) - 1 and indent == 2 and line == ("stop",):
                pass  # don't print the last stop
            else:
                for l in pprint_logic(line, indent, loops):
                    yield l

    elif opcode(exp) == "or" and len(exp) > 1:
        yield " " * indent + "if"
        for l in pprint_logic(exp[1], indent + INDENT_LEN, loops):
            yield l

        for line in exp[2:]:
            yield " " * indent + "or"
            for l in pprint_logic(line, indent + INDENT_LEN, loops):
                yield l

    else:
        for l in pretty_line(exp):
            yield " " * indent + l


def to_real_int(exp):
    if type(exp) == int and get_bit(exp, 255):
        return -arithmetic.sub(0, exp)
    else:
        return exp


def unsigned(exp):
    """a number as the unsigned word it is: 2**256 - 1 rather than -1"""
    if type(exp) == int:
        return exp % 2**256
    return exp


def pretty_line(r, add_color=True):
    col = partial(colorize, add_color=add_color)
    pret = partial(prettify, parentheses=False, add_color=add_color)

    if type(r) is str:
        yield COLOR_GRAY + "# " + r + ENDC

    #    elif r ~ ('jumpdest', ...):
    #        pass

    elif m := match(r, ("comment", ":text")):
        yield COLOR_GRAY + "# " + prettify(m.text, add_color=False) + ENDC

    elif match(r, ("log", ":params", ...)):
        _, params, *topics = r

        # the params of a log are its data, then its topics after the first,
        # which is the signature of the event (unless it's anonymous) - the
        # topics are the indexed ones
        data_params = pretty_memory(params, add_color=False)
        topic_params = tuple(
            prettify(t, add_color=False, parentheses=False) for t in topics[1:]
        )
        res_params = data_params + topic_params

        abi = event_abi(topics[0]) if topics else None

        if not topics:
            # a log of no topics (see OUTPUT.md): its data, if it has any
            yield col(" ".join(("log", ", ".join(res_params))).rstrip(), COLOR_GRAY)
            return

        if abi is None:
            if type(topics[0]) == int:
                # the signature of an event this doesn't know: all of it
                e = padded_hex(topics[0], 64)
            else:
                e = prettify(topics[0], add_color=False, parentheses=False)
            listed = data_params + tuple("indexed " + t for t in topic_params)
            yield col(f"log {e}:{' ' if listed else ''}{', '.join(listed)}", COLOR_GRAY)
            return

        inputs = fix_input_names(abi["inputs"])
        fname = abi["name"]
        # (a tuple as the types it's made of: the signature is what's hashed)
        e = "{}({})".format(
            fname,
            ", ".join(
                f"{canonical_type(i['type'], i.get('components'))} {i['name']}"
                for i in inputs
            ),
        )

        # the ones not indexed are in the data, the indexed ones are topics
        not_indexed = [i for i in inputs if not i.get("indexed")]
        indexed = [i for i in inputs if i.get("indexed")]
        in_log = not_indexed + indexed

        if len(inputs) == 0 and len(res_params) == 0:
            yield col(f"log {e}", COLOR_GRAY)

        elif len(not_indexed) == len(data_params) and len(indexed) == len(topic_params):
            p_list = [
                (
                    canonical_type(i["type"], i.get("components"))
                    + (" indexed" if i.get("indexed") else ""),
                    i["name"],
                    p,
                )
                for i, p in zip(in_log, res_params)
            ]
            # in the order of the declaration (by the inputs themselves: two of
            # them may be alike)
            pos = {id(i): k for k, i in enumerate(in_log)}
            p_list = [p_list[pos[id(i)]] for i in inputs]

            if len(p_list) == 1:
                yield col(
                    f"log {fname}({p_list[0][0]} {p_list[0][1]}={p_list[0][2]})",
                    COLOR_GRAY,
                )
            else:
                ind = len(f"log   ")
                first = p_list[0]
                last = p_list[-1]

                def pline(p):
                    return f"{p[0]} {p[1]}={p[2]}"

                yield col(f"log {fname}(", COLOR_GRAY)  #
                yield col(f"      {pline(first)},", COLOR_GRAY)

                for p in p_list[1:-1]:
                    yield col(" " * ind + f"{pline(p)},", COLOR_GRAY)

                yield col(" " * ind + f"{pline(last)})", COLOR_GRAY)
        else:
            # not the params of the abi - an event of the same signature,
            # indexed otherwise (ERC-721's Approval, ERC-20's): the
            # signature, then what the log has
            listed = data_params + tuple("indexed " + t for t in topic_params)
            yield col(
                f"log {e}{':' if listed else ''} {', '.join(listed)}".rstrip(),
                COLOR_GRAY,
            )

    elif m := match(r, ("callcode", ":gas", ":addr", ":wei", ":fname", ":fparams")):
        gas, addr, wei, fname, fparams = m.gas, m.addr, m.wei, m.fname, m.fparams
        fname, fparams = split_selector(fname, fparams)
        name = callee_name(fname, add_color=add_color)

        if type(addr) == int:
            addr = hex(addr)
        addr = prettify(addr, add_color=add_color)
        gas = prettify(gas, parentheses=False, add_color=add_color)

        if name is not None:
            yield f"{COLOR_WARNING}codecall{ENDC} {addr}.{name} with:"
        else:
            yield f"{COLOR_WARNING}codecall{ENDC} {addr} with:"
            if fname is not None:
                yield "   funct " + prettify(fname, add_color=add_color)

        if wei != 0:
            wei = prettify(wei, parentheses=False, add_color=add_color)
            yield f"   value {wei} {COLOR_GRAY}wei{ENDC}"

        yield f"     gas {gas}"

        if fparams := pretty_memory(fparams, add_color=add_color, selector=True):
            yield "    args {}".format(", ".join(fparams))

    elif m := match(r, ("delegatecall", ":gas", ":addr", ":fname", ":fparams")):
        gas, addr, fname, fparams = m.gas, m.addr, m.fname, m.fparams
        fname, fparams = split_selector(fname, fparams)
        name = callee_name(fname, add_color=add_color)

        if type(addr) == int:
            addr = hex(addr)
        addr = prettify(addr, add_color=add_color)
        gas = prettify(gas, parentheses=False, add_color=add_color)

        if name is not None:
            yield f"{COLOR_WARNING}delegate{ENDC} {addr}.{name} with:"
        else:
            yield f"{COLOR_WARNING}delegate{ENDC} {addr} with:"
            if fname is not None:
                yield "   funct " + prettify(fname, add_color=add_color)

        yield f"     gas {gas}"

        if fparams := pretty_memory(fparams, add_color=add_color, selector=True):
            yield "    args {}".format(", ".join(fparams))

    elif opcode(r) == "selfdestruct":
        addr = r[1]
        yield col("selfdestruct(", COLOR_WARNING) + col(
            pret(addr, add_color=False, parentheses=False), FAIL
        ) + col(")", COLOR_WARNING)

    elif m := match(r, ("precompiled", ":var_name", ":func_name", ":params")):
        yield "{} = {}({}) {}".format(
            col(var_name(m.var_name), COLOR_BLUE),
            m.func_name,
            ", ".join(pretty_memory(m.params, add_color=add_color)),
            COLOR_GRAY + "# precompiled" + ENDC,
        )

    elif m := match(r, ("create", ":wei", ":code")):
        yield f"create contract with {pret(m.wei)} wei"
        yield f"                code: {', '.join(pretty_memory(m.code, add_color=add_color))}"

    elif m := match(r, ("create2", ":wei", ":code", ":salt")):
        yield f"create2 contract with {pret(m.wei)} wei"
        yield f"                salt: {pret(m.salt)}"
        yield f"                code: {', '.join(pretty_memory(m.code, add_color=add_color))}"

    elif m := match(r, ("call", ":gas", ":addr", ":wei", ":fname", ":fparams")):
        gas, addr, wei, fname, fparams = m.gas, m.addr, m.wei, m.fname, m.fparams
        fname, fparams = split_selector(fname, fparams)

        if type(addr) == int:
            if len(hex(addr)) > 22 + 2:
                addr = padded_hex(addr, 40)  # todo: padded hex
            else:
                addr = hex(addr)  # if it's longer, padded hex returns '???'

        addr = pret(addr)
        gas = pretty_gas(gas, wei, add_color)

        name = callee_name(fname, add_color=add_color)
        if name is not None:
            yield f"call {addr}.{name} with:"
        else:
            yield f"call {addr} with:"
            if fname is not None:
                yield f"   funct {pret(fname)}"

        if wei != 0:
            wei = prettify(wei, parentheses=False, add_color=add_color)
            yield f"   value {wei} {COLOR_GRAY}wei{ENDC}"

        yield f"     gas {gas}"

        if fparams := pretty_memory(fparams, add_color=add_color, selector=True):
            yield "    args {}".format(", ".join(fparams))

    elif m := match(r, ("staticcall", ":gas", ":addr", ":wei", ":fname", ":fparams")):
        gas, addr, wei, fname, fparams = m.gas, m.addr, m.wei, m.fname, m.fparams
        fname, fparams = split_selector(fname, fparams)
        if type(addr) == int:
            addr = hex(addr)

        addr = prettify(addr, add_color=add_color, parentheses=False)
        gas = pretty_gas(gas, wei, add_color)

        name = callee_name(fname, add_color=add_color)
        if name is not None:
            yield f"static call {addr}.{name} with:"
        else:
            yield f"static call {addr} with:"
            if fname is not None:
                yield f"     funct {pret(fname)}"

        yield f"        gas {gas}"

        if fparams := pretty_memory(fparams, add_color=add_color, selector=True):
            yield "       args {}".format(", ".join(fparams))

    elif m := match(r, ("label", ":name", ":setvars")):
        yield COLOR_GREEN + f"label {str(m.name)} setvars: {str(m.setvars)}" + ENDC

    elif opcode(r) == "goto":
        _, *rest = r
        yield COLOR_GREEN + f"continue {str(rest)}" + ENDC

    elif m := match(r, ("continue", ":jd", ":setvars")):
        for v in sequential_setvars(m.setvars):
            yield str(list(pretty_line(v, add_color=True))[0])
        yield COLOR_GREEN + "continue " + ENDC  # +str(jd)+ENDC

    elif opcode(r) == "setvar":
        yield prettify(r, add_color=add_color)

    elif m := match(r, ("internal", ":entry", ":args", ":results")):
        # the call of a recursive internal function (see vm.VM.internal),
        # the variables of its results
        call = internal_name(m.entry) + "(" + ", ".join(pret(a) for a in m.args) + ")"
        if m.results:
            yield ", ".join(pret(("var", v)) for v in m.results) + " = " + call
        else:
            yield call

    elif m := match(r, ("leave", ":values")):
        # the end of a recursive internal function: its results
        yield "return " + ", ".join(pret(v) for v in m.values) if m.values else "return"

    elif opcode(r) == "setmem":
        yield prettify(r, add_color=add_color)

    elif m := match(r, ("set", ":idx", ":val")):
        idx, val = m.idx, m.val

        if m := match(val, ("add", ":int:v", idx)):
            v = m.v

            if v == -1:
                yield prettify(idx, add_color=add_color) + "--"
            elif v == 1:
                yield prettify(idx, add_color=add_color) + "++"

            elif v < 0:
                yield prettify(idx, add_color=add_color) + " -= " + prettify(
                    -v, add_color=add_color, parentheses=False
                )
            else:
                yield prettify(idx, add_color=add_color) + " += " + prettify(
                    v, add_color=add_color, parentheses=False
                )

        elif m := match(val, ("add", idx, ("mul", -1, ":v"))):
            v = m.v
            yield prettify(idx, add_color=add_color) + " -= " + prettify(
                v, add_color=add_color, parentheses=False
            )
        elif m := match(val, ("add", idx, ":v")):
            v = m.v
            yield prettify(idx, add_color=add_color) + " += " + prettify(
                v, add_color=add_color, parentheses=False
            )
        elif m := match(val, ("add", ("mul", -1, ":v"), idx)):
            v = m.v
            yield prettify(idx, add_color=add_color) + " -= " + prettify(
                v, add_color=add_color, parentheses=False
            )
        elif m := match(val, ("add", ":v", idx)):
            v = m.v
            yield prettify(idx, add_color=add_color) + " += " + prettify(
                v, add_color=add_color, parentheses=False
            )
        else:
            yield prettify(idx, add_color=add_color) + " = " + prettify(
                val, add_color=add_color, parentheses=False
            )

    elif opcode(r) == "stop":
        yield "stop"

    elif opcode(r) == "undefined":
        params = tuple(r[1:])
        yield COLOR_WARNING + "..." + ENDC + COLOR_GRAY + f"  # Decompilation aborted, sorry: {params}" + ENDC

    elif opcode(r) == "invalid":
        # not a revert: all the gas is used (an assert of solidity < 0.8, a
        # jump to where it can't)
        yield "invalid"

    elif r == ("revert", None):
        yield "revert"

    elif (
        m := match(r, (":op", ("mem", ("range", ":mem_idx", ":mem_len"))))
    ) and m.op in ("revert", "return"):
        op, mem_idx, mem_len = m.op, m.mem_idx, m.mem_len

        if op == "revert":
            yield "revert with memory"
        else:
            yield op + " memory"

        if m := match(mem_len, ("sub", ":mem_until", mem_idx)):
            yield f"  from  {pret(mem_idx)}"
            yield f"    to {pret(m.mem_until)}"
        else:
            yield "  from " + pret(mem_idx)
            yield "   " + col("len", COLOR_WARNING) + " " + pret(mem_len)

    elif opcode(r) in ("return", "revert") and len(r) == 2:
        op, param = r

        if op == "revert":
            op = "revert with"

        if op == "revert with" and match(param, ("bytes", 4, int)):
            # a custom error without params: its selector alone
            param = ("data", param)

        res_mem = pretty_memory(
            param, add_color=True, abi_text=True, selector=op == "revert with"
        )
        ret_val = ", ".join(res_mem)

        if not res_mem:
            # no data (see OUTPUT.md)
            yield "stop" if op == "return" else "revert"
        elif m := match(r, ("revert", ("data", ("bytes", 4, PANIC), ":int:panic_code"))):
            explanation = (
                (f" {COLOR_GRAY}# " + PANIC_CODES[m.panic_code] + ENDC)
                if m.panic_code in PANIC_CODES
                else ""
            )
            yield f"{op} Panic({m.panic_code}) {explanation}"
        elif (
            (m := match(r, ("revert", ("data", ("bytes", 4, ERROR), ...))))
            and len(res_mem) == 2
            and res_mem[1][:1] == "'"
        ):
            # revert("...") / require(..., "...")
            yield f"{op} {res_mem[1]}"
        elif len(clean_color(ret_val)) < 120 or opcode(param) != "data":
            yield f"{op} {ret_val}"
        else:
            # split long returns into lines. e.g. kitties.getKitten, or kitties.tokenMetadata
            #            yield str(len(ret_val))
            res_mem = list(res_mem)
            if res_mem[0] == "32" and len(res_mem) > 1:
                res_mem.pop(0)
                res_mem[0] = (
                    "32, " + res_mem[0]
                )  # happens often, this is probably an array structure,
                # and sole `32` in first line looks ugly

            if len(res_mem) == 1:
                yield f"{op} {res_mem[0]}"
            else:
                yield f"{op} {res_mem[0]}, "
                for idx, l in enumerate(res_mem[1:]):
                    comma = "," if idx != len(res_mem) - 2 else ""
                    yield " " * len(op) + " " + l + comma

    #            assert op == 'revert'
    #            yield "{} with {}".format(op, ret_val) # adding 'with' to make it more readable

    elif m := match(r, ("store", ":size", ":off", ":idx", ":val")):
        size, off, idx, val = m.size, m.off, m.idx, m.val
        stor_addr = prettify(("stor", size, off, idx), add_color=add_color)
        stor_val = prettify(val, add_color=add_color, parentheses=False)

        yield "{} = {}".format(stor_addr, stor_val)

    elif m := match(r, ("tstore", ":key", ":val")):
        yield "{} = {}".format(
            prettify(("tload", m.key), add_color=add_color),
            prettify(m.val, add_color=add_color, parentheses=False),
        )

    elif type(r) == list and len(r) > 1:
        yield "{} {}".format(
            r[0],
            ", ".join([prettify(x, True, False, add_color=add_color) for x in r[1:]]),
        )

    elif type(r) == list:
        yield str(r[0])

    else:
        yield str(r)


def pretty_type(t):
    if m := match(t, ("def", ":name", ":loc", ("mask", ":size", ":off"))):
        return (
            pretty_type(("def", m.name, m.loc, m.size))
            + COLOR_GRAY
            + (f" offset {m.off}" if m.off > 0 else "")
            + ENDC
        )

    elif m := match(t, ("def", ":name", ":loc", ":bts")):
        name, loc, bts = m.name, m.loc, m.bts
        if type(loc) == int and loc > 1000:
            loc = hex(loc)
        return f"  {COLOR_GREEN}{name}{ENDC} is {pretty_type(bts)} {COLOR_GRAY}at storage {loc}{ENDC}"

    elif t == ("struct", 1):
        return "struct"

    elif t == "bytes":
        # a string has the same storage
        return "bytes"

    elif t == "struct":
        return "struct"

    elif m := match(t, ("struct", ":int:num")):
        return f"struct {m.num} bytes"

    elif m := match(t, ("array", ":bts")):
        return f"array of " + pretty_type(m.bts)

    elif m := match(t, ("mapping", ":bts")):
        return f"mapping of " + pretty_type(m.bts)

    elif type(t) == int:
        return mask_to_type(t, force=True)

    else:
        assert False, f"unknown type {t}"


def pretty_loc(loc, add_color=True):
    """a place of the storage (see storage.py): name[key].field_n, stor[slot]"""
    col = partial(colorize, color=COLOR_GREEN, add_color=add_color)
    pret = partial(prettify, parentheses=False, add_color=add_color)
    op = loc[0]
    if op == "sv":
        return col(loc[1])
    if op == "si":
        return pretty_loc(loc[1], add_color) + col("[") + pret(loc[2]) + col("]")
    if op in ("sl", "sbl"):
        return pretty_loc(loc[1], add_color) + col(".length")
    if op == "sf":
        return pretty_loc(loc[1], add_color) + col(".field_") + pret(loc[2])
    if op == "sr":
        return col("stor[") + pret(loc[1]) + col("]")
    return str(loc)


def pretty_stor(exp, add_color=True):
    col = partial(colorize, color=COLOR_GREEN, add_color=add_color)
    stor = partial(pretty_stor, add_color=add_color)
    pret = partial(prettify, parentheses=False, add_color=add_color)

    if m := match(exp, ("stor", ("length", ":idx"))):
        return stor(m.idx) + col(".length")

    if m := match(exp, ("loc", ":loc")):
        return col(f"stor_l{m.loc}")

    if m := match(exp, ("name", ":name", ":loc")):
        return col(m.name)

    #    if exp ~ ('stor', (:op, :param)) and op in ('loc', 'name'):
    # with top-level fields, it's just a different stor
    # variable. with lower-level we treat it as a struct
    #        return stor((op, param))

    if m := match(exp, ("stor", ":loc")):
        # with top-level fields, it's just a different stor
        # variable. with lower-level we treat it as a struct
        return stor(m.loc)

    if m := match(exp, ("field", ":off", ":loc")):
        return stor(m.loc) + col(f".field_{pret(m.off, add_color=False)}")

    if m := match(exp, ("type", ":size", ":loc")):
        if m.size == 256:
            # prettify removes 256 masks by default, force it
            return (
                col("uint256(", color=COLOR_GRAY)
                + stor(m.loc)
                + col(")", color=COLOR_GRAY)
            )
        else:
            return pret(("mask", m.size, 0, stor(m.loc)))

    def pr_idx(idx):
        if opcode(idx) == "data":
            _, *terms = idx
            return col("][").join([pret(t) for t in terms])
        else:
            return pret(idx)

    if m := match(exp, ("map", ":idx", ":var")):
        idx, var = m.idx, m.var
        return stor(var) + col("[") + pr_idx(idx) + col("]")

    if m := match(exp, ("array", ("mul", ":int:any", ":idx"), ":var")):
        exp = (
            "array",
            m.idx,
            m.var,
        )  # nasty hack to not display storage[2*idx] for storages
        # that are structs
        # this should be handled in sparser really
    #        return stor(var) + col('[') + pr_idx(idx) + col(']')

    if m := match(exp, ("array", ":idx", ":var")):
        idx, var = m.idx, m.var
        return stor(var) + col("[") + pr_idx(idx) + col("]")

    if m := match(exp, ("length", ":var")):
        return stor(m.var) + col(".length")

    if m := match(exp, ("stor", ":loc")):
        return col("stor[") + pret(m.loc) + col("]")

    if m := match(exp, ("stor", ":size", ":off", ":loc")):
        return pret(("mask", m.size, m.off, col("stor[") + pret(m.loc) + col("]")))

    return col("stor[") + pret(exp) + col("]")


def pretty_num(exp, add_color):
    if type(exp) == float:
        if exp - int(exp) == 0:
            exp = int(exp)

    if type(exp) == int and exp > 8**50:
        return hex(
            exp
        )  # dealing with binary data probably, usually in call code - display in hex

    if type(exp) == int and exp != 0:
        count = 18
        while count >= 9:
            if exp % (10**count) == 0:
                if exp // (10**count) == 1:
                    return f"10**{count}"
                else:
                    return f"{exp // (10**count)} * 10**{count}"

            count -= 1

        count = 6
        if exp % (10**count) == 0:
            if exp // (10**count) == 1:
                return f"10**{count}"
            else:
                return f"{exp // (10**count)} * 10**{count}"

    if type(exp) == int:
        if (
            type(exp) == int and (exp & 2**256 - 1) < 8**30
        ):  # if it's larger than 30 bytes, it's probably
            # an address, not a negative number
            # (the word it is: -2**256 + 128 is 128)
            return str(to_real_int(exp & 2**256 - 1))

        elif exp > 0:
            return hex(exp)

        else:
            return str(exp)

    # print('warn: weird float exp', exp)
    return str(exp)


"""

    Precedence of the operators as printed, the higher binding tighter, as
    in Python: an operand is in parentheses if its operator binds less
    tightly than it needs to. `parentheses` is how tightly an expression
    needs to bind where it's printed: an int, False for nothing (a whole
    line, a subscript, an argument), True for anything but an atom.

"""

OR, AND, NOT, CMP, BOR, BXOR, BAND, SHIFT, ADD, MUL, UNARY, POW, ATOM = range(1, 14)

OPERATOR_PRECEDENCE = {
    " or ": OR,
    " and ": AND,
    " | ": BOR,
    " ^ ": BXOR,
    " & ": BAND,
    " == ": CMP,
    " != ": CMP,
    " < ": CMP,
    " > ": CMP,
    " <= ": CMP,
    " >= ": CMP,
    " <′ ": CMP,
    " >′ ": CMP,
    " <=′ ": CMP,
    " >=′ ": CMP,
    " << ": SHIFT,
    " >> ": SHIFT,
    " + ": ADD,
    " - ": ADD,
    " +′ ": ADD,
    " * ": MUL,
    " / ": MUL,
    " % ": MUL,
    " *′ ": MUL,
    " /′ ": MUL,
    " %′ ": MUL,
    "**": POW,
}


def context(parentheses, top_level=False):
    if top_level or parentheses is False or parentheses is None:
        return 0
    if parentheses is True:
        return ATOM
    return parentheses


def num_precedence(text):
    """The precedence of a number as pretty_num prints it."""
    if " * " in text:
        return MUL
    if text.startswith("-"):
        return UNARY
    if "**" in text:
        return POW
    return ATOM


# the names of the variables of loops ('i','j','k','l','m','n','o','p','q','r'
# aren't: they read as indexes), the ones the text doesn't have for something
# else (see set_names)
NICE_NAMES = (
    "idx",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
)

# the names the text has for what the EVM gives, and the first names of the
# ones it has with a dot (call.value, block.number...): no param nor variable
# has one of them
BUILTIN_NAMES = frozenset(
    (
        "caller",
        "chainid",
        "gas_remaining",
        "call",
        "calldata",
        "this",
        "tx",
        "block",
        "return_data",
        "ext_call",
        "ext_code",
        "memcopy",
        "delegate",
        "create",
        "create2",
        "code",
        "eth",
        "mem",
        "stor",
        "transient",
        "msize",
        "True",
        "False",
    )
    # (ecrecover.result...)
    + tuple(precompiled.values())
)

_names = {"storage": frozenset(), "taken": BUILTIN_NAMES}


def set_names(storage=None, params=None):
    """
    The names of the storage variables (of the contract), and of the params
    (of the function printed): the variables of its loops have others - `s =
    ...` would be a write to the storage variable s, and `x` a param x.
    """
    if storage is not None:
        _names["storage"] = frozenset(storage)
    _names["taken"] = BUILTIN_NAMES | _names["storage"] | frozenset(params or ())


def var_name(name):
    """
    The name a variable with a name (the result of a precompiled contract,
    `_3`...) is printed with: its own, then _ while that's the name of
    something else - `signer_ = ecrecover(...)` where signer is a storage
    variable or a param.
    """
    while name in _names["taken"]:
        name += "_"
    return name


def prettify(exp, rem_bool=False, parentheses=True, top_level=False, add_color=False):
    col = partial(colorize, add_color=add_color)
    pret = partial(prettify, add_color=add_color, parentheses=False)
    ctx = context(parentheses, top_level)

    def wrap(text, prec):
        return f"({text})" if prec < ctx else text

    def operand(e, prec, **kw):
        # e, printed as needing to bind at least as tightly as prec
        return prettify(e, parentheses=prec, add_color=add_color, **kw)

    if rem_bool:
        exp = simplify_bool(exp)
        if m := match(exp, ("xor", ":a", ":b")):
            # true when it isn't 0: when they differ
            exp = ("iszero", ("eq", m.a, m.b))
        if opcode(exp) == "bool":
            return prettify(
                exp,
                rem_bool=rem_bool,
                parentheses=parentheses,
                top_level=top_level,
                add_color=add_color,
            )

    if type(exp) == int and exp % (24 * 3600) == 0 and exp > 24 * 3600:
        exp = ("mul", exp // (24 * 3600), 24, 3600)

    if type(exp) == int and exp % 3600 == 0 and exp > 3600:
        exp = ("mul", exp // 3600, 3600)
        # also tried return col('seconds(', COLOR_GRAY) + '1 hour' + col(')', COLOR_GRAY)
        # but seemed less intuitive, e.g. 0xf64B584972FE6055a770477670208d737Fff282f calcMaxWithdraw
        # and 3600 every programmer should know, by heart, means 1 hour :)
        #
        # also, not tackling single minutes because too often they are not time related

    if type(exp) in (int, float):
        text = pretty_num(exp, add_color)
        return wrap(text, num_precedence(text))

    if opcode(exp) in precompiled.values():
        return f"{exp[0]}({pret(exp[1])})"

    if (
        m := match(exp, ("arr", ":int:num", ("mask_shl", Any, Any, Any, ":str:s")))
    ) and len(m.s) == m.num + 2:
        return m.s

    if m := match(exp, ("param", ":name")):
        return col(m.name, COLOR_GREEN)

    if m := match(exp, ("range", ":loc", ":size")):
        return "{} {} {}".format(pret(m.loc), col("len", COLOR_HEADER), pret(m.size))

    if opcode(exp) == "data":
        # the bytes of its elements one after the other
        return "concat(" + ", ".join(pretty_memory(exp, add_color=add_color)) + ")"

    if m := match(exp, ("bytes", ":size", ":val")):
        return pretty_bytes(m.size, m.val, add_color, parentheses=ctx)

    if m := match(exp, ("signextend", ":int:b", ":val")):
        val = m.val
        if (m2 := match(val, ("type", ":size", ":loc"))) and m2.size == 8 * (m.b + 1):
            # a field of the storage of that size: an int rather than an uint
            val = m2.loc
        return (
            col(f"int{8 * (m.b + 1)}(", COLOR_GRAY) + pret(val) + col(")", COLOR_GRAY)
        )

    if m := match(exp, ("signextend", ":b", ":val")):
        return f"signextend({pret(m.b)}, {pret(m.val)})"

    if opcode(exp) == "arr" and len(exp) > 1:
        _, l, *terms = exp
        return (
            col("Array(len=", COLOR_GRAY)
            + pret(l)
            + col(", data=", COLOR_GRAY)
            + ", ".join(pretty_memory(("data",) + tuple(terms), add_color=add_color))
            + col(")", COLOR_GRAY)
        )

    if m := match(exp, ("blockhash", ":number")):
        return f"block.hash({pret(m.number)})"

    if m := match(exp, ("extcodehash", ":addr")):
        return f"ext_code.hash({pret(m.addr)})"

    if m := match(exp, ("extcodesize", ":addr")):
        return f"ext_code.size({pret(m.addr)})"

    if m := match(exp, ("extcodecopy", ":addr", ("range", ":start", ":size"))):
        # the bytes of the code of the account (see OUTPUT.md)
        return f"ext_code({pret(m.addr)}).data[{pret(m.start)} len {pret(m.size)}]"

    if opcode(exp) in ("max", "min", "byte", "clz"):
        _, *terms = exp
        return "{}({})".format(opcode(exp), ", ".join([pret(e) for e in terms]))

    if opcode(exp) == "safemath":
        # a function of the contract's def SafeMath (see safemath.py): its
        # operands (one for inc and dec), then the messages it fails with,
        # string literals
        _, name, *args = exp
        n = 1 if name.rstrip("0123456789") in ("inc", "dec") else 2
        return (
            f"SafeMath.{name}("
            + ", ".join([pret(e) for e in args[:n]] + list(args[n:]))
            + ")"
        )

    if exp == "number":
        return "block.number"

    if exp == "calldatasize":
        return "calldata.size"

    if exp == "returndatasize":
        return "return_data.size"

    if exp == "difficulty":
        return "block.difficulty"

    if exp == "basefee":
        return "block.basefee"

    if exp == "blobbasefee":
        return "block.blobbasefee"

    if m := match(exp, ("blobhash", ":idx")):
        return f"blobhash({pret(m.idx)})"

    if m := match(exp, ("tload", ":key")):
        return col("transient[", COLOR_GRAY) + pret(m.key) + col("]", COLOR_GRAY)

    if exp == "gasprice":
        # (of the transaction)
        return "tx.gasprice"

    if exp == "timestamp":
        return "block.timestamp"

    if exp == "coinbase":
        return "block.coinbase"

    if exp == "gaslimit":
        return "block.gas_limit"

    if exp == "callvalue":
        return "call.value"

    if exp == "address":
        return "this.address"

    if exp == ("mask_shl", 160, 0, 0, "caller"):
        return "caller"

    if exp == "caller":
        return "caller"

    if exp == ("mask_shl", 160, 0, 0, "origin"):
        return "tx.origin"

    if (m := match(exp, (":op", ":a", ":b", ":c"))) and m.op in ("mulmod", "addmod"):
        return f"{m.op}({pret(m.a)}, {pret(m.b)}, {pret(m.c)})"

    if exp == "origin":
        return "tx.origin"

    if exp == "gas":
        return "gas_remaining"

    if exp == ("bool", 1):
        return "True"

    if exp == ("bool", 0):
        return "False"

    if m := match(exp, ("code.data", ":c_start", ":c_len")):
        return f"code.data[{pret(m.c_start)} len {pret(m.c_len)}]"

    if m := match(exp, ("balance", ":addr")):
        return f"eth.balance({pret(m.addr)})"

    if opcode(exp) == "sha3":
        # of the bytes of its terms one after the other
        _, *terms = exp
        if len(terms) == 1 and opcode(terms[0]) == "data":
            terms = terms[0][1:]
        return "sha3({})".format(
            ", ".join(pretty_memory(("data",) + tuple(terms), add_color=add_color))
        )

    #    if exp ~ ('mask_shl', 251, 5, 0, :val):
    #        return pret(('mul', 32, val))

    if (m := match(exp, ("mask_shl", 251, 5, 0, ":val"))) or (
        m := match(exp, ("mask", 251, 5, ":val"))
    ):
        val = m.val
        if m := match(val, ("add", 31, ":num")):
            return f"ceil32({pret(m.num)})"
        else:
            return f"floor32({pret(val)})"

    if (
        m := match(exp, ("call.data", ("add", 36, ("param", ":p_name")), ":size"))
    ) and m.size == ("cd", ("add", 4, ("param", m.p_name))):
        return f"{col(m.p_name+'[', C.green)}" + "all" + col("]", C.green)

    if (m := match(exp, (":name", ":offset", ":size"))) and is_array(
        m.name
    ):  # in ('call.data', 'ext_call.return_data'):
        if m.size == 32:
            return m.name + f"[{pret(m.offset)}]"
        else:
            return m.name + f"[{pret(m.offset)} len {pret(m.size)}]"

    if (
        (
            m := match(
                exp,
                (
                    "mask_shl",
                    ":size",
                    ":offset",
                    ":shl",
                    ("stor", ":s_size", ":s_off", ":s_idx"),
                ),
            )
        )
        and safe_le_op(m.s_size, m.size)
        and m.shl == 0
        and m.offset == 0
    ):
        return pret(("stor", m.s_size, m.s_off, m.s_idx))

    if (
        (m := match(exp, ("mask_shl", ":int:size", ":int:off", ":int:shl", ":val")))
        and opcode(m.val) == "st"
        and type(m.val[1]) == int
        and 0 < m.off == -m.shl
        and m.val[1] <= m.off + m.size
    ):
        # a storage access of fewer bits than the mask's top: its bits from
        # off on, x >> off
        if m.off < 8:
            return pret(("div", m.val, 2**m.off), parentheses=ctx)
        op_form = COLOR_BOLD + " >> " + ENDC if add_color else " >> "
        return wrap(operand(m.val, SHIFT) + op_form + pret(m.off), SHIFT)

    if m := match(exp, ("sall", ":loc")):
        # the bytes of a bytes (or string) of the storage
        return pretty_loc(m.loc, add_color) + col("[all]", COLOR_GREEN)

    if m := match(exp, ("st", ":size", ":loc", ":width")):
        # an access of the storage (see storage.py)
        text = pretty_loc(m.loc, add_color)
        if m.size == m.width or m.width is None:
            return text
        name = "address" if m.size == 160 else f"uint{pret(m.size)}"
        return col(name + "(", COLOR_GRAY) + text + col(")", COLOR_GRAY)

    if opcode(exp) == "stor":
        return pretty_stor(exp, add_color=add_color)

    if opcode(exp) == "type":
        return pretty_stor(exp, add_color=add_color)

    if opcode(exp) == "field":
        return pretty_stor(exp, add_color=add_color)

    if m := match(exp, ("cd", ":num")):
        parsed_exp = get_param_name(exp, add_color=add_color)

        if type(parsed_exp) != str:
            return "cd[" + pret(parsed_exp[1]) + "]"
        else:
            return parsed_exp

    if m := match(exp, ("var", ":int:idx")):
        # (not a name the text has for something else, see set_names)
        free = [n for n in NICE_NAMES if n not in _names["taken"]]
        if m.idx < len(free):
            name = free[m.idx]
        else:
            name = "var" + str(m.idx)
            while name in _names["taken"]:
                name += "_"

        return col(name, COLOR_BLUE)

    if m := match(exp, ("var", ":name")):
        return col(var_name(str(m.name)), COLOR_BLUE)

    if m := match(exp, ("mem", ("range", ":loc", 32))):
        exp = ("mem", m.loc)

    if m := match(exp, ("mem", ("range", ":loc", ":size"))):
        return (
            col("mem[", COLOR_HEADER)
            + pret(m.loc)
            + col(" len ", COLOR_HEADER)
            + pret(m.size)
            + col("]", COLOR_HEADER)
        )

    if m := match(exp, ("mem", ":idx")):
        assert opcode(m.idx) != "range"

        return col("mem[", COLOR_HEADER) + pret(m.idx) + col("]", COLOR_HEADER)

    if m := match(exp, ("setvar", ":idx", ":val")):  # shouldn't be pretty line?
        return pret(("var", m.idx)) + " = " + pret(m.val, parentheses=False)

    if m := match(exp, ("setmem", ":idx", ":val")):  # --,,--
        val = m.val
        if type(val) == int and val >= 2**256 and (r := match(m.idx, ("range", Any, ":int:n"))):
            # bytes of more than a word: as many as the range
            val = ("bytes", r.n, val)
        return pret(("mem", m.idx)) + " = " + ", ".join(pretty_memory(val, add_color))

    if exp == ("mask_shl", 32, 224, 0, ("cd", 0)):
        # the first 4 bytes of the calldata, as msg.sig
        return col("call.func_hash", C.green)

    if exp == ("mask_shl", 32, 224, -224, ("cd", 0)):
        return wrap(col("call.func_hash", C.green) + " >> 224", SHIFT)

    if m := match(exp, ("mask_shl", ":size", ":offset", ":shl", ":val")):
        size, offset, shl, val = m.size, m.offset, m.shl, m.val

        if (
            all_concrete(size, offset, shl)
            and exp[1] + exp[2] == 256
            and exp[2] == -exp[3]
            and exp[2] < 8
            and not may_be_wide(val)
        ):
            # e.g. (Mask(255, 1, eth.balance(this.address)) >> 1
            #           --> eth.balance(this.address) / 2
            # for offsets smaller than 8 (of a word: the top of the mask is
            # its top)

            if exp[3] <= 8:
                return pret(("div", exp[4], 2 ** -exp[3]), parentheses=ctx)
            else:
                return pret(("shr", exp[3], exp[4]), parentheses=ctx)

        if (
            (type(exp[1]), type(exp[2]), type(exp[3])) == (int, int, int)
            and exp[2] == exp[3]
            and exp[2] < 8
        ):
            # e.g. (Mask(255, 1, eth.balance(this.address)) << x
            #           --> eth.balance(this.address) * 2**x
            # for offsets smaller than 8

            if (
                (size + offset != 256 or offset != 0 or may_be_wide(val))
                and opcode(val) != "store"
            ):  # opcode=store - hotfix for 0x000000000045Ef846Ac1cB7fa62cA926D5701512
                # the bits [offset, offset + size) in place: not the ones below
                # offset, which the mask clears (nor, of a value of more than
                # a word, the ones past 256)
                val = ("mask", size, offset, val)

            if exp[3] == 0:
                return pret(val, parentheses=ctx)
            elif exp[3] <= 8 and exp[3] >= -8:
                return pret(("mul", val, 2 ** exp[3]), parentheses=ctx)
            elif exp[3] > 0:
                return pret(
                    (
                        "shl",
                        exp[3],
                        val,
                    ),
                    parentheses=ctx,
                )
            else:
                return pret(
                    (
                        "shr",
                        -exp[3],
                        val,
                    ),
                    parentheses=ctx,
                )

        if all_concrete(size, offset, shl, val):
            return pret(apply_mask(exp[4], exp[1], exp[2], exp[3]), parentheses=ctx)

        if (
            all_concrete(size, offset, shl)
            and offset == -shl
            and 8 <= offset < 256
            and 0 < size <= 256 - offset
            and opcode(val) != "data"
        ):
            # val >> offset, with the bits above size cut off if there are any
            # e.g. Mask(16, 160, x) >> 160 --> uint16(x >> 160)
            op_form = COLOR_BOLD + " >> " + ENDC if add_color else " >> "
            shifted = operand(val, SHIFT) + op_form + pret(offset)

            if size + offset == 256:
                return wrap(shifted, SHIFT)

            if size == 1:
                # its lowest bit: not a bool (bool(x) is x != 0)
                op_mod = COLOR_BOLD + " % " + ENDC if add_color else " % "
                return wrap(f"({shifted}){op_mod}2", MUL)

            type_name = mask_to_type(size)
            if type_name is None and size % 8 == 0:
                type_name = f"uint{size}"

            if type_name is not None:
                return col(type_name + "(", COLOR_GRAY) + shifted + col(")", COLOR_GRAY)

        if (
            not all_concrete(offset, shl)
            and add_op(offset, shl) == 0
            and is_word(offset)
            and add_op(size, offset) == 256
        ):
            # its bits from offset up, moved down: val >> offset (0 when the
            # offset is 256 or more, as the mask is then)
            op_form = COLOR_BOLD + " >> " + ENDC if add_color else " >> "
            return wrap(
                operand(val, SHIFT) + op_form + operand(offset, SHIFT + 1), SHIFT
            )

        sign = shift_sign(shl)
        mask = ("mask", size, offset, val)

        if shl == 0:
            exp = mask

        elif sign == 1:
            if all_concrete(size, offset, shl) and size + shl == 256 and offset == 0:
                # the bits that the multiplication keeps
                exp = ("mul", 2**shl, val)
            elif type(shl) == int and shl < 7:
                exp = ("mul", 2**shl, mask)
            else:
                exp = ("shl", shl, mask)

        elif sign == -1:
            right = minus_op(shl)
            if type(right) == int and right <= 8:
                exp = ("div", mask, 2**right)
            else:
                exp = ("shr", right, mask)

        else:
            # a shift that may be one way or the other (see OUTPUT.md)
            exp = ("shift", mask, shl)

    if m := match(exp, ("mask", ":size", 0, ":val")):
        size, val = m.size, m.val

        if size == 256 and not may_be_wide(val):
            return pret(val, parentheses=ctx)

        if type(size) == int and size not in (1, 256):
            # (the lowest bit of a number isn't a bool: x % 2, below)
            if size == 255:
                type_name = "uint255"
            else:
                type_name = mask_to_type(size)
                if type_name is None and 0 < size < 256 and size % 8 == 0:
                    # uint24, uint96... like the ones above
                    type_name = f"uint{size}"

            if type_name is not None:
                return (
                    col(type_name + "(", COLOR_GRAY) + pret(val) + col(")", COLOR_GRAY)
                )

    if m := match(exp, ("bool", ":val")):
        if opcode(m.val) in ("lt", "gt", "iszero", "le", "ge", "bool"):
            return pret(m.val, parentheses=ctx)
        else:
            return "bool(" + pret(m.val) + ")"

    if m := match(exp, ("mask", ":size", ":offset", ":val")):
        size, offset, val = m.size, m.offset, m.val
        if type(size) == int and offset == 0 and size < 64:
            return pret(("mod", val, 2**size), parentheses=ctx)
        else:
            return "Mask({}, {}, {})".format(pret(size), pret(offset), pret(val))

    if m := match(exp, ("shift", ":val", ":amount")):
        return f"shift({pret(m.val)}, {pret(m.amount)})"

    if (m := match(exp, (":op", ":off", ":val"))) and (
        m.op == "sar" or (m.op == "shr" and not isinstance(m.off, int))
    ):
        # the shifts that the vm leaves as they are: the arithmetic ones (>>′,
        # like the other signed operations), and the ones by a symbolic amount
        op_form = " >>′ " if m.op == "sar" else " >> "
        if add_color:
            op_form = COLOR_BOLD + op_form + ENDC
        return wrap(operand(m.val, SHIFT) + op_form + operand(m.off, SHIFT + 1), SHIFT)

    #    if opcode(exp) in ('byte', 'bytes8', 'uint16', 'bytes4', 'addr', 'int256'):
    #        return prettify('{}({})'.format(opcode(exp).lower(), prettify(exp[1], add_color=add_color)), add_color=add_color)

    opcode_to_arithm = {
        "sub": " - ",
        "div": " / ",
        "mul": " * ",
        "gt": " > ",
        "lt": " < ",
        "le": " <= ",
        "ge": " >= ",
        "or": " | ",
        "eq": " == ",
        "mod": " % ",
        "shl": " << ",
        "shr": " >> ",
        "exp": "**",
        "and": " & ",
        "sge": " >=′ ",
        "sle": " <=′ ",
        "sgt": " >′ ",
        "slt": " <′ ",
        "sadd": " +′ ",
        "smul": " *′ ",
        "sdiv": " /′ ",
        "smod": " %′ ",
        "xor": " ^ ",
    }

    def pretty_adds(exp):
        if opcode(exp) != "add":
            return prettify(exp, add_color=add_color)

        if type(exp[1]) in (int, float):
            real = exp[1]
            if type(real) == float and int(real) == real:
                real = int(real)  # 32.0 -> 32
            if type(real) == int:
                real = to_real_int(real)
            terms = exp[2:]
        else:
            real = 0
            terms = exp[1:]

        res = ""
        for x in terms:
            if res == "":
                res = operand(x, ADD)
            elif opcode(x) == "mul" and type(x[1]) == int and x[1] < 0:
                res += " - " + operand(minus_op(x), ADD + 1)
            else:
                res += " + " + operand(x, ADD)

        if real > 0 or real <= -(2**128):
            # a big one - a hash, say - is added, however large it is
            res += " + " + operand(real % 2**256, ADD)
        elif real < 0:
            res += " - " + operand(-real, ADD + 1)

        return wrap(res, ADD)

    if opcode(exp) == "not":
        # bitwise, as in python: `not` is the logical one (iszero)
        return wrap(col("~", COLOR_BOLD) + operand(exp[1], UNARY), UNARY)

    if opcode(exp) == "add":
        return pretty_adds(exp)

    if (
        opcode(exp) == "mul"
        and len(exp) == 3
        and to_exp2(exp[1]) != None
        and to_exp2(exp[1]) > 32
    ):
        exp = ("shl", to_exp2(exp[1]), exp[2])

    if m := match(exp, ("mul", -1, ":val")):
        return wrap("-" + operand(m.val, UNARY), UNARY)

    if m := match(exp, ("mul", 1, ":val")):
        return pret(m.val, parentheses=ctx)

    if m := match(exp, ("mul", 1, ...)):
        return pret(("mul",) + exp[2:], parentheses=ctx)

    if m := match(exp, ("div", ":num", 1)):
        return pret(m.num, parentheses=ctx)

    if m := match(exp, ("exp", ":a", ":n")):
        return wrap(operand(m.a, POW + 1) + "**" + operand(m.n, POW), POW)

    if opcode(exp) in ("land", "lor"):
        # python's and / or (see postprocess.short_circuits): the operand that
        # decides - printed as the value it is where it may be the one given
        # (the last of an and, any of an or), unless only its truth counts
        op_form = " and " if opcode(exp) == "land" else " or "
        prec = OPERATOR_PRECEDENCE[op_form]
        if add_color:
            op_form = COLOR_BOLD + op_form + ENDC
        last = len(exp) - 2
        parts = [
            operand(e, prec, rem_bool=rem_bool or (opcode(exp) == "land" and i < last))
            for i, e in enumerate(exp[1:])
        ]
        return wrap(op_form.join(parts), prec)

    if opcode(exp) in opcode_to_arithm:
        if opcode(exp) in ["shl", "shr"]:
            exp = exp[0], exp[2], exp[1]

        if opcode(exp) in ("lt", "gt", "le", "ge"):
            # unsigned: a number of the top half of the words is no negative
            exp = (exp[0],) + tuple(unsigned(e) for e in exp[1:])

        op_form = opcode_to_arithm[opcode(exp)]
        prec = OPERATOR_PRECEDENCE[op_form]
        # the operands after the first bind tighter: a / (b / c) isn't
        # a / b / c, a * (b / c) isn't a * b / c
        rest_prec = prec + 1
        if prec == CMP:
            # comparisons don't chain
            first_prec = CMP + 1
        else:
            first_prec = prec

        if add_color:
            op_form = COLOR_BOLD + op_form + ENDC

        def fold_ands(exp):
            assert opcode(exp) == "and"

            res = tuple()
            for e in exp[1:]:
                if opcode(e) == "and":
                    e = fold_ands(e)
                    res += e[1:]
                else:
                    res += (e,)

            return ("and",) + res

        if opcode(exp) in ("and", "or") and (
            all(is_bool(e) for e in exp[1:]) or (rem_bool and opcode(exp) == "or")
        ):
            # python's and / or: of truth values, they're the bitwise ones;
            # of any values, an or is true where the bitwise one is
            if opcode(exp) == "and":
                exp = fold_ands(exp)
            op_form = " and " if opcode(exp) == "and" else " or "
            prec = OPERATOR_PRECEDENCE[op_form]
            if add_color:
                op_form = COLOR_BOLD + op_form + ENDC
            # where its value counts, not only its truth, they give the
            # operand that decides: the last of an and, any of an or, which
            # is then printed as the 0 or 1 it is (bool(x), not x)
            last = len(exp) - 2
            parts = [
                operand(
                    e,
                    prec + 1,
                    rem_bool=rem_bool or (opcode(exp) == "and" and i < last),
                )
                for i, e in enumerate(exp[1:])
            ]
        else:
            parts = [operand(exp[1], first_prec)] + [
                operand(e, rest_prec) for e in exp[2:]
            ]

        return wrap(op_form.join(parts), prec)

    if m := match(exp, ("iszero", ":val")):
        val = m.val

        def comparison(left, op, right):
            return wrap(operand(left, CMP + 1) + op + operand(right, CMP + 1), CMP)

        if opcode(val) in ("lt", "gt"):
            val = (val[0],) + tuple(unsigned(e) for e in val[1:])

        if m := match(val, ("gt", ":left", ":right")):
            return comparison(m.left, " <= ", m.right)

        if m := match(val, ("lt", ":left", ":right")):
            return comparison(m.left, " >= ", m.right)

        if m := match(val, ("eq", ":left", ":right")):
            if type(m.left) in (str, int):
                return comparison(m.right, " != ", m.left)
            else:
                return comparison(m.left, " != ", m.right)

        # (of its operand, only the truth counts: not bool(x) is not x)
        return wrap("not " + operand(val, NOT, rem_bool=True), NOT)

    return str(exp)


def pretty_gas(gas, value, add_color):
    return prettify(gas, add_color=add_color, parentheses=False)


def event_abi(topic):
    """The abi of the event whose signature is topic, if it's known."""
    if type(topic) != int:
        return None
    topic %= 2**256  # (-1 is the word 2**256 - 1, see simplify_exp)

    abi = fetch_sig(padded_hex(topic, 64)[:10])
    if abi is None or abi.get("type") != "event":
        return None

    # the database is by the first 4 bytes: the whole signature must match
    from eth_hash.auto import keccak

    signature = "{}({})".format(
        abi["name"],
        ",".join(canonical_type(i["type"], i.get("components")) for i in abi["inputs"]),
    )
    if keccak(signature.encode()) != topic.to_bytes(32, "big"):
        return None

    return abi


def fits_bytes(val, size):
    """
    Whether the number val is written in size bytes: neither of them
    negative (a size of a length that is), nor so many bytes that python
    would run out of memory writing them.
    """
    return 0 <= size <= 2**24 and 0 <= val and val.bit_length() <= 8 * size


def pretty_bytes(size, val, add_color=False, parentheses=False):
    """
    ("bytes", size, val): a word is shown as its value, text as a string,
    anything else as its value with its width - Bytes(size, val).
    """
    if size == 32:
        return prettify(val, add_color=add_color, parentheses=parentheses)

    if type(val) == int and type(size) == int and fits_bytes(val, size):
        if text := pretty_text(val.to_bytes(size, "big"), short=True):
            return text
        val = "0x" + format(val, f"0{2 * size}x") if val else "0"
        return f"{colorize('Bytes(', COLOR_GRAY, add_color)}{size}{colorize(', ', COLOR_GRAY, add_color)}{val}{colorize(')', COLOR_GRAY, add_color)}"

    return (
        colorize("Bytes(", COLOR_GRAY, add_color)
        + prettify(size, add_color=add_color, parentheses=False)
        + colorize(", ", COLOR_GRAY, add_color)
        + prettify(val, add_color=add_color, parentheses=False)
        + colorize(")", COLOR_GRAY, add_color)
    )


def sequential_setvars(setvars):
    """
    The setvars of a continue, one after the other.

    They all happen at once: each one reads the values from before any of
    them (idx = idx + 1 and s = s + 3 * idx add the idx of the iteration
    that ends). Printed one per line, they are read one after the other, so
    one that reads a variable goes before the one that sets it, and when two
    read each other's (a swap), a copy of one of them is made first.
    """
    pending = [sv for sv in setvars if sv[2] != ("var", sv[1])]
    res = []
    copies = 0
    while pending:
        for i, (_, idx, _val) in enumerate(pending):
            if not any(
                contains(other[2], ("var", idx))
                for j, other in enumerate(pending)
                if j != i
            ):
                res.append(pending.pop(i))
                break
        else:
            _, idx, _val = pending[0]
            copies += 1
            copy = ("var", "_old" if copies == 1 else f"_old{copies}")
            res.append(("setvar", copy[1], ("var", idx)))
            pending = [pending[0]] + [
                (op, i2, replace(v2, ("var", idx), copy)) for op, i2, v2 in pending[1:]
            ]

    return res


def known_fname(sel, add_color=False):
    """
    The function (or error...) of the selector sel, a number of 4 bytes, as
    it's printed - when the database knows it, else None.
    """
    # (a selector of zero bytes first is 0x00fdd58e, not 0xfdd58e)
    name = Loader.find_sig("0x%08x" % sel, add_color)
    if name is None or re.match(r"unknown_?([0-9a-f]{8})?\(", clean_color(name)):
        # (a name of the database that's no name: nor are its params)
        return None
    # the names the database doesn't have are no names of the callee
    return re.sub(r" _param\d+(?=[,)])", "", name)


def split_selector(fname, fparams):
    """
    The selector and the params of a call (see vm.VM.call_data): a call
    whose data is all in its params, when they start with 4 bytes that are a
    number, has them for selector and the rest for params.
    """
    if fname is not None:
        return fname, fparams
    if match(fparams, ("bytes", 4, ":int:")):
        return fparams, None
    if opcode(fparams) == "data" and match(fparams[1:2], ((("bytes", 4, ":int:")),)):
        rest = fparams[2:]
        return fparams[1], ("data",) + rest if len(rest) > 1 else (rest or (None,))[0]
    return fname, fparams


def callee_name(fname, add_color=False):
    """
    What a call calls, printed after its address (`x.f(...)`): the function
    of its selector when that's known, `unknown1234abcd(?)` when it isn't (no
    params made up), the bytes of memory it is (`x.mem[a len 4]`) - None when
    it's none of these (a selector computed: printed as `funct ...`), or when
    the call has no data.
    """
    if m := match(fname, ("bytes", 4, ":int:val")):
        fname = m.val
    if type(fname) == int and 0 <= fname < 2**32:
        return known_fname(fname, add_color) or f"unknown{fname:08x}(?)"
    if opcode(fname) == "mem":
        return prettify(fname, add_color=add_color)
    return None


# the characters of a text: the printable ones and the whitespace
TEXT_CHARS = set(map(chr, range(0x20, 0x7F))) | {"\n", "\r", "\t"}


def data_bytes(exp):
    """The bytes of exp as an element of a data, if they're known."""
    if m := match(exp, ("bytes", ":int:size", ":int:val")):
        if fits_bytes(m.val, m.size):
            return m.val.to_bytes(m.size, "big")
    elif type(exp) == int and 0 <= exp < 2**256:
        return exp.to_bytes(32, "big")

    return None


def pretty_text(b, short=False):
    """
    The bytes b as a string literal, if they're text: with letters or
    digits, or any printable ones if short (a separator).
    """
    text = b.decode("latin-1")
    if not text or not all(c in TEXT_CHARS for c in text):
        return None
    if not any(c.isalnum() for c in text) and not (short and len(text) <= 2):
        return None

    for char, escaped in (
        ("\\", "\\\\"),
        ("'", "\\'"),
        ("\n", "\\n"),
        ("\r", "\\r"),
        ("\t", "\\t"),
    ):
        text = text.replace(char, escaped)

    return f"'{text}'"


def arr_text(exp):
    """An ("arr", len, ...) of text: as a string literal."""
    if opcode(exp) != "arr" or len(exp) < 2:
        return None
    _, l, *terms = exp
    chunks = [data_bytes(t) for t in terms]
    if type(l) == int and None not in chunks:
        b = b"".join(chunks)
        if len(b) >= l and not any(b[l:]) and (text := pretty_text(b[:l])):
            return text
    return None


def pretty_memory(exp, add_color=False, abi_text=False, selector=False):
    """
    The elements of a list of data, as they're printed. abi_text: it's the
    data of a return or a revert, where a string that's all the data (after
    a selector) is the ABI-encoded string (see OUTPUT.md) - an ABI-encoded
    string is printed 'text' there, and bytes of text that are all the data
    Bytes(n, 'text'). Elsewhere a string is its bytes, and an ABI-encoded
    one an Array(len=n, data='text'). selector: it's the data of a revert or
    of a call, whose first 4 bytes are a selector, printed as the error or
    the function it is (elsewhere they're 4 bytes).
    """
    if exp is None:
        return tuple()

    if exp == "mem":
        return (prettify(exp, add_color=add_color),)

    if opcode(exp) != "data":
        res = prettify(exp, add_color=add_color, parentheses=False)
        if abi_text:
            res = raw_text(exp, res)
        return (res,)

    exp = exp[1:]

    if len(exp) == 0:
        return ()

    res = []

    idx = 0

    def word(e):
        # the number of a word of the data
        if m := match(e, ("bytes", 32, ":int:val")):
            return m.val
        return e

    while idx < len(exp):
        if selector and idx == 0 and (m := match(exp[0], ("bytes", 4, ":int:selector"))):
            # a selector: of an error, of an event... - its 8 digits when it's
            # not known (a word that small is printed in decimal)
            sel = m.selector % 2**32
            res.append(known_fname(sel, add_color) or f"{sel:#010x}")
            idx += 1
            continue

        el = exp[idx]

        first = idx == 0 or (idx == 1 and match(exp[0], ("bytes", 4, Any)))
        if abi_text and first and idx == len(exp) - 1 and (text := arr_text(el)):
            # all the data is an ABI-encoded string (after a selector)
            res.append(text)
            idx += 1
            continue

        # an ABI-encoded string: its offset, its length, the words of its
        # bytes, padded with zeroes - when it's all the data is (after a
        # selector): return 'text', revert with Error(string), 'text' (an
        # Array of it elsewhere). A string elsewhere is its bytes.
        if (
            first
            and word(el) == 32
            and idx + 1 < len(exp)
            and type(word(exp[idx + 1])) == int
        ):
            length = word(exp[idx + 1])
            size = 32 * ((length + 31) // 32)
            # its bytes, in as many parts as they come
            b, end = b"", idx + 2
            while end < len(exp) and len(b) < size:
                if (chunk := data_bytes(exp[end])) is None:
                    break
                b += chunk
                end += 1
            if 0 < length and len(b) == size and end == len(exp):
                if not any(b[length:]) and (text := pretty_text(b[:length])):
                    if not abi_text:
                        text = f"Array(len={length}, data={text})"
                    res.append(text)
                    idx = end
                    continue

        # bytes that are text, as parts of the data (hashed, say)
        end = idx
        while end < len(exp) and data_bytes(exp[end]) is not None:
            end += 1
        b = b"".join(data_bytes(e) for e in exp[idx:end])
        if len(b) >= 4 and (text := pretty_text(b)):
            if abi_text and first and end == len(exp):
                # all the data: its bytes, not the ABI-encoded string
                text = f"Bytes({len(b)}, {text})"
            res.append(text)
            idx = end
            continue

        text = pretty_element(el, add_color)
        if abi_text and first and idx == len(exp) - 1:
            text = raw_text(el, text)
        res.append(text)
        idx = idx + 1

    return tuple(res)


def raw_text(el, printed):
    """
    printed, the element el that's all the data of a return or a revert:
    Bytes(n, 'text') if it's bytes of text - a string alone is ABI-encoded.
    """
    if clean_color(printed)[:1] == "'" and (b := data_bytes(el)) is not None:
        return f"Bytes({len(b)}, {printed})"
    return printed


def with_width(el):
    """
    An element of bytes (of a data, a sha3, a return...) whose width isn't
    a word, with it: Bytes(n, el). It's in how the element is written
    (see memloc.sizeof), which rewrites for display don't keep - a mask of
    the lowest 8 bits becomes a division, uint8(x >> 8) x / 256.
    """
    if opcode(el) == "bytes" and sized(el[2]):
        # the number bytes of another width make, as n bytes: zeroes and
        # them, or their last n bytes (see memloc.resize_bytes)
        res = resize_bytes(el[2], el[1])
        return el if res is None else res
    if sized(el):
        return el
    if m := match(el, ("st", ":size", Any, Any)):
        # an access of the storage (see storage.py): as wide as it reads
        width = m.size
    else:
        width = sizeof(el)
    if width == 256 or width is None:
        return el
    if type(width) == int:
        if width % 8 or width <= 0:
            return el
        return ("bytes", width // 8, el)
    return ("bytes", ("div", width, 8), el)


def setmem_value(val, n):
    """
    The value of a write to n bytes of memory, as it's printed: a number
    (its low n bytes are written), or a list of data of n bytes - bytes of
    another width are the number they make (see memloc.keep_setmem_width).
    """
    if opcode(val) == "bytes":
        if not sized(val[2]) and val[1] != n:
            # a number
            return val[2]
        val = val[2]
    if not sized(val) or width_of(val) == 8 * n:
        return val
    res = resize_bytes(val, n)
    return val if res is None else res


def fix_widths(trace):
    """the elements of what is bytes, with their width when it isn't a word"""

    def f(exp):
        if type(exp) != tuple:
            return exp
        if (m := match(exp, ("setmem", ("range", Any, ":int:n"), ":val"))) and (
            val := setmem_value(m.val, m.n)
        ) != m.val:
            return exp[:2] + (val,)
        positions = byte_elements(exp)
        if opcode(exp) in ("call", "staticcall", "callcode", "delegatecall"):
            # but the selector: it's printed as the function it calls
            positions = positions[-1:] if positions and positions[-1] == len(exp) - 1 else ()
        if not positions:
            return exp
        res = []
        for i, e in enumerate(exp):
            if i in positions:
                e = with_width(e)
                if opcode(e) == "data" and opcode(exp) in ("data", "sha3", "arr"):
                    # (its elements, in the list it's in)
                    res.extend(e[1:])
                    continue
            res.append(e)
        return tuple(res)

    return replace_f(trace, f)


def pretty_element(el, add_color=False):
    """an element of a data: a word, or Bytes(n, value)"""
    return prettify(with_width(el), add_color=add_color, parentheses=False)
