import logging

import panoramix.folder as folder
import panoramix.safemath as safemath
import panoramix.storage as storage
from panoramix.postprocess import short_circuits
from panoramix.matcher import match
from panoramix.prettify import (
    fix_widths,
    pprint_ast,
    pprint_trace,
    pretty_stor,
    set_names,
)
from panoramix.utils.helpers import (
    COLOR_GREEN,
    ENDC,
    opcode,
    replace_f,
    replace_lines,
    to_exp2,
    tuplify,
)

from panoramix.function import Function, InternalFunction

logger = logging.getLogger(__name__)


def deserialize(trace):
    res = []
    for line in trace:
        line_t = tuple(line)

        if opcode(line_t) == "while":
            _, cond, path, lid, setvars = line_t
            cond = tuplify(cond)
            setvars = tuplify(setvars)
            assert type(lid) == str

            path = deserialize(path)
            res.append(("while", cond, path, lid, setvars))

        elif opcode(line_t) == "if":
            _, cond, if_true, if_false = line_t
            cond = tuplify(cond)
            if_true = deserialize(if_true)
            if_false = deserialize(if_false)
            res.append(("if", cond, if_true, if_false))

        else:
            res.append(tuplify(line))

    return res


class Contract:
    def __init__(self, functions, problems, code=None, internal=()):
        self.problems = problems
        self.functions = []
        for func in functions.values():
            self.functions.append(func)
        # the recursive internal functions, decompiled apart (see
        # vm.VM.internal), by their entry
        self.internal = list(internal)

        self.code = code
        self.lang = "solidity"
        self.stor_defs = {}
        # the functions of the contract's SafeMath (see safemath.py)
        self.safemath = []

    def json(self) -> dict:
        res = {
            "problems": self.problems,
            "stor_defs": self.stor_defs,
            "functions": [f.serialize() for f in self.functions],
        }
        if self.safemath:
            res["safemath"] = "\n".join(safemath.pretty_defs(self.safemath))
        if self.internal:
            res["internal"] = [f.serialize() for f in self.internal]
        return res

    def load(self, data):
        self.problems = data["problems"]
        self.functions = []
        self.stor_defs = data["stor_defs"] if "stor_defs" in data else {}

        for func in data["functions"]:
            self.functions.append(
                Function(hash=func["hash"], trace=deserialize(func["trace"]))
            )

        return self

    def postprocess(self):
        try:
            self.lang, self.stor_defs = storage.rewrite_functions(
                self.functions + self.internal, self.code
            )
        except Exception:
            # this is critical, because it causes full contract to display very
            # badly, and cannot be limited in scope to just one affected function
            logger.exception("Storage postprocessing failed. This is very bad!")
            self.stor_defs = []
            # the accesses as the slots they are
            storage.rewrite_raw(self.functions + self.internal)

        # (not the names of variables nor of params, see prettify.set_names)
        storage_names = [d[1] for d in self.stor_defs]
        set_names(storage=storage_names)

        for func in self.functions:
            func.rename_params(storage_names)

            def replace_names(exp):
                if (
                    m := match(exp, ("cd", ":int:idx"))
                ) and m.idx in func.inferred_params:
                    return ("param", func.inferred_params[m.idx][1])
                return exp

            func.trace = replace_f(func.trace, replace_names)

        # const list, sort by putting all-caps consts at the end - looks way better this way
        self.consts = [
            f for f in self.functions if f.const and f.name.upper() != f.name
        ] + [f for f in self.functions if f.const and f.name.upper() == f.name]

        self.make_asts()

        try:
            self.safemath = safemath.rewrite(self.functions + self.internal)
        except Exception:
            logger.exception("SafeMath failed: the arithmetic stays as it is.")
            self.safemath = []

    def make_asts(self):

        for func in self.functions + self.internal:
            func.ast = self.make_ast(func.trace)

    def make_ast(self, trace):
        trace = short_circuits(trace)
        trace = folder.fold(trace)
        trace = fix_widths(trace)

        def other_1(exp):
            if (
                (
                    m := match(
                        exp, ("mask_shl", ":int:size", ":n_size", ":size_n", ":str:val")
                    )
                )
                and 256 - m.size == m.n_size
                and m.size - 256 == m.size_n
                and m.size + 16 == len(m.val) * 8
                and len(m.val) > 0
                and m.val[0] == m.val[-1] == "'"
            ):  # +16 because '' in strings
                return m.val
            else:
                return exp

        def other_2(exp):
            if (
                m := match(exp, ("if", ("eq", ":a", ":b"), ":if_true"))
            ) and m.if_true == [("return", ("eq", m.a, m.b))]:
                return ("if", ("eq", m.a, m.b), [("return", ("bool", 1))])

            elif (m := match(exp, ("mask_shl", 160, 0, 0, ":str:e"))) and m.e in (
                "address",
                "coinbase",
                "caller",
                "origin",
            ):
                return m.e

            elif (
                (
                    m := match(
                        exp, ("mask_shl", ":int:size", ":int:off", ":int:m_off", ":e")
                    )
                )
                and m.m_off == -m.off
                and m.off in range(1, 9)
                and m.size + m.off in [8, 16, 32, 64, 128, 256]
            ):
                if opcode(m.e) == "st" and type(m.e[1]) == int and m.e[1] <= m.size + m.off:
                    # a storage access no wider than the mask's top
                    return ("div", m.e, 2**m.off)
                return ("div", ("mask", m.size + m.off, 0, m.e), 2**m.off)

            else:
                return exp

        trace = replace_f(trace, other_1)
        trace = replace_f(trace, other_2)
        return trace
