import hashlib
import json
import logging
import os
import os.path
import sys
from typing import Optional, List

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
    cleanup_mul_1,
    colorize,
    opcode,
    cache_dir,
)
from panoramix.utils.supplement import fetch_sig

logger = logging.getLogger(__name__)

_abi = None
_func = None


def set_func_params_if_none(params):
    logger.debug("set_func_params_if_none %s - %s", params, _func)
    if "inputs" not in _func:
        res = []
        for t, n in params.values():
            res.append({"type": t, "name": n})

        _func["inputs"] = res


def set_func(hash):
    global _func
    global _abi

    assert _abi is not None
    _func = _abi[hash]


def clear_func():
    """No function of the abi printed: the calldata as it is (see
    get_param_name) - the text of a recursive internal function."""
    global _func
    _func = None


def is_dynamic(kind, components=None):
    """If a param of the type is dynamic: in the head, only its offset."""
    if kind in ("bytes", "string") or kind.endswith("[]"):
        return True

    if kind.endswith("]"):
        count = kind[kind.rindex("[") + 1 : -1]
        return not count.isdigit() or is_dynamic(kind[: kind.rindex("[")], components)

    if kind == "tuple":
        # without its components, where it ends isn't known
        return components is None or any(
            is_dynamic(c["type"], c.get("components")) for c in components
        )

    return False


# more words than any calldata holds (gas pays for them)
MAX_HEAD_WORDS = 2**16


def head_words(kind, name, components=None):
    """
    [(type, name)] of the words a param takes in the head of the calldata:
    all the elements of a static tuple or array, the offset to anything
    dynamic. (A static array of more words than calldata can hold - the
    database has bytes32[1263941234127518272] - as one.)
    """
    if is_dynamic(kind, components):
        return [(kind, name)]

    if kind.endswith("]"):
        base = kind[: kind.rindex("[")]
        count = int(kind[kind.rindex("[") + 1 : -1])
        res = []
        for idx in range(min(count, MAX_HEAD_WORDS + 1)):
            res += head_words(base, f"{name}[{idx}]", components)
            if len(res) > MAX_HEAD_WORDS:
                return [(kind, name)]
        return res

    if kind == "tuple":
        res = []
        for idx, c in enumerate(components):
            c_name = c.get("name") or f"_param{idx + 1}"
            res += head_words(c["type"], f"{name}.{c_name}", c.get("components"))
        return res

    return [(kind, name)]


def calldata_params(inputs):
    """{position in the calldata: (type, name)} for the words of the params."""
    res = {}
    loc = 4
    for p in inputs:
        for word in head_words(p["type"], p["name"], p.get("components")):
            res[loc] = word
            loc += 32

    return res


def canonical_type(kind, components=None):
    """The type as in a signature: a tuple is its components in parentheses."""
    if kind.startswith("tuple") and components is not None:
        inside = ",".join(
            canonical_type(c["type"], c.get("components")) for c in components
        )
        return f"({inside}){kind[len('tuple'):]}"

    return kind


def has_length(kind):
    """If the data a param points to starts with its length."""
    return kind in ("bytes", "string", "array") or kind.endswith("[]")


def get_param_name(cd, add_color=False, func=None):
    global _func
    loc = match(cd, ("cd", ":loc")).loc

    if _abi is None:
        return cd

    if _func is None:
        return cd

    if "inputs" not in _func:
        return cd

    params = calldata_params(_func["inputs"])
    names = {name: kind for kind, name in params.values()}

    if type(loc) != int:
        cd = cleanup_mul_1(cd)
        loc = cd[1]

        # a param is where the data it points to is (after the selector)
        if (m := match(loc, ("add", 4, ("param", ":name")))) and has_length(
            names.get(m.name, "")
        ):
            return colorize(m.name + ".length", COLOR_GREEN, add_color)

        if (m := match(loc, ("add", ":int:offset", ("cd", ":int:point_loc")))) and (
            m.point_loc in params
        ):
            kind, name = params[m.point_loc]
            if m.offset == 4 and has_length(kind):
                return colorize(name + ".length", COLOR_GREEN, add_color)

            # an element of an array of words or of offsets
            if kind.endswith("[]"):
                components = {p["name"]: p.get("components") for p in _func["inputs"]}
                element = head_words(kind[:-2], name, components.get(name))
            else:
                element = None
            if (
                m.offset >= 36
                and (m.offset - 36) % 32 == 0
                and (kind == "array" or element is not None)
                and (element is None or len(element) == 1)
            ):
                return colorize(
                    f"{name}[{(m.offset - 36) // 32}]", COLOR_GREEN, add_color
                )

        return cd

    if loc not in params:  # an unusual parameter
        return cd

    return colorize(params[loc][1], COLOR_GREEN, add_color)


def get_abi_name(hash):
    a = _abi[hash]
    if "inputs" in a:
        return "{}({})".format(
            a["name"],
            ",".join(
                canonical_type(x["type"], x.get("components")) for x in a["inputs"]
            ),
        )
    else:
        return "{}(?)".format(a["name"])


def get_func_params(hash) -> Optional[List]:
    a = _abi[hash]
    logger.debug("get_func_params for %s is %s", hash, a.get("inputs"))
    return a.get("inputs")


def set_func_params(hash, inputs):
    """The params of the function (of the abi): their names, say."""
    _abi[hash]["inputs"] = inputs


def get_func_name(hash, add_color=False):
    a = _abi[hash]
    logger.debug("get_func_name for abi %s", a)
    if "inputs" in a:
        return "{}({})".format(
            a["name"],
            ", ".join(
                [
                    # (a tuple as the types it's made of, see canonical_type)
                    canonical_type(x["type"], x.get("components"))
                    + " "
                    + colorize(
                        x["name"],
                        COLOR_GREEN,
                        add_color,
                    )
                    for x in a["inputs"]
                ]
            ),
        )
    else:
        return "{}(?)".format(a["name"])


def fix_input_names(inputs: List[dict]):
    for i, input in enumerate(inputs):
        if not input["name"]:
            input["name"] = f"_param{i+1}"

    return inputs


def make_abi(hash_targets):
    global _abi

    hashes = list(hash_targets.keys())

    result = {}

    for h, target in hash_targets.items():
        res = {
            "name": "unknown" + h[2:],
        }

        if h.startswith("0x"):
            sig = fetch_sig(h)
            if sig:
                res = {
                    "name": sig["name"],
                    "inputs": fix_input_names(sig["inputs"]),
                }
        else:  # assuming index is a name - e.g. for _fallback()
            res = {
                "name": h,
            }

        res["target"] = target

        result[h] = res

    _abi = result
