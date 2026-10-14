import collections
import logging
from copy import copy
from time import gmtime, strftime

import panoramix.core.arithmetic as arithmetic
from panoramix.core.algebra import (
    _max_op,
    add_op,
    apply_mask,
    apply_mask_to_storage,
    bits,
    calc_max,
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
from panoramix.core.arithmetic import is_zero, to_real_int
from panoramix.core.masks import get_bit, to_mask, to_neg_mask
from panoramix.core.memloc import (
    apply_mask_to_range,
    low_zero_bits,
    memloc_overwrite,
    range_overlaps,
    sizeof,
    split_setmem,
    split_store,
    splits_mem,
)
from panoramix.matcher import Any, match
from panoramix.prettify import pformat_trace, pprint_trace
from panoramix.utils.helpers import (
    contains,
    find_f_list,
    find_f_set,
    is_array,
    opcode,
    replace,
    replace_f,
    rewrite_trace,
    rewrite_trace_full,
    rewrite_trace_multiline,
    to_exp2,
    walk_trace,
)


"""

    Some nasty last-minute hacks and heurestics I wrote to finally get the April release to production.

    One of the very few places in Panoramix that are blatantly mathematically incorrect, but
    help to make a ton of contracts way more readable (and - in practice - being always valid)

"""


def postprocess_exp(exp):
    """
    An array in a list of data - its offset, then after the other words its
    length and content - as Array(len=..., data=...), that is ABI-encoded:
    its content padded with zeroes to whole words (see OUTPUT.md). Where the
    data has that padding only: its content is whole words, or is as many
    bytes as its length followed by the zeroes up to them. DSNote's log of
    msg.data, `log ... call.value, 64, calldata.size, call.data[0 len
    calldata.size]`, has none.
    """
    if opcode(exp) != "data":
        return exp

    terms = exp[1:]
    concrete = [t for t in terms if type(t) == int and t % 32 == 0]
    if len(concrete) != 1:
        return exp

    loc = concrete[0] // 32
    if not (
        loc + 1 < len(terms)
        and loc > terms.index(concrete[0])
        # the offset is one of words up to the length
        and all(sizeof(t) == 256 for t in terms[: loc + 1])
    ):
        return exp

    content = abi_content(terms[loc], terms[loc + 1 :])
    if content is None:
        return exp

    arr = ("arr", terms[loc]) + content
    t2 = tuple([arr if t == loc * 32 else t for t in terms[:loc]])
    return ("data",) + t2


def abi_content(length, content):
    """
    What an Array(len=length, data=...) of the bytes of content is printed
    with - as they are if they're whole words (padding them adds nothing),
    without the zeroes after its `length` bytes up to whole words - None if
    they're neither.
    """
    content = tuple(content)
    size = byte_size(content)

    data = content
    while data and (data[-1] == 0 or match(data[-1], ("bytes", Any, 0))):
        data = data[:-1]
    words = ("mask_shl", 251, 5, 0, add_op(31, length))
    if (
        size is not None
        and sub_op(byte_size(data), length) == 0
        and sub_op(size, words) == 0
    ):
        return data

    if size is not None and low_zero_bits(size) >= 5:
        return content

    return None


def byte_size(terms):
    """how many bytes the elements of a data are, None if that's not known"""
    res = 0
    for t in terms:
        if m := match(t, ("bytes", ":n", Any)):
            n = m.n
        elif (m := match(t, (":op", Any, ":n"))) and is_array(m.op):
            n = m.n
        elif m := match(t, ("mem", ("range", Any, ":n"))):
            n = m.n
        elif type(w := sizeof(t)) is int and w % 8 == 0:
            n = w // 8
        else:
            return None
        res = add_op(res, n)
    return res


def postprocess_trace(line):
    """
    let's find all the stuff like

     if (some_len % 32) == 0:
        return Array(some_len, some_stuff)
     else:
        mem[...] = leftover
        return Array(some_len, some_stuff, leftover)

    and replace it with just return Array(some_len, some_stuff)

    in theory this is incorrect, because perhaps program does something totally different
    in the one branch, andd something entirely different in another.
    but this cleans up tremendous amounts of output, and didn't find a counterexample yet.
    """

    #    if line ~ ('setmem', ('range', :s, ('mask_shl', 251, 5, 0, ('add', 31, ('cd', ('add', 4, :param))))), ('data', ('call.data', ('add', 36, param), ('cd', ('add', 4, param))), ('mem', ...))):
    #        lin = ('setmem', ('range', s, ('cd', ('add', 4, param))), ('call.data', ('add', 36, param), ('cd', ('add', 4, param))))
    #        return [lin]

    if m := match(
        line, ("if", ("iszero", ("storage", 5, 0, ":l")), ":if_true", ":if_false")
    ):
        l, if_true, if_false = m.l, m.if_true, m.if_false

        def find_arr_l(exp):
            if match(exp, ("arr", ("storage", 256, 0, l), ...)):
                return [exp]

        true_arr = find_f_list(if_true, find_arr_l)
        false_arr = find_f_list(if_false, find_arr_l)

        if len(true_arr) > 0 and len(true_arr) == len(false_arr):
            return if_true

    if m := match(
        line, ("if", ("iszero", ("mask_shl", 5, 0, 0, ":l")), ":if_true", ":if_false")
    ):
        l, if_true, if_false = m.l, m.if_true, m.if_false

        def find_arr_l(exp):
            if match(exp, ("arr", l, ...)):
                return [exp]

        true_arr = find_f_list(if_true, find_arr_l)
        false_arr = find_f_list(if_false, find_arr_l)

        if len(true_arr) > 0 and len(true_arr) == len(false_arr):
            return if_true

    """
        When writing strings to storage, there are usually three cases - when string is 0,
        when string is < 31 (special format that takes just one storage slot), and when string >= 32.

        e.g. 0xf97187f566eC6374cB08470CCe593fF0Dd36d8A9, baseURI
             0xFcD0d8E3ae34922A2921f7E7065172e5317f8ad8, name

        When it's longer than 31 bytes, it isn't empty: the check that it is
        can go.
    """

    if m := match(line, ("if", ("lt", 31, ":some_len"), ":if_true", ":if_false")):
        some_len, if_true, if_false = m.some_len, m.if_true, m.if_false
        if len(if_true) >= 2:
            # the `if iszero(len)` may be followed by more code when its
            # branches got merged (see vm.merge_branches)
            first, second, *rest = if_true
            if (
                opcode(first) == "store"
                and contains(first, some_len)
                and (
                    m := match(
                        second,
                        ("if", ("iszero", some_len), ":deep_true", ":deep_false"),
                    )
                )
            ):
                if_true = [first] + m.deep_false + rest
                return [("if", ("lt", 31, some_len), if_true, if_false)]

    return [line]


def rewrite_memcpy(lines):  # 2
    assert len(lines) == 2
    l1 = lines[0]
    l2 = lines[1]

    if m := match(
        l1,
        (
            "setmem",
            (
                "range",
                ":s",
                ("mask_shl", 251, 5, 0, ("add", 31, ("cd", ("add", 4, ":param")))),
            ),
            (
                "data",
                ("call.data", ("add", 36, ":param"), ("cd", ("add", 4, ":param"))),
                ("mem", ...),
            ),
        ),
    ):
        return (
            "setmem",
            ("range", m.s, ("cd", ("add", 4, m.param))),
            ("call.data", ("add", 36, m.param), ("cd", ("add", 4, m.param))),
        )


# (setmem (range (add 128 (mask_shl 251 5 0 (add 31 (cd (add 4 (cd 68)))))) (mask_shl 251 5 0 (add 31 (cd (add 4 (cd 68)))))) (data (call.data (add 36 (cd 68)) (cd (add 4 (cd 68)))) (mem (range (add 128 (cd (add 4 (cd 68)))) (add (mask_shl 251 5 0 (add 31 (cd (add 4 (cd 68))))) (mul -1 (cd (add 4 (cd 68)))))))))
#        (if (iszero (mask_shl 5 0 0 (cd (add 4 (cd 68))))) (t


"""

    test case for above:

    (store 256 0 0 (add 1 (mask_shl 255 0 1 (cd (add 4 (cd 36))))))
        (while (gt (add 160 (mask_shl 251 5 0 (add 31 (cd (add 4 (cd 4))))) (cd (add 4 (cd 36)))) (var 0)) (t
          (store 256 0 (add (var 1) (sha3 0)) (mem (range (var 0) 32)))
          (continue id8785 ((setvar 1 (add 1 (var 1))) (setvar 0 (add 32 (var 0)))))
        ) id8785 [('setvar', 1, 0), ('setvar', 0, ('add', 160, ('mask_shl', 251, 5, 0, ('add', 31, ('cd', ('add', 4, ('cd', 4)))))))])
        (while (gt (mask_shl 251 5 -5 (add 31 (storage 256 0 (length (loc 0))))) (var 0)) (t
          (store 256 0 (add (var 0) (sha3 0)) 0)
          (continue id3054 ((setvar 0 (add 1 (var 0)))))
        ) id3054 [('setvar', 0, ('mask_shl', 251, 0, -5, ('add', 31, ('cd', ('add', 4, ('cd', 36))))))])
        (store 256 0 1 (add 1 (mask_shl 255 0 1 (cd (add 4 (cd 4))))))
        (while (gt (add 128 (cd (add 4 (cd 4)))) (var 0)) (t
          (store 256 0 (add (var 1) (sha3 1)) (mem (range (var 0) 32)))
          (continue id2702 ((setvar 1 (add 1 (var 1))) (setvar 0 (add 32 (var 0)))))
        ) id2702 [('setvar', 1, 0), ('setvar', 0, 128)])
        (while (gt (mask_shl 251 5 -5 (add 31 (storage 256 0 (length (loc 1))))) (var 0)) (t
          (store 256 0 (add (var 0) (sha3 1)) 0)
          (continue id6799 ((setvar 0 (add 1 (var 0)))))
        ) id6799 [('setvar', 0, ('mask_shl', 251, 0, -5, ('add', 31, ('cd', ('add', 4, ('cd', 4))))))])

"""
