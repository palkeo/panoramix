from panoramix.utils.helpers import clear_caches, is_array, opcode

"""

    for a given expression, returns variants of it with each value being 0 or 2^256-1

    e.g. for ('ADD', ('mem', 64), ('mem', 256))
         returns
            ('ADD', 0, 0),
            ('ADD', 0, MAX_number),
            ('ADD', MAX_number, 0),
            ('ADD', MAX_number, MAX_number)

"""


MAX_number = 2**230 - 1
MAX_number2 = 2**230 - 1

# Whether mem[64] is the free memory pointer of solidity, which it sets to
# 0x80 (0x60 before 0.4.22) before anything else: then it's at least 0x60
# (see Loader.load_binary). Vyper's mem[64] is a word like another, a length
# say.
FREE_MEMORY_POINTER = True


def set_free_memory_pointer(on):
    global FREE_MEMORY_POINTER
    if on != FREE_MEMORY_POINTER:
        FREE_MEMORY_POINTER = on
        # (what was decided with the other assumption)
        clear_caches()


def variants(exp):
    var = extract_variables(exp)
    for p in possibilities(list(var)):
        yield replace_dict(exp, p)


"""

    implementation

"""


def extract_variables(exp):
    if type(exp) == int:
        return set()

    if opcode(exp) in (
        "var",
        "mem",
        "cd",
        "storage",
        "call.data",
        "sha3",
        "calldatasize",
    ) or is_array(opcode(exp)):
        return set([exp])

    if type(exp) == str and exp in (
        "x",
        "y",
        "z",
        "sth",
        "unknown",
        "undefined",
        "callvalue",
        "number",
        "timestamp",
        "address",
    ):
        return set([exp])

    if type(exp) == str and exp != "data" and "data" in exp:
        return set([exp])

    if type(exp) != tuple:
        return set([exp])

    res = set()
    for e in exp[1:]:
        res = res.union(extract_variables(e))

    return res


def possibilities(var):
    if len(var) > 0:
        current = var[0]
        if len(var) == 1:
            yield {current: MAX_number}
            yield {current: MAX_number2}

            if current == ("mem", ("range", 64, 32)) and FREE_MEMORY_POINTER:
                yield {current: 96}
            else:
                # (calldatasize too: a call has 0 to 3 bytes of data, the
                # sweeper contract's fallback say)
                yield {current: 0}

        else:
            for p in possibilities(var[1:]):
                p[current] = MAX_number
                yield p
                p[current] = MAX_number2
                yield p

                if current == ("mem", ("range", 64, 32)) and FREE_MEMORY_POINTER:
                    p[current] = 96
                else:
                    p[current] = 0
                yield p


def replace_dict(exp, dic):
    # All the variables at once: one can contain another (mem[_1] and _1),
    # replacing them one after the other would depend on their order.
    for idx, val in dic.items():
        if exp == idx:
            return val

    if type(exp) != tuple:
        return exp

    return tuple(replace_dict(e, dic) for e in exp)
