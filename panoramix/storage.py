"""
The storage of a contract, as its functions use it.

Every access of the functions - ("storage", size, off, slot) read,
("store", size, off, slot, value) written - is at a path from a variable of
the contract: its slot, an element of a mapping or of an array, a member of a
struct... (see parse). What the variables are comes from all the paths (see
Layout), their names from the getters. Each access is then printed as its
path - name[key], name[i].field_N, name.length - so that it reads, by the
rules of OUTPUT.md, as the slot and the bits it is. That is checked for each
one, on random values of what the slot is computed from (see Check): an
access that doesn't read so is printed raw, stor[slot], which always does.

The printed forms (see prettify.pretty_st):
    ("st", size, loc): the `size` bits at loc (a cast when it's less than
                       what loc has)
    loc: ("sv", name)              a variable
         ("si", base, key)         base[key], of a mapping, an array, a bytes
         ("sl", base)              base.length: the slot of an array
         ("sbl", base)             base.length: the length of a bytes
         ("sf", base, n)           base.field_n: from the bit n of base on
         ("sr", slot)              stor[slot]
"""

import bisect
import hashlib
import logging
import math
from functools import lru_cache

from panoramix.core.algebra import add_op, mul_op
from panoramix.core.memloc import sized, sizeof, value_bits
from panoramix.matcher import Any, match
from panoramix.utils.helpers import find_f_list, opcode, replace_f

logger = logging.getLogger(__name__)

M = 2**256 - 1


def keccak(b):
    from eth_hash.auto import keccak as k

    return int.from_bytes(k(b), "big")


def w32(n):
    return (n % 2**256).to_bytes(32, "big")


"""
    the slots that are hashes, as the compiler leaves them when they're known
"""

# keccak(n): where the data of the array (or bytes) at the slot n starts
DATA_SLOTS = 4096
# keccak(key . n): a mapping at the slot n, of a small key
MAP_KEYS = 16
MAP_SLOTS = 64
# what's added to a hash to reach a member, an element
MAX_DELTA = 2**16


@lru_cache(maxsize=None)
def hash_table(lang):
    """(hash, path) sorted by hash, the hashes a compiler writes as numbers"""
    table = [(keccak(w32(n)), ("data", ("var", n))) for n in range(DATA_SLOTS)]
    for n in range(MAP_SLOTS):
        for k in range(MAP_KEYS):
            data = w32(n) + w32(k) if lang == "vyper" else w32(k) + w32(n)
            table.append((keccak(data), ("map", ("var", n), k, "word")))
    table.sort()
    return table


def hash_constant(c, lang):
    """(path, delta) when the number c is delta after a hash of the table"""
    table = hash_table(lang)
    c %= 2**256
    i = bisect.bisect_right(table, (c, ("~",))) - 1
    if i >= 0 and 0 <= c - table[i][0] < MAX_DELTA:
        return table[i][1], c - table[i][0]
    return None


"""
    parsing the slots
"""


def word(e):
    """a word of a data, as its value"""
    if m := match(e, ("bytes", 32, ":v")):
        if not sized(m.v):
            return m.v
    return e


def is_word(e):
    """whether e, as an element of a data, is a word"""
    if opcode(e) == "bytes":
        return e[1] == 32 and not sized(e[2])
    if sized(e):
        return False
    try:
        return sizeof(e) == 256
    except AssertionError:
        return False


def parse(idx, lang):
    """
    The path of the slot idx:
        ("var", n)                  the slot n
        ("map", p, key, kind)       keccak(key . slot(p)) - vyper: slot . key;
                                    kind "word", or "bytes" when key is data
        ("data", p)                 keccak(slot(p)): the data of an array
        ("off", p, e)               slot(p) + e
        ("blen", p)                 the length of the bytes at slot(p)
        ("raw", idx)                none of those
    """
    idx = word(idx)

    if type(idx) == int:
        idx %= 2**256
        if h := hash_constant(idx, lang):
            path, delta = h
            return ("off", path, delta) if delta else path
        return ("var", idx)

    if m := match(idx, ("loc", ":int:n")):
        return ("var", m.n)

    if m := match(idx, ("length", ":key")):
        # the length of the bytes (or string) at the slot key (see
        # simplify.replace_bytes_or_string_length)
        inner = parse(m.key, lang)
        return ("raw", idx) if inner[0] == "raw" else ("blen", inner)

    op = opcode(idx)

    if op == "sha3":
        parts = idx[1][1:] if len(idx) == 2 and opcode(idx[1]) == "data" else idx[1:]
        if not parts:
            return ("raw", idx)
        if len(parts) == 1 and is_word(parts[0]):
            inner = parse(parts[0], lang)
            return ("raw", idx) if inner[0] == "raw" else ("data", inner)
        if lang == "vyper":
            slot, key = parts[0], parts[1:]
        else:
            slot, key = parts[-1], parts[:-1]
        if not is_word(slot):
            return ("raw", idx)
        inner = parse(slot, lang)
        if inner[0] == "raw":
            return ("raw", idx)
        if len(key) == 1 and is_word(key[0]):
            return ("map", inner, word(key[0]), "word")
        return ("map", inner, ("data",) + tuple(key), "bytes")

    if op == "add":
        terms = []
        for t in idx[1:]:
            # (a sum of sums is one)
            terms += list(t[1:]) if opcode(t) == "add" else [t]
        # the base: a hash, or a number that is one
        bases = [
            k
            for k, t in enumerate(terms)
            if opcode(t) == "sha3" or (type(t) == int and hash_constant(t, lang))
        ]
        if len(bases) == 1:
            base = terms.pop(bases[0])
            if type(base) == int:
                path, delta = hash_constant(base, lang)
                if delta:
                    terms.append(delta)
            else:
                path = parse(base, lang)
            if path[0] == "raw":
                return ("raw", idx)
            e = add_op(*terms) if len(terms) > 1 else terms[0]
            return ("off", path, e)

        if not bases:
            consts = [k for k, t in enumerate(terms) if type(t) == int]
            if len(consts) == 1:
                c = terms.pop(consts[0]) % 2**256
                e = add_op(*terms) if len(terms) > 1 else terms[0]
                return ("off", ("var", c), e)

        return ("raw", idx)

    if from_calldata(idx):
        # a slot computed from a param: the element of an array at 0 (as
        # vyper puts them)
        return ("off", ("var", 0), idx)

    return ("raw", idx)


def from_calldata(e):
    """whether e is a param, or masks and shifts of one"""
    if opcode(e) in ("cd", "param"):
        return True
    if opcode(e) == "mask_shl":
        return from_calldata(e[4])
    return False


def steps(path):
    """
    (root slot, the steps from it) of a path, None if it's raw:
        ("map", key, kind), ("data",), ("off", e), ("blen",)
    """
    kind = path[0]
    if kind == "var":
        return path[1], []
    if kind == "raw":
        return None
    inner = steps(path[1])
    if inner is None:
        return None
    root, st = inner
    if kind == "map":
        return root, st + [("map", path[2], path[3])]
    if kind == "data":
        return root, st + [("data",)]
    if kind == "blen":
        return root, st + [("blen",)]
    if kind == "off":
        e = path[2]
        if not st and type(e) == int:
            # a slot after the root is another variable
            return (root + e) % 2**256, []
        if st and st[-1][0] == "off":
            # offsets one after the other are one
            return root, st[:-1] + [("off", add_op(st[-1][1], e))]
        return root, st + [("off", e)]
    raise ValueError(path)


"""
    elements: the index and the member an offset from the start of an array is
"""


def log2(n):
    return n.bit_length() - 1 if n > 0 and n & (n - 1) == 0 else None


def linear(e):
    """(index, stride, constant) with e == stride * index + constant"""
    if type(e) == int:
        return 0, 0, e
    if opcode(e) == "add":
        consts = [t for t in e[1:] if type(t) == int]
        rest = [t for t in e[1:] if type(t) != int]
        c = sum(consts)
        if len(rest) == 1:
            i, s, c2 = linear(rest[0])
            return i, s, c + c2
        return add_op(*rest), 1, c
    if (m := match(e, ("mul", ":int:s", ":i"))) and m.s > 0:
        return m.i, m.s, 0
    if (m := match(e, ("mask_shl", ":int:size", 0, ":int:shl", ":i"))) and (
        m.size + m.shl == 256 and 0 < m.shl < 256
    ):
        # i << shl, the bits shifted out lost: 2**shl * i
        return m.i, 2**m.shl, 0
    return e, 1, 0


def packed(e, off, width):
    """
    The index of the element of `width` bits at slot + e, bits off, of an
    array packed `256 / width` in a slot - or None if it isn't one.
    """
    if type(width) != int or not 0 < width <= 256:
        # (an access of a size computed at runtime, say)
        return None
    per = 256 // width
    lp, lw = log2(per), log2(width)
    if lp is None or lw is None or per < 2:
        return None

    if type(off) == int:
        if type(e) == int and off % width == 0 and e >= 0:
            return e * per + off // width
        return None

    if not (m := match(off, ("mask_shl", ":int:a", 0, lw, ":x"))):
        return None
    if m.a == lp and e == ("mask_shl", 256 - lp, lp, -lp, m.x):
        return m.x
    if m.a < lp and e == 0:
        return ("mask_shl", m.a, 0, 0, m.x)
    return None


"""
    the check: what a printed access reads, on random values
"""


class World:
    """random values for what a slot is computed from, the same for the same"""

    def __init__(self, seed):
        self.seed = seed
        self.cache = {}

    def sym(self, e):
        k = repr(e)
        if k not in self.cache:
            h = hashlib.blake2b(f"{self.seed}:{k}".encode(), digest_size=32).digest()
            v = int.from_bytes(h, "big")
            # small values too: indexes, lengths
            if self.seed % 2:
                v %= 2**16
            self.cache[k] = v
        return self.cache[k]


class Unknown(Exception):
    pass


def ev(e, w):
    """the value of e in the world w"""
    if type(e) == bool:
        return int(e)
    if type(e) == int:
        return e % 2**256
    if type(e) != tuple or not e:
        return w.sym(e)
    op = e[0]
    if op == "bytes" and e[1] == 32:
        return ev(e[2], w)
    if op == "add":
        return sum(ev(t, w) for t in e[1:]) % 2**256
    if op == "mul":
        r = 1
        for t in e[1:]:
            r = r * ev(t, w) % 2**256
        return r
    if op == "div" and len(e) == 3:
        b = ev(e[2], w)
        return ev(e[1], w) // b if b else 0
    if op == "mod" and len(e) == 3:
        b = ev(e[2], w)
        return ev(e[1], w) % b if b else 0
    if op == "mask_shl":
        size, off, shl = (signed(ev(x, w)) for x in e[1:4])
        if not (0 <= size <= 256 and -256 <= off <= 256 and -512 <= shl <= 512):
            raise Unknown()
        v = ev(e[4], w)
        if off >= 0:
            v &= ((1 << size) - 1) << off
        else:
            v &= (1 << max(size + off, 0)) - 1
        v = v << shl if shl >= 0 else v >> -shl
        return v % 2**256
    if op == "sha3":
        parts = e[1][1:] if len(e) == 2 and opcode(e[1]) == "data" else e[1:]
        return keccak(b"".join(evb(p, w) for p in parts))
    return w.sym(e)


def evb(e, w):
    """the bytes of e as an element of a data"""
    if opcode(e) == "bytes" and type(e[1]) == int and 0 < e[1] <= 32:
        return (ev(e[2], w) % 2 ** (8 * e[1])).to_bytes(e[1], "big")
    if opcode(e) == "data":
        return b"".join(evb(t, w) for t in e[1:])
    if sized(e):
        # a range: its bytes, whatever they are, the same for the same
        return hashlib.blake2b(f"{w.seed}:{e!r}".encode(), digest_size=40).digest()
    width = sizeof(e)
    if type(width) != int or width % 8 or not 0 < width <= 256:
        raise Unknown()
    return (ev(e, w) % 2**width).to_bytes(width // 8, "big")


def signed(v):
    return v - 2**256 if v >= 2**255 else v


WORLDS = [World(s) for s in range(4)]


"""
    the types: what's at a slot, from how the functions reach it

        ("value", width)            a number of width bits at the bit 0
        ("struct", slots)           values read by their bits (.field_N),
                                    over `slots` slots (None: not known)
        ("mapping", t)              t at keccak(key . slot)
        ("array", t, stride)        its length at the slot, the element i at
                                    keccak(slot) + i * stride - packed
                                    256 / width in a slot for a narrow value
        ("fixed", t, stride)        the element i at slot + i * stride
        ("bytes",)                  its word at the slot, its data words at
                                    keccak(slot) + i
"""


def elem_width(t):
    """the bits an element of type t takes in its slot (a packed array's)"""
    if t[0] == "value" and t[1] < 256 and 256 % t[1] == 0 and t[1] >= 8:
        return t[1]
    return 256


def is_bytes_slot(fields):
    """
    The fields (size, off) read from the slot of a bytes or a string: its
    lowest bit tells a long one from a short one, the length being the rest.
    """
    return (1, 0) in fields and bool(fields & {(255, 1), (7, 1), (248, 8)})


def typeof(accs):
    """
    The type of what's at a slot, from its accesses (steps from it, size,
    off): the most of them, those that don't fit it are printed raw.
    """
    kinds = {}
    for st, size, off in accs:
        if not st:
            continue
        s = st[0]
        if s[0] == "off":
            k = "member" if type(s[1]) == int else "fixed"
        elif s[0] == "blen":
            k = "data"
        else:
            k = s[0]
        kinds[k] = kinds.get(k, 0) + 1

    direct = {(size, off) for st, size, off in accs if not st}

    if not kinds:
        if all(off == 0 for size, off in direct) and all(
            type(size) == int for size, off in direct
        ):
            return ("value", max(size for size, off in direct))
        widths = {size for size, off in direct if type(off) != int}
        if len(widths) == 1 and elem_width(("value", min(widths))) < 256:
            # the elements of a packed array, at bits computed at runtime
            return ("fixed", ("value", widths.pop()), 1)
        return ("struct", 1)

    kind = max(kinds, key=lambda k: (kinds[k], k))
    if set(kinds) == {"fixed", "member"}:
        # elements at the slot plus an index: some at a constant one (a
        # Vyper string's length, its first word)
        kind = "fixed"

    if kind == "map":
        inner = [
            (st[1:], size, off) for st, size, off in accs if st and st[0][0] == "map"
        ]
        return ("mapping", typeof(inner))

    if kind == "member":
        members = [
            st[0][1]
            for st, size, off in accs
            if st and st[0][0] == "off" and type(st[0][1]) == int
        ]
        return ("struct", max(members) + 1 if min(members) >= 0 else None)

    if kind == "data":
        blen = any(st and st[0][0] == "blen" for st, size, off in accs)
        if blen or is_bytes_slot(direct):
            return ("bytes",)
        elems = [
            (st[1:], size, off) for st, size, off in accs if st and st[0][0] == "data"
        ]
        return ("array",) + array_type(elems)

    # a fixed array: its elements at the slot plus their index
    elems = [(st, size, off) for st, size, off in accs if st and st[0][0] == "off"]
    return ("fixed",) + array_type(elems)


def element_offset(st):
    """(offset from the start of the elements, the steps after it)"""
    if st and st[0][0] == "off":
        return st[0][1], st[1:]
    return 0, st


def array_type(elems):
    """(element type, stride) of an array, from the accesses of its elements"""
    widths = set()
    for st, size, off in elems:
        e, rest = element_offset(st)
        if (
            not rest
            and type(size) == int
            and 8 <= size < 256
            and packed(e, off, size) is not None
        ):
            widths.add(size)
    if len(widths) == 1 and all(
        not element_offset(st)[1]
        and packed(element_offset(st)[0], off, size) is not None
        for st, size, off in elems
    ):
        return ("value", widths.pop()), 1

    strides = []
    for st, size, off in elems:
        e, rest = element_offset(st)
        i, s, c = linear(e)
        if s:
            strides.append(s)
    stride = 1
    if strides:
        stride = strides[0]
        for s in strides[1:]:
            stride = math.gcd(stride, s)

    inner = []
    for st, size, off in elems:
        e, rest = element_offset(st)
        i, s, c = linear(e)
        member = c % stride
        inner.append((([("off", member)] if member else []) + rest, size, off))

    t = typeof(inner)
    if stride > 1 and t[0] != "struct":
        t = ("struct", stride)
    elif t[0] == "struct":
        t = ("struct", stride)
    return t, stride


"""
    the printed accesses
"""


class Fail(Exception):
    """an access that can't be printed as its path"""


def value_width(t):
    """what a loc of type t is, read whole: the bits of a value, else a word"""
    return t[1] if t[0] == "value" else 256


def form(base, t, st, size, off):
    """
    The printed access of `size` bits at `off` of what's at the steps st from
    base, a loc of type t.
    """
    if not st:
        if t[0] == "value":
            width = t[1]
            if off == 0 and type(size) == int and size <= width:
                return ("st", size, base, width)
            if (
                type(off) == int
                and type(size) == int
                and 0 < off
                and off + size <= width
            ):
                # some bits of the value
                return ("mask_shl", size, off, -off, ("st", width, base, width))
        if t[0] == "array" and off == 0:
            # the slot of an array is its length
            return ("st", size, ("sl", base), 256)
        if t[0] == "bytes" and off == 0:
            # the slot of a bytes, whatever it holds (the length, the data)
            return ("st", size, base, 256)
        if t[0] == "fixed":
            # elements of a packed array in the slot itself
            return element(base, t, [], size, off)
        # (the first word of a struct of more slots is its .field_0)
        return field(base, 0, size, off, whole=not (t[0] == "struct" and t[1] != 1))

    s = st[0]
    if s[0] == "map":
        if t[0] != "mapping":
            raise Fail()
        return form(("si", base, s[1]), t[1], st[1:], size, off)

    if s[0] == "blen":
        if t[0] != "bytes" or st[1:]:
            raise Fail()
        loc = ("st", 256, ("sbl", base), 256)
        if off == 0 and size == 256:
            return loc
        return ("mask_shl", size, off, mul_op(-1, off), loc)

    if s[0] == "data":
        if t[0] == "bytes":
            e, rest = element_offset(st[1:])
            if rest:
                raise Fail()
            return field(("si", base, e), 0, size, off)
        if t[0] != "array":
            raise Fail()
        return element(base, t, st[1:], size, off)

    if s[0] == "off":
        if t[0] == "fixed":
            return element(base, t, st, size, off)
        if type(s[1]) == int:
            if t[0] != "struct" or st[1:]:
                raise Fail()
            return field(base, 256 * s[1], size, off)
        raise Fail()

    raise Fail()


def field(base, bit, size, off, whole=True):
    """
    The bits [off, off + size) of the slot bit / 256 from base - the whole
    word at the bit 0 as base itself, when whole.
    """
    if type(off) != int or type(size) != int:
        raise Fail()
    n = bit + off
    if n == 0 and size == 256 and whole:
        return ("st", 256, base, 256)
    return ("st", size, ("sf", base, n), 256 - n % 256)


def element(base, t, st, size, off):
    """an access of an element of the array base (st: from its start)"""
    _, et, stride = t
    e, rest = element_offset(st)
    width = elem_width(et)
    if width < 256 and not rest:
        i = packed(e, off, size)
        if i is None or size != width:
            raise Fail()
        return ("st", size, ("si", base, i), width)

    i, s, c = linear(e)
    if s % stride:
        raise Fail()
    if s:
        index = mul_op(s // stride, i) if s != stride else i
        if c // stride:
            index = add_op(index, c // stride)
    else:
        index = c // stride
    member = c % stride
    rest = ([("off", member)] if member else []) + rest
    return form(("si", base, index), et, rest, size, off)


"""
    the check
"""


def read(f, defs, lang, w):
    """(slot, off, size) the printed access f reads, in the world w"""
    if m := match(
        f, ("mask_shl", ":int:size", ":int:d", ":int:shl", ("st", Any, ":loc", Any))
    ):
        slot, off, width, t = read_loc(m.loc, defs, lang, w)
        return slot, off + m.d, m.size
    if m := match(f, ("st", ":size", ":loc", Any)):
        slot, off, width, t = read_loc(m.loc, defs, lang, w)
        return slot, off, m.size
    raise Unknown()


def read_loc(loc, defs, lang, w):
    """(slot, bit, width, type) of a loc, in the world w"""
    op = loc[0]
    if op == "sv":
        d = defs[loc[1]]
        return d["slot"], d.get("off", 0), value_width(d["type"]), d["type"]
    if op == "sr":
        return ev(loc[1], w), 0, 256, None
    if op == "sf":
        slot, bit, width, t = read_loc(loc[1], defs, lang, w)
        n = loc[2]
        return (slot + n // 256) % 2**256, n % 256, 256 - n % 256, None
    if op == "sl":
        slot, bit, width, t = read_loc(loc[1], defs, lang, w)
        return slot, 0, 256, None
    if op == "sbl":
        slot, bit, width, t = read_loc(loc[1], defs, lang, w)
        return ("blen", slot), 0, 256, None
    if op == "si":
        slot, bit, width, t = read_loc(loc[1], defs, lang, w)
        key = loc[2]
        if t is None:
            raise Unknown()
        if t[0] == "mapping":
            if opcode(key) == "data":
                kb = evb(key, w)
            else:
                kb = w32(ev(key, w))
            data = w32(slot) + kb if lang == "vyper" else kb + w32(slot)
            return keccak(data), 0, value_width(t[1]), t[1]
        if t[0] == "bytes":
            return (keccak(w32(slot)) + ev(key, w)) % 2**256, 0, 256, None
        if t[0] in ("array", "fixed"):
            _, et, stride = t
            start = keccak(w32(slot)) if t[0] == "array" else slot
            i = ev(key, w)
            width = elem_width(et)
            if width < 256:
                per = 256 // width
                return (start + i // per) % 2**256, width * (i % per), width, et
            return (start + i * stride) % 2**256, 0, value_width(et), et
    raise Unknown()


def access_value(size, off, idx, w):
    """(slot, off, size) of an access, in the world w"""
    if m := match(idx, ("length", ":key")):
        key = m.key
        if m2 := match(key, ("loc", ":int:n")):
            key = m2.n
        return ("blen", ev(key, w)), ev(off, w), ev(size, w)
    return ev(idx, w), ev(off, w), ev(size, w)


def checks(f, size, off, idx, defs, lang):
    """whether f reads as the access (size, off, idx), in every world"""
    try:
        for w in WORLDS:
            got = read(f, defs, lang, w)
            want = access_value(size, off, idx, w)
            got = (got[0], got[1] % 2**256, got[2] % 2**256)
            if got != want:
                return False
        return True
    except (Unknown, KeyError, ValueError, AssertionError, OverflowError):
        return False


def raw(size, off, idx, write=False):
    """the access as the slot it is: stor[idx]"""
    if m := match(idx, ("length", ":key")):
        # the length of a bytes (see simplify.replace_bytes_or_string_length)
        key = m.key[1] if opcode(m.key) == "loc" else m.key
        slot = ("st", 256, ("sr", key), 256)
        length = (
            "mask_shl",
            255,
            1,
            -1,
            (
                "and",
                slot,
                (
                    "add",
                    -1,
                    ("mask_shl", 248, 0, 8, ("iszero", ("st", 1, ("sr", key), 256))),
                ),
            ),
        )
        if write:
            raise Fail()
        return (
            ("mask_shl", size, off, ("mul", -1, off), length)
            if off != 0 or size != 256
            else length
        )
    if opcode(idx) == "loc":
        idx = idx[1]
    base = ("sr", idx)
    if type(size) != int:
        # as many bits as computed at runtime
        if write:
            return ("st", size, ("sf", base, off), None)
        return ("mask_shl", size, off, ("mul", -1, off), ("st", 256, base, 256))
    if off == 0:
        return ("st", size, base, 256)
    if type(off) == int and off > 0:
        return ("st", size, ("sf", base, off), 256 - off)
    if write:
        # bits at an offset computed at runtime
        return ("st", size, ("sf", base, off), None)
    return ("mask_shl", size, off, ("mul", -1, off), ("st", 256, base, 256))


"""
    the whole contract
"""


def find_accesses(exp, res):
    """
    The accesses in exp: (size, off, idx) -> False when it's only read,
    "const" when it's written numbers only, True when it's written others.
    """
    if type(exp) == list:
        for e in exp:
            find_accesses(e, res)
        return
    if type(exp) != tuple:
        return
    if opcode(exp) == "storage" and len(exp) == 4:
        res.setdefault(exp[1:], False)
    elif opcode(exp) == "store" and len(exp) == 5:
        if type(exp[4]) != int:
            res[exp[1:4]] = True
        elif res.get(exp[1:4]) is not True:
            res[exp[1:4]] = "const"
    for e in exp:
        find_accesses(e, res)


def language(code, accesses):
    """ "vyper" or "solidity": from the metadata, else from how mappings hash"""
    tail = bytes(code[-80:]) if code else b""
    if b"vyper" in tail:
        return "vyper"
    if b"solc" in tail or b"bzzr" in tail or b"ipfs" in tail:
        return "solidity"
    votes = 0
    for size, off, idx in accesses:
        for h in find_hashes(idx):
            parts = h[1][1:] if len(h) == 2 and opcode(h[1]) == "data" else h[1:]
            if len(parts) != 2:
                continue
            a, b = word(parts[0]), word(parts[1])
            if type(a) == int and type(b) != int:
                votes += 1
            elif type(b) == int and type(a) != int:
                votes -= 1
    return "vyper" if votes > 0 else "solidity"


def checked_below(trace, loops=False, numbers=True):
    """
    {x: n} for the checks that x < n, that revert otherwise, in a trace -
    and with loops, the conditions x < n of its loops (in their bodies).
    Without numbers, x < y as well, whatever y is (a length read from the
    storage), with n None.
    """
    res = {}
    n = ":int:n" if numbers else ":n"

    def reverts(branch):
        return len(branch) == 1 and opcode(branch[0]) in ("revert", "invalid")

    def add(cond):
        """cond must hold for the execution to go on"""
        while m := match(cond, ("iszero", ("iszero", ":c"))):
            cond = m.c
        for pattern, delta in (
            (("lt", ":x", n), 0),
            (("gt", n, ":x"), 0),
            (("le", ":x", n), 1),
            (("ge", n, ":x"), 1),
            (("iszero", ("ge", ":x", n)), 0),
            (("iszero", ("le", n, ":x")), 0),
            (("iszero", ("gt", ":x", n)), 1),
            (("iszero", ("lt", n, ":x")), 1),
        ):
            if not (m := match(cond, pattern)) or type(m.x) == int:
                continue
            if not numbers:
                res[m.x] = None
            elif 0 < m.n + delta < 2**32:
                res.setdefault(m.x, m.n + delta)
                res[m.x] = min(res[m.x], m.n + delta)

    def visit(lines):
        for line in lines:
            if opcode(line) == "if" and len(line) == 4:
                _, cond, if_true, if_false = line
                if reverts(if_false):
                    add(cond)
                if reverts(if_true):
                    add(("iszero", cond))
                visit(if_true)
                visit(if_false)
            elif opcode(line) == "while":
                if loops:
                    add(line[1])
                visit(line[2])

    visit(trace)
    return res


def var_values(trace):
    """name -> the values a variable is set to in a trace: by setvars, at the
    start of a loop, by its steps"""
    res = {}

    def visit(lines):
        for line in lines:
            op = opcode(line)
            if op == "setvar" and len(line) == 3:
                res.setdefault(line[1], []).append(line[2])
            elif op == "if":
                for branch in line[2:]:
                    visit(branch)
            elif op == "while":
                visit(line[2])
                visit(line[4])
            elif op == "continue":
                visit(line[2])

    visit(trace)
    return res


def points(v):
    """whether v may be a slot a storage pointer holds: a hash, a number that
    big, a word of the memory (where solidity keeps them)"""
    return bool(
        find_f_list(
            v,
            lambda x: [x]
            if (type(x) == int and not -(2**32) < x < 2**32)
            or opcode(x) in ("sha3", "mem")
            else [],
        )
    )


def index_like(e, bounded, values, seen=()):
    """
    Whether e is an index - of a fixed array, its element at the slot plus
    e: made of numbers, of what the code checks is below something (a
    param, a loop's counter), of variables set to such (from 0 on), of
    numbers of a few bits (a uint16) or remainders. Not a pointer to the
    storage (a hash, a slot read from the memory, a param it doesn't check):
    the slot it points to plus a member's offset is no element of an array
    at that offset. bounded: what the code checks is below something (see
    checked_below), values: the values of the variables (see var_values).
    """
    if type(e) == int:
        return -(2**32) < e < 2**32
    op = opcode(e)
    if op == "var":
        if e[1] in seen:
            # (its step, from itself: s + 1)
            return True
        vs = values.get(e[1], [])
        if any(points(v) for v in vs):
            # (a pointer, below the end of what it points to, say)
            return False
        if e in bounded:
            return True
        return bool(vs) and all(index_like(v, bounded, values, seen + (e[1],)) for v in vs)
    if e in bounded:
        return True
    if op in ("add", "mul", "div"):
        return all(index_like(t, bounded, values, seen) for t in e[1:])
    if op == "mod":
        # (a remainder: below what it's taken of)
        return True
    if op == "mask_shl" and all(type(t) == int for t in e[1:4]):
        return e[1] <= 64 or index_like(e[4], bounded, values, seen)
    return False


def find_hashes(exp):
    if type(exp) != tuple:
        return []
    res = [exp] if opcode(exp) == "sha3" else []
    for e in exp[1:]:
        res += find_hashes(e)
    return res


def layout(accs):
    """
    The variables of a slot, (off, end) each, from its accesses (size, off,
    written - see find_accesses): what's written (a read of more than one is
    printed raw), else what's read. A number written says nothing of how
    wide it is - a store of the word, split, sets 1 in a byte and 0 in the
    bits above it -, so its bits are variables of their own only where
    nothing else is written.
    """
    written = [(size, off) for size, off, wr in accs if wr is True]
    numbers = [(size, off) for size, off, wr in accs if wr == "const"]
    fields = merge_ranges(written)
    numbers = merge_ranges(uncovered(numbers, fields))
    return merge_ranges(
        [(e - o, o) for o, e in fields + numbers]
        or [(size, off) for size, off, wr in accs]
    )


def slot_class(st):
    """
    What the slots of the same place of a variable have in common - the
    steps to them, without their keys and indexes: the slot of every element
    of a mapping, the second slot of every struct of an array.
    """
    if st is None:
        return None
    root, sts = st
    res = []
    for s in sts:
        if s[0] == "map":
            res.append(("map", s[2]))
        elif s[0] == "data":
            res.append(("data",))
        elif s[0] == "off":
            i, stride, c = linear(s[1])
            res.append(("off", stride, c))
        else:
            return None
    return root, tuple(res)


class Regroup:
    """
    The stores of the variables of a slot, not of the parts a store of the
    whole word was split in (memloc.split_store): 1 in the bits 216-224 and
    0 in the 224-240 are 1 in the uint16 at 216 and 0 in the uint8 at 232;
    1 in the bits 0-8 and 0 in the 8-160 of an element of a mapping of
    addresses is 1 in it. The variables are those of all the slots of the
    same place (see slot_class, layout).
    """

    def __init__(self, functions, lang):
        accesses = {}
        for f in functions:
            find_accesses(f.trace, accesses)
        self.cls = {}
        by_cls = {}
        for (size, off, idx), wr in accesses.items():
            try:
                c = slot_class(steps(parse(idx, lang)))
            except Exception:
                c = None
            self.cls[(size, off, idx)] = c
            if c is not None:
                by_cls.setdefault(c, []).append((size, off, wr))
        self.fields = {
            c: layout(accs)
            for c, accs in by_cls.items()
            if all(type(size) == int and type(off) == int for size, off, wr in accs)
        }

    def store(self, line):
        """(class, size, off, value, idx) of a store of bits of a slot of known fields"""
        if not (m := match(line, ("store", ":int:size", ":int:off", ":idx", ":val"))):
            return None
        c = self.cls.get((m.size, m.off, m.idx))
        if c not in self.fields:
            return None
        return c, m.size, m.off, m.val, m.idx

    def trace(self, trace):
        if type(trace) != list:
            return trace
        lines = [
            (
                tuple(self.trace(e) if type(e) == list else e for e in line)
                if type(line) == tuple
                else line
            )
            for line in trace
        ]
        res, i = [], 0
        while i < len(lines):
            first = self.store(lines[i])
            j = i + 1
            while (
                first is not None
                and j < len(lines)
                and (s := self.store(lines[j])) is not None
                and s[4] == first[4]
            ):
                j += 1
            if first is not None:
                res += self.run(lines[i:j])
            else:
                res.append(lines[i])
            i = j
        return res

    def run(self, lines):
        """stores one after the other in the same slot, per variable of it"""
        run = [self.store(line) for line in lines]
        c, idx = run[0][0], run[0][4]
        # parts of different bits, their values not reading the slot (else
        # they'd read what the others write) - nor another of the same place:
        # it may be the same
        bits = sorted((off, off + size) for _, size, off, val, i in run)
        if any(a[1] > b[0] for a, b in zip(bits, bits[1:])):
            return lines
        for _, size, off, val, i in run:
            reads = {}
            find_accesses(val, reads)
            if any(self.cls.get(a) in (None, c) for a in reads):
                return lines

        # a number in whole variables (0 in the word: a delete) stays one
        fields = self.fields[c]

        def whole(size, off):
            inside = [(lo, hi) for lo, hi in fields if lo < off + size and off < hi]
            return all(off <= lo and hi <= off + size for lo, hi in inside) and (
                sum(hi - lo for lo, hi in inside) == size
            )

        res = [
            ("store", size, off, idx, val)
            for _, size, off, val, i in run
            if type(val) == int and whole(size, off)
        ]
        run = [r for r in run if not (type(r[3]) == int and whole(r[1], r[2]))]
        for lo, hi in fields:
            parts = []
            for _, size, off, val, i in run:
                a, b = max(lo, off), min(hi, off + size)
                if a >= b:
                    continue
                if type(val) == int:
                    # (a number's bits in this variable)
                    parts.append((b - a, a, (val >> (a - off)) % 2 ** (b - a)))
                elif (a, b) == (off, off + size):
                    parts.append((size, off, val))
                else:
                    # a value over two variables
                    return lines
            if not parts:
                continue
            covered = sum(size for size, off, val in parts) == hi - lo
            others = [p for p in parts if type(p[2]) != int]
            if covered and not others:
                value = sum(val << (off - lo) for size, off, val in parts)
                parts = [(hi - lo, lo, value)]
            elif (
                covered
                and len(others) == 1
                and others[0][1] == lo
                and value_bits(others[0][2]) <= others[0][0]
                and all(val == 0 for size, off, val in parts if type(val) == int)
            ):
                parts = [(hi - lo, lo, others[0][2])]
            for size, off, val in parts:
                res.append(("store", size, off, idx, val))

        if sorted(line[1:3] for line in res) == sorted(
            line[1:3] for line in lines
        ) or not all(whole(line[1], line[2]) for line in res):
            # the same stores - or some still not of whole variables
            return lines
        for line in res:
            self.cls[line[1:4]] = c
        return sorted(res, key=lambda line: line[2])


def uncovered(parts, ranges):
    """the (size, off) of the bits of the parts (size, off) no range (off, end) has"""
    res = []
    for size, off in parts:
        pos, end = off, off + size
        for o, e in sorted(ranges):
            if e <= pos or o >= end:
                continue
            if o > pos:
                res.append((o - pos, pos))
            pos = max(pos, e)
        if pos < end:
            res.append((end - pos, pos))
    return res


def merge_ranges(fields):
    """the intervals (off, end) the fields (size, off) cover, overlapping ones merged"""
    res = []
    for off, end in sorted((off, off + size) for size, off in fields):
        if res and off < res[-1][1]:
            res[-1] = (res[-1][0], max(res[-1][1], end))
        else:
            res.append((off, end))
    return res


def type_name(width):
    if width == 160:
        return "address"
    return f"uint{width}"


# what a name of the output means already
RESERVED = {
    "stor",
    "mem",
    "call",
    "calldata",
    "block",
    "tx",
    "msg",
    "this",
    "caller",
    "Mask",
    "Bytes",
    "Array",
    "concat",
    "sha3",
    "bool",
    "address",
    "ceil32",
    "floor32",
    "addmod",
    "mulmod",
    "max",
    "min",
    "ext_call",
    "ext_code",
    "eth",
    "transient",
    "not",
    "and",
    "or",
    "if",
    "while",
    "return",
    "revert",
}


def good_name(name):
    """whether a getter's name can name its variable"""
    import re

    return (
        name.isidentifier()
        and not name.startswith("unknown")
        and not name.startswith("_")
        and name not in RESERVED
        and not re.fullmatch(
            r"(u?int|bytes|stor)\d*|stor[0-9A-F]+(_\d+)?|stor\d+_\d+", name
        )
    )


def getter_access(func):
    """
    (size, off, idx) of what a getter returns - the value of a variable, or
    it as its type makes it: a cast, left-aligned (bytesN), sign-extended,
    a bool - or ("root", n) for the members of a struct at n.
    """
    if not func.getter:
        return None
    if m := match(func.getter, ("struct", ("loc", ":int:n"))):
        return ("root", m.n)
    returns = getattr(func, "returns", None) or []
    if len(returns) != 1 or len(returns[0]) != 2:
        return None
    ret = replace_f(word(returns[0][1]), narrow)
    if m := match(ret, ("bool", ":s")):
        ret = m.s
    if m := match(ret, ("signextend", Any, ":s")):
        ret = m.s
    if (
        (m := match(ret, ("mask_shl", ":int:size", 0, ":int:shl", ":s")))
        and opcode(m.s) == "storage"
        and type(m.s[1]) is int
    ):
        # (a read of a size computed at runtime is no getter's)
        s_size = m.s[1]
        if (m.shl == 0 and m.size >= s_size) or (
            m.size == s_size and m.shl == 256 - m.size
        ):
            ret = m.s
    if opcode(ret) == "storage" and len(ret) == 4:
        return ret[1:]
    return None


def bytes_tail(trace, key, selector):
    """
    The trace with what's after a first part of it - its checks of the
    params, say - replaced by the return of the bytes (or string) of the
    storage at the slot key, ABI-encoded, `return Array(len=b.length,
    data=b[all])`, when that's the same - or None. That is checked by
    running both (see runtrace), on bytes as the compiler keeps them, short
    and long, of every length the code may treat apart, and params clean
    and dirty, and of the numbers the code compares with; the trace may
    read nothing else of the storage.
    """
    import random

    from panoramix.runtrace import Machine, Unsupported, keccak

    rnd = random.Random(0)
    lengths = [0, 1, 2, 30, 31, 32, 33, 63, 64, 65, 96, 97, 300]
    # and about the numbers of the code (a length compared with one)
    around = set()
    for c in find_f_list(
        trace, lambda e: [e] if type(e) == int and 0 < e < 4096 else []
    ):
        around |= {c - 1, c, c + 1, c // 2, c // 2 + 1}
    lengths += sorted(around - set(lengths))[:48]
    values = [0, 1, 2, 255, 2**16 - 1, 2**160 - 1, 2**160, 2**255, 2**256 - 1]

    worlds = []
    for n, length in enumerate(lengths):
        content = bytes(rnd.randrange(1, 256) for _ in range(length))
        params = [values[(n + 3 * k) % len(values)] for k in range(8)]
        calldata = selector.to_bytes(4, "big") + b"".join(
            p.to_bytes(32, "big") for p in params
        )
        worlds.append((calldata, content))
    # and params of the numbers the conditions of the code compare with, and
    # of the ones next to them: `if tokenId == 1337: return ''` is a branch
    # none of those values takes
    near = set()
    for cond in find_f_list(
        trace, lambda e: [e[1]] if opcode(e) in ("if", "while") else []
    ):
        for c in find_f_list(cond, lambda e: [e] if type(e) == int else []):
            near |= {(c + d) % 2**256 for d in (-1, 0, 1)}
    for n, value in enumerate(sorted(near - set(values))[:64]):
        length = lengths[n % len(lengths)]
        content = bytes(rnd.randrange(1, 256) for _ in range(length))
        worlds.append((selector.to_bytes(4, "big") + value.to_bytes(32, "big") * 8, content))
    for n in (0, 31):
        # calldata too short for a param
        worlds.append((selector.to_bytes(4, "big") + b"\x01" * n, b"ab"))

    def run(t, calldata, content):
        """(how t ends, whether it returned the bytes) - None if it can't be run"""
        words = {}
        reached = []

        def sload(slot):
            if slot not in words:
                raise Unsupported("another slot")
            return words[slot]

        def bytes_length(slot):
            v = sload(slot)
            return (v - 1) // 2 if v & 1 else (v & 0xFF) // 2

        def bytes_data(slot):
            reached.append(slot)
            return content

        m = Machine(calldata, sload, bytes_length, bytes_data=bytes_data)
        try:
            slot = m.ev(key)
            padded = content + b"\0" * (-len(content) % 32)
            if len(content) < 32:
                words[slot] = int.from_bytes(padded.ljust(32, b"\0"), "big") | 2 * len(
                    content
                )
            else:
                words[slot] = 2 * len(content) + 1
                base = keccak(slot.to_bytes(32, "big"))
                for i in range(len(padded) // 32 + 1):
                    words[(base + i) % 2**256] = int.from_bytes(
                        padded[32 * i : 32 * i + 32].ljust(32, b"\0"), "big"
                    )
            return m.run(t), bool(reached)
        except (Unsupported, RecursionError, ValueError, OverflowError, KeyError):
            return None

    orig = [run(trace, *w) for w in worlds]
    if None in orig or not any(r[0][0] == "return" for r in orig):
        return None
    tail = [
        (
            "return",
            ("data", ("arr", ("storage", 256, 0, ("length", key)), ("sbytes", key))),
        )
    ]

    def returns_bytes(r, content):
        """whether the run r returned the content, ABI-encoded"""
        data = content + b"\0" * (-len(content) % 32)
        data = (32).to_bytes(32, "big") + len(content).to_bytes(32, "big") + data
        return r[0] == ("return", data)

    for n, cand in enumerate(splits(trace, tail)):
        if n > 200:
            break
        reached = False
        for w, r in zip(worlds, orig):
            got = run(cand, *w)
            if got is None or got[0] != r[0]:
                break
            if w[1] and returns_bytes(r, w[1]) and not got[1]:
                # the bytes returned by what's kept of the code, not by the
                # tail (a branch kept returns something else: `return ''`)
                break
            reached = reached or got[1]
        else:
            if reached:
                return cand
    return None


def splits(trace, tail):
    """
    The trace with what's after a first part of it replaced by tail - the
    least first: after each line, and into the branch of an if that goes on
    when the other ends (a check), then the line kept.
    """

    def ends(lines):
        return bool(lines) and opcode(lines[-1]) in (
            "revert",
            "invalid",
            "return",
            "stop",
        )

    for i, line in enumerate(trace):
        yield trace[:i] + tail
        if opcode(line) == "if" and len(line) == 4:
            _, cond, a, b = line
            rest = list(trace[i + 1 :])
            if ends(a):
                for sub in splits(list(b) + rest, tail):
                    yield trace[:i] + [("if", cond, a, sub)]
            if ends(b):
                for sub in splits(list(a) + rest, tail):
                    yield trace[:i] + [("if", cond, sub, b)]


def getter_path(t, sts, size, off):
    """
    Whether the access of what's at the steps sts from a root of type t is
    what a getter of it returns: a value of it, reached by keys and indexes
    that are the getter's params - not the length of an array, nor a bytes,
    nor an element found some other way. (Of a bytes, its length is: the
    getter returns its bytes, see Storage.bytes_getter.)
    """
    if t is None or (t[0] == "bytes" and not sts):
        return False
    try:
        f = form(("sv", None), t, sts, size, off)
    except Fail:
        return False
    if opcode(f) != "st":
        return False
    loc, keys = f[2], []
    if loc[0] == "sbl" and sts[-1:] == [("blen",)]:
        loc = loc[1]
    while loc[0] != "sv":
        if loc[0] == "si":
            keys.append(loc[2])
        elif loc[0] != "sf":
            return False
        loc = loc[1]
    return all(from_calldata(k) for k in keys) and len(set(keys)) == len(keys)


class Storage:
    def __init__(self, functions, lang):
        self.lang = lang
        self.accesses = {}
        where = {}  # access -> (what's bounded, the variables' values) where it's made
        for f in functions:
            find_accesses(f.trace, self.accesses)
            accesses = {}
            find_accesses(f.trace, accesses)
            known = None
            for a in accesses:
                if known is None:
                    known = (
                        checked_below(f.trace, loops=True, numbers=False),
                        var_values(f.trace),
                    )
                where.setdefault(a, []).append(known)

        self.steps = {}
        pointers = set()  # the accesses at a slot plus what's no index
        for a in self.accesses:
            size, off, idx = a
            try:
                st = steps(parse(idx, lang))
            except Exception:
                logger.exception("storage path of %s", idx)
                st = None
            if (
                st is not None
                and st[1]
                and st[1][0][0] == "off"
                and type(st[1][0][1]) != int
                and not any(index_like(st[1][0][1], *known) for known in where[a])
            ):
                pointers.add(a)
            self.steps[a] = st

        # A slot plus what's no index (a storage pointer plus a member's
        # offset, a slot read from the calldata) is taken for an element of a
        # fixed array only where nothing else is: not to make the variable,
        # the mapping or the array at that slot an array of what it isn't -
        # it's the slot it is then.
        others = {st[0] for a, st in self.steps.items() if st is not None and a not in pointers}
        for a in pointers:
            if self.steps[a][0] in others:
                self.steps[a] = None

        by_root = {}
        for a, st in self.steps.items():
            size, off, idx = a
            if st is not None and type(size) == int:
                by_root.setdefault(st[0], []).append(
                    (st[1], size, off, self.accesses[a])
                )

        self.type_roots(by_root)

        # the fixed arrays whose length the functions check: the slots after
        # the first are theirs, not other variables - but the ones a getter
        # returns, as a variable (an array found at the slot plus an index
        # that is one after another's, its length absorbed them)
        self.lengths = self.fixed_lengths(functions)
        named = set()
        for f in functions:
            a = getter_access(f)
            if a is not None and a[0] != "root" and (st := self.steps.get(a)):
                if not st[1]:
                    named.add(st[0])
        moved = False
        for n, length in sorted(self.lengths.items()):
            t = self.types.get(n)
            if n not in by_root or t is None or t[0] != "fixed":
                continue
            width = elem_width(t[1])
            slots = -(-length // (256 // width)) if width < 256 else length * t[2]
            for r in range(n + 1, n + slots):
                if r in self.fields and r in by_root and r not in named:
                    for a, st in list(self.steps.items()):
                        if st == (r, []):
                            self.steps[a] = (n, [("off", r - n)])
                    by_root[n] += [
                        ([("off", r - n)], size, off, wr)
                        for st, size, off, wr in by_root.pop(r)
                    ]
                    moved = True
        if moved:
            self.type_roots(by_root)

        # the functions that return a bytes (or string) of the storage, as
        # running them shows: key -> its slot
        self.bytes_getters = {}
        for f in functions:
            if (key := self.bytes_getter(f)) is not None:
                self.bytes_getters[f] = key

        self.names(functions)

    def type_roots(self, by_root):
        """what's at each root"""
        self.types = {}
        self.fields = {}  # root -> [(off, end)] of a root that holds values
        for n, accs in by_root.items():
            if all(not st for st, size, off, wr in accs) and all(
                type(off) == int for st, size, off, wr in accs
            ):
                read = {(size, off) for st, size, off, wr in accs}
                if {(1, 0), (7, 1), (255, 1)} <= read:
                    # (none of its data read: a bytes, or a string, only if
                    # its length is read as theirs is - the lowest bit tells
                    # a short one, whose length is the 7 bits from the bit 1,
                    # from a long one, the bits from 1 on. A flag in the bit
                    # 0 of an uint8 and the 7 bits above it are no bytes)
                    self.types[n] = ("bytes",)
                    continue
                self.fields[n] = layout([(size, off, wr) for st, size, off, wr in accs])
                continue
            try:
                self.types[n] = typeof([(st, size, off) for st, size, off, wr in accs])
            except Exception:
                logger.exception("storage type at %s", n)

    def fixed_lengths(self, functions):
        """
        root -> the length of the fixed array there, when the functions that
        index it check the index is below a number first (and all the same)
        """
        found = {}
        for f in functions:
            accesses = {}
            find_accesses(f.trace, accesses)
            checked = checked_below(f.trace)
            if not checked:
                continue
            for a in accesses:
                st = self.steps.get(a)
                if (
                    not st
                    or not st[1]
                    or st[1][0][0] != "off"
                    or type(st[1][0][1]) == int
                ):
                    continue
                n = st[0]
                t = self.types.get(n)
                if t is None or t[0] != "fixed":
                    continue
                size, off, idx = a
                e = st[1][0][1]
                if elem_width(t[1]) < 256 and type(size) == int:
                    i = packed(e, off, size)
                else:
                    i, s, c = linear(e)
                    i = i if s == t[2] and c == 0 else None
                if i is not None and i in checked:
                    found.setdefault(n, set()).add(checked[i])
        return {n: ls.pop() for n, ls in found.items() if len(ls) == 1}

    def names(self, functions):
        """the names of the variables, one each, from the getters"""
        getters = {}
        for f in sorted(functions, key=lambda f: f.name):
            if not f.getter and f not in self.bytes_getters:
                continue
            name = f.name.split("(")[0]
            if name.startswith("get") and len(name) > 3 and name[3].isupper():
                # getBalance: balance
                name = name[3].lower() + name[4:]
            if not good_name(name):
                continue
            if f in self.bytes_getters:
                length = (256, 0, ("length", self.bytes_getters[f][0]))
                n, sts = self.steps[length]
                if getter_path(self.types.get(n), sts, 256, 0):
                    getters.setdefault(("root", n), name)
                continue
            a = getter_access(f)
            if a is None:
                continue
            if a[0] == "root":
                getters.setdefault(("root", a[1]), name)
                continue
            size, off, idx = a
            st = self.steps[a] if a in self.steps else steps(parse(idx, self.lang))
            if st is None:
                continue
            n, sts = st
            if n in self.fields and not sts:
                # (a field is bits known: not the bits of an access at a
                # place computed at runtime, a byte of the slot by an index)
                if type(off) is int and type(size) is int:
                    getters.setdefault(("field", n, off, off + size), name)
            elif getter_path(self.types.get(n), sts, size, off):
                getters.setdefault(("root", n), name)

        self.defs = {}
        self.var_of = {}  # ("root", n) / ("field", n, off, end) -> name
        taken = set()

        def unique(name, n):
            if name in taken:
                name = f"{name}_{n if n < 2**32 else hex(n)[2:10]}"
            k = 2
            base = name
            while name in taken:
                name = f"{base}_{k}"
                k += 1
            taken.add(name)
            return name

        def default(n, off=0):
            if n < 2**32:
                name = f"stor{n}"
            else:
                name = "stor" + hex(n)[2:6].upper()
            return name if not off else f"{name}_{off}"

        for n in sorted(set(self.types) | set(self.fields)):
            if n in self.fields:
                for off, end in self.fields[n]:
                    name = getters.get(("field", n, off, end)) or default(n, off)
                    name = unique(name, n)
                    self.var_of[("field", n, off, end)] = name
                    self.defs[name] = {
                        "slot": n,
                        "off": off,
                        "type": ("value", end - off),
                    }
            else:
                name = getters.get(("root", n)) or default(n)
                name = unique(name, n)
                self.var_of[("root", n)] = name
                self.defs[name] = {"slot": n, "type": self.types[n]}

    def bytes_getter(self, f):
        """
        The slot (its expression) of the bytes of the storage the function
        f returns, ABI-encoded - whatever its code is: that is checked by
        running it (see bytes_tail) - and f's trace returning them so, or
        None.
        """
        if not getattr(f, "read_only", False) or type(f.hash) != str:
            return None
        try:
            selector = int(f.hash, 16)
        except ValueError:
            return None
        keys = []
        for e in find_f_list(f.trace, lambda e: [e] if opcode(e) == "storage" else []):
            k = (
                e[3][1]
                if opcode(e[3]) == "length"
                else e[3] if e[1:3] == (1, 0) else None
            )
            if opcode(k) == "loc":
                k = k[1]
            if k is not None and k not in keys:
                keys.append(k)
        for key in keys[:3]:
            key = ("loc", key) if type(key) == int else key
            length = (256, 0, ("length", key))
            if length not in self.steps:
                try:
                    self.steps[length] = steps(parse(("length", key), self.lang))
                except Exception:
                    continue
            st = self.steps[length]
            if st is None or st[0] not in self.types:
                continue
            try:
                f_len = form(("sv", None), self.types[st[0]], st[1], 256, 0)
            except Fail:
                continue
            if not match(f_len, ("st", 256, ("sbl", Any), 256)):
                continue
            if (trace := bytes_tail(f.trace, key, selector)) is not None:
                return key, trace
        return None

    def getter_trace(self, f):
        """the trace of a bytes getter: `return name[all]`, ABI-encoded - or None"""
        if f not in self.bytes_getters:
            return None
        key, trace = self.bytes_getters[f]
        f_len = self.form(256, 0, ("length", key), False)
        if not (m := match(f_len, ("st", 256, ("sbl", ":loc"), 256))):
            return None
        return self.rewrite(
            replace_f(trace, lambda e: ("sall", m.loc) if e == ("sbytes", key) else e)
        )

    def form(self, size, off, idx, write):
        """the printed access, checked, else raw"""
        st = self.steps.get((size, off, idx))
        f = None
        if st is not None and type(size) == int:
            n, sts = st
            try:
                f = self.path_form(n, sts, size, off)
            except Fail:
                f = None
            if (
                write
                and opcode(f) == "mask_shl"
                and opcode(f[4]) == "st"
                and opcode(f[4][2]) == "sv"
                and self.defs[f[4][2][1]].get("off", 0) == 0
            ):
                # some bits of a variable at the bit 0 of its slot
                f = ("st", size, ("sf", f[4][2], off), 256 - off)
            if f is not None and write and opcode(f) != "st":
                f = None
            if f is not None and not checks(f, size, off, idx, self.defs, self.lang):
                logger.warning(
                    "storage access %s printed as %s doesn't read as it",
                    (size, off, idx),
                    f,
                )
                f = None
        if f is None:
            f = raw(size, off, idx, write)
        return f

    def path_form(self, n, sts, size, off):
        if n in self.fields:
            if sts or type(off) != int:
                raise Fail()
            for o, e in self.fields[n]:
                if o <= off and off + size <= e:
                    name = self.var_of[("field", n, o, e)]
                    return form(("sv", name), ("value", e - o), [], size, off - o)
            raise Fail()
        if ("root", n) not in self.var_of:
            raise Fail()
        name = self.var_of[("root", n)]
        return form(("sv", name), self.types[n], sts, size, off)

    def rewrite(self, trace):
        """the trace with the accesses printed as their paths"""
        return rewrite_accesses(trace, self.form)

    def header(self):
        """the definitions, as the output lists them"""
        res = []
        for name, d in sorted(
            self.defs.items(), key=lambda x: (x[1]["slot"], x[1].get("off", 0), x[0])
        ):
            t = d["type"]
            if t[0] == "fixed" and d["slot"] in self.lengths:
                # (its length, for the reader)
                t = t + (self.lengths[d["slot"]],)
            res.append(("sdef", name, d["slot"], d.get("off", 0), t))
        return res


def rewrite_accesses(trace, form):
    """the trace with each access as form(size, off, idx, write) prints it"""
    forms = {}

    def get(size, off, idx, write):
        k = (size, off, idx, write)
        if k not in forms:
            forms[k] = form(size, off, idx, write)
        return forms[k]

    def parts(exp):
        """(what exp is made of, what of: list, tuple, "set") - or None for
        what stays as it is"""
        if type(exp) == list:
            return exp, list
        if type(exp) != tuple:
            return None
        if opcode(exp) == "storage" and len(exp) == 4:
            return [get(exp[1], exp[2], exp[3], False)], "one"
        if opcode(exp) == "store" and len(exp) == 5:
            size, off, idx, val = exp[1:]
            target = get(size, off, idx, True)
            if (
                opcode(target) == "st"
                and type(target[1]) == int
                and not (opcode(target[2]) == "sf" and type(target[2][2]) != int)
            ):
                return [target, val], "set"
            # bits at an offset computed at runtime: the whole word
            whole = ("st", 256, ("sr", idx), 256)
            mask = ("mask_shl", size, 0, off, M)
            merged = (
                "or",
                ("and", whole, ("not", mask)),
                ("mask_shl", size, 0, off, val),
            )
            return [whole, merged], "set"
        return exp, tuple

    # (in a loop rather than recursively: the trace of a function that calls
    # itself, unrolled, can nest deeper than python's recursion limit)
    done = []
    todo = [(trace, None)]
    while todo:
        exp, made = todo.pop()
        if made is not None:
            # what's made of the last n done
            kind, n = made
            args = done[len(done) - n :]
            del done[len(done) - n :]
            if kind == "one":
                done.append(args[0])
            elif kind == "set":
                done.append(("set",) + tuple(args))
            else:
                done.append(kind(args))
            continue
        p = parts(exp)
        if p is None:
            done.append(exp)
            continue
        elems, kind = p
        todo.append((None, (kind, len(elems))))
        todo.extend((e, None) for e in reversed(elems))

    return done[0]


def rewrite_raw(functions):
    """every access as the slot it is, when the paths can't be found"""
    for func in functions:
        func.trace = rewrite_accesses(func.trace, raw)


def narrow(exp):
    """
    A read of the storage as wide as what's used of it: of the low bits of
    storage(184, 72, x) << 224, only 32 are - storage(32, 72, x) << 224; of
    Mask(32, 72, storage(104, 0, x)), the bits 72 to 104 - storage(32, 72, x)
    << 72.
    """
    if (
        (
            m := match(
                exp,
                (
                    "mask_shl",
                    ":int:size",
                    ":int:moff",
                    ":shl",
                    ("storage", ":int:s_size", ":int:s_off", ":idx"),
                ),
            )
        )
        and 0 < m.size
        and 0 <= m.moff
        and m.moff + m.size <= m.s_size
        and (m.size, m.moff) != (m.s_size, 0)
        and not (
            # a word times 2**shl (32 * x): the bits it loses on top are no
            # field of it
            (m.s_size, m.s_off, m.moff) == (256, 0, 0)
            and m.shl == 256 - m.size
            and m.size % 8
        )
    ):
        shl = add_op(m.shl, m.moff) if m.moff else m.shl
        return (
            "mask_shl",
            m.size,
            0,
            shl,
            ("storage", m.size, m.s_off + m.moff, m.idx),
        )
    return exp


def rewrite_functions(functions, code=None):
    """puts the paths of the storage in the functions, returns the definitions"""
    for f in functions:
        f.trace = replace_f(f.trace, narrow)

    accesses = {}
    for f in functions:
        find_accesses(f.trace, accesses)
    lang = language(code, accesses)
    regroup = Regroup(functions, lang)
    for f in functions:
        f.trace = regroup.trace(f.trace)
    s = Storage(functions, lang)
    for f in functions:
        f.trace = s.getter_trace(f) or s.rewrite(f.trace)
    return lang, s.header()


def pretty_type(t):
    kind = t[0]
    if kind == "value":
        return type_name(t[1])
    if kind == "struct":
        if t[1] is None:
            return "struct"
        return f"struct of {t[1]} slot" + ("s" if t[1] != 1 else "")
    if kind == "mapping":
        return "mapping of " + pretty_type(t[1])
    if kind == "array":
        return "array of " + pretty_type(t[1])
    if kind == "fixed":
        if len(t) > 3 and t[1][0] == "value":
            return f"{pretty_type(t[1])}[{t[3]}]"
        if len(t) > 3:
            return f"fixed array of {t[3]} {pretty_type(t[1])}"
        return "fixed array of " + pretty_type(t[1])
    if kind == "bytes":
        return "bytes"
    return str(t)


def pretty_def(d):
    from panoramix.utils.helpers import COLOR_GRAY, COLOR_GREEN, ENDC

    _, name, slot, off, t = d
    loc = hex(slot) if slot >= 2**32 else str(slot)
    return (
        f"  {COLOR_GREEN}{name}{ENDC} is {pretty_type(t)} {COLOR_GRAY}at storage {loc}"
        + (f" offset {off}" if off else "")
        + ENDC
    )
