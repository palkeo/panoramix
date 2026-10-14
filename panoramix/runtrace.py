"""
A trace, run on concrete values.

What a rewrite of a whole function rests on, when no rule of the algebra
proves it - a string getter's copy loop is `return name` - is checked by
running the function's trace on values of what it reads (see
storage.bytes_getter). Anything this doesn't know how to run raises
Unsupported: then the rewrite isn't made.
"""

from panoramix.core.memloc import sizeof
from panoramix.utils.helpers import opcode

M = 2**256 - 1


class Unsupported(Exception):
    pass


class Halt(Exception):
    """the end of the call: "return", "revert", "stop", "invalid" and its data"""

    def __init__(self, kind, data=b""):
        self.kind = kind
        self.data = data


class Continue(Exception):
    def __init__(self, jd, setvars):
        self.jd = jd
        self.setvars = setvars


def keccak(b):
    from eth_hash.auto import keccak as k

    return int.from_bytes(k(b), "big")


def signed(v):
    return v - 2**256 if v >= 2**255 else v


def apply_mask(v, size, off, shl):
    if not (0 <= size <= 4096 and abs(off) <= 4096 and abs(shl) <= 4096):
        raise Unsupported("mask")
    if off < 0:
        size += off
        off = 0
    v &= ((1 << max(size, 0)) - 1) << off
    v = v << shl if shl >= 0 else v >> -shl
    return v & M


class Machine:
    """
    calldata: bytes; sload(slot) -> the word of the storage there;
    bytes_length(slot) -> the length of the bytes (or string) at the slot
    (see simplify.replace_bytes_or_string_length), bytes_data(slot) -> its
    bytes (("sbytes", slot), see storage.bytes_tail).
    """

    def __init__(
        self,
        calldata,
        sload,
        bytes_length,
        bytes_data=None,
        callvalue=0,
        max_steps=20000,
    ):
        self.calldata = calldata
        self.sload = sload
        self.bytes_length = bytes_length
        self.bytes_data = bytes_data
        self.callvalue = callvalue
        self.vars = {}
        self.mem = bytearray()
        self.steps = 0
        self.max_steps = max_steps

    def run(self, trace):
        """(kind, data) of how the trace ends"""
        try:
            self.run_trace(trace)
        except Halt as h:
            return h.kind, h.data
        return "stop", b""

    # ---- memory

    def mread(self, off, size):
        if off + size > 2**20:
            raise Unsupported("memory")
        if off + size > len(self.mem):
            self.mem.extend(b"\0" * (off + size - len(self.mem)))
        return bytes(self.mem[off : off + size])

    def mwrite(self, off, data):
        if off + len(data) > 2**20:
            raise Unsupported("memory")
        if off + len(data) > len(self.mem):
            self.mem.extend(b"\0" * (off + len(data) - len(self.mem)))
        self.mem[off : off + len(data)] = data

    def cdbytes(self, off, n):
        if n > 2**20:
            # (as the memory, see mread: bytes of a length read from the
            # calldata, say, would never be made)
            raise Unsupported("calldata")
        chunk = self.calldata[off : off + n] if off < len(self.calldata) else b""
        return chunk + b"\0" * (n - len(chunk))

    # ---- expressions

    def small(self, e):
        """a size, an offset, a shift: a small number, maybe negative"""
        if type(e) == int:
            return e
        return signed(self.ev(e))

    def ev(self, e):
        self.steps += 1
        if self.steps > 20 * self.max_steps:
            raise Unsupported("steps")
        if type(e) == int:
            return e & M
        if e == "calldatasize":
            return len(self.calldata)
        if e == "callvalue":
            return self.callvalue
        if type(e) != tuple or not e or type(e[0]) != str:
            raise Unsupported(repr(e)[:80])
        op = e[0]

        if op == "bytes":
            return self.ev(e[2])
        if op == "var":
            if e[1] not in self.vars:
                raise Unsupported("var")
            return self.vars[e[1]]
        if op == "loc":
            return self.ev(e[1])
        if op == "cd":
            off = self.ev(e[1])
            return int.from_bytes(self.cdbytes(off, 32), "big") if off < 2**32 else 0
        if op == "mask_shl":
            size, off, shl = (self.small(x) for x in e[1:4])
            return apply_mask(self.ev(e[4]), size, off, shl)
        if op == "storage":
            size, off = self.small(e[1]), self.small(e[2])
            if opcode(e[3]) == "length":
                v = self.bytes_length(self.ev(e[3][1]))
            else:
                v = self.sload(self.ev(e[3]))
            if off < 0:
                # a field moved left (see algebra.apply_mask_to_storage)
                return ((v & ((1 << size) - 1)) << -off) & M
            return apply_mask(v, size, off, -off)
        if op == "mem":
            if opcode(e[1]) != "range":
                raise Unsupported("mem")
            p, n = self.ev(e[1][1]), self.ev(e[1][2])
            if n > 32:
                raise Unsupported("mem")
            return int.from_bytes(self.mread(p, n), "big")
        if op == "call.data":
            p, n = self.ev(e[1]), self.ev(e[2])
            if n > 32 or p >= 2**32:
                raise Unsupported("call.data")
            return int.from_bytes(self.cdbytes(p, n), "big")
        if op == "sha3":
            if len(e) != 2:
                raise Unsupported("sha3")
            return keccak(self.evb(e[1]))

        args = [self.ev(x) for x in e[1:]]
        if op == "add":
            return sum(args) & M
        if op == "mul":
            r = 1
            for a in args:
                r = r * a & M
            return r
        if op in ("and", "or", "xor"):
            r = M if op == "and" else 0
            for a in args:
                r = r & a if op == "and" else r | a if op == "or" else r ^ a
            return r
        if len(args) == 1:
            a = args[0]
            if op == "not":
                return ~a & M
            if op == "iszero":
                return int(a == 0)
            if op == "bool":
                return int(a != 0)
        if len(args) == 2:
            a, b = args
            if op == "div":
                return a // b if b else 0
            if op == "mod":
                return a % b if b else 0
            if op == "sdiv":
                if b == 0:
                    return 0
                q = abs(signed(a)) // abs(signed(b))
                return (q if (signed(a) >= 0) == (signed(b) >= 0) else -q) & M
            if op == "smod":
                if b == 0:
                    return 0
                r = abs(signed(a)) % abs(signed(b))
                return (r if signed(a) >= 0 else -r) & M
            if op == "exp":
                return pow(a, b, 2**256)
            if op == "signextend":
                if a < 31:
                    bit = 8 * a + 7
                    b &= (1 << (bit + 1)) - 1
                    if b >> bit:
                        b |= M ^ ((1 << (bit + 1)) - 1)
                return b
            if op in ("lt", "gt", "le", "ge", "eq"):
                return int(
                    {
                        "lt": a < b,
                        "gt": a > b,
                        "le": a <= b,
                        "ge": a >= b,
                        "eq": a == b,
                    }[op]
                )
            if op in ("slt", "sgt", "sle", "sge"):
                a, b = signed(a), signed(b)
                return int(
                    {"slt": a < b, "sgt": a > b, "sle": a <= b, "sge": a >= b}[op]
                )
            if op == "shr":
                return b >> a if a < 256 else 0
            if op == "shl":
                return (b << a) & M if a < 256 else 0
            if op == "sar":
                return (signed(b) >> min(a, 256)) & M
            if op == "byte":
                return (b >> (248 - 8 * a)) & 0xFF if a < 32 else 0
            if op == "max":
                return max(a, b)
            if op == "min":
                return min(a, b)
        raise Unsupported(op)

    def evb(self, e):
        """the bytes of e as an element of a data"""
        op = opcode(e)
        if op == "data":
            # an ("arr", len, content...) in it is ABI-encoded: its offset,
            # then its length and content (padded to 32 bytes) after the rest
            heads, tails = [], []
            for el in e[1:]:
                if opcode(el) == "arr":
                    heads.append(len(tails))
                    tail = b"".join(self.evb(t) for t in el[2:])
                    tail += b"\0" * (-len(tail) % 32)
                    tails.append(self.ev(el[1]).to_bytes(32, "big") + tail)
                else:
                    heads.append(self.evb(el))
            pos = sum(32 if type(h) == int else len(h) for h in heads)
            res = b""
            for h in heads:
                if type(h) == int:
                    res += pos.to_bytes(32, "big")
                    pos += len(tails[h])
                else:
                    res += h
            return res + b"".join(tails)
        if op == "bytes":
            n = self.ev(e[1])
            if n > 32 and not (type(e[2]) == int and e[2] >= 0) or n > 2**20:
                raise Unsupported("bytes")
            v = e[2] if type(e[2]) == int and e[2] >= 2**256 else self.ev(e[2])
            return (v & ((1 << (8 * n)) - 1)).to_bytes(n, "big")
        if op == "mem":
            if opcode(e[1]) != "range":
                raise Unsupported("mem")
            return self.mread(self.ev(e[1][1]), self.ev(e[1][2]))
        if op == "call.data":
            return self.cdbytes(self.ev(e[1]), self.ev(e[2]))
        if op == "sbytes" and self.bytes_data is not None:
            return self.bytes_data(self.ev(e[1]))
        # a value: as wide as the memory model makes it (see memloc.sizeof)
        try:
            width = sizeof(e)
        except AssertionError:
            raise Unsupported("width")
        if type(width) != int or width % 8 or not 0 <= width <= 256:
            raise Unsupported("width")
        v = e if type(e) == int else self.ev(e)
        return (v & ((1 << width) - 1)).to_bytes(width // 8, "big")

    # ---- statements

    def run_trace(self, trace, loops=()):
        for line in trace:
            self.steps += 1
            if self.steps > self.max_steps:
                raise Unsupported("steps")
            self.run_line(line, loops)

    def run_line(self, line, loops):
        op = opcode(line)
        if op == "setvar":
            self.vars[line[1]] = self.ev(line[2])
        elif op == "setmem":
            if opcode(line[1]) != "range":
                raise Unsupported("setmem")
            p, n = self.ev(line[1][1]), self.ev(line[1][2])
            data = self.evb(line[2])
            if len(data) != n:
                if opcode(line[2]) in ("data", "mem", "call.data", "bytes") or n > 32:
                    raise Unsupported("setmem of another size")
                # a number: its low n bytes
                data = (int.from_bytes(data, "big") & ((1 << (8 * n)) - 1)).to_bytes(
                    n, "big"
                )
            self.mwrite(p, data)
        elif op == "if" and len(line) == 3:
            if self.ev(line[1]):
                self.run_trace(line[2], loops)
        elif op == "if":
            self.run_trace(line[2] if self.ev(line[1]) else line[3], loops)
        elif op == "while":
            _, cond, body, jd, setvars = line
            # all at once, as a continue's (see replace_vars)
            new = {}
            for sv in setvars:
                if opcode(sv) != "setvar":
                    raise Unsupported("while")
                new[sv[1]] = self.ev(sv[2])
            self.vars.update(new)
            while self.ev(cond):
                self.steps += 1
                if self.steps > self.max_steps:
                    raise Unsupported("steps")
                try:
                    self.run_trace(body, loops + (jd,))
                except Continue as c:
                    if c.jd != jd:
                        raise
                    new = {}
                    for sv in c.setvars:
                        if opcode(sv) != "setvar":
                            raise Unsupported("continue")
                        new[sv[1]] = self.ev(sv[2])
                    self.vars.update(new)
                    continue
                break  # the body ended without continuing: out of the loop
        elif op == "continue":
            raise Continue(line[1], line[2])
        elif op in ("return", "revert"):
            raise Halt(op, b"" if line[1] is None else self.evb(line[1]))
        elif op in ("stop", "invalid"):
            raise Halt(op)
        else:
            raise Unsupported(op)
