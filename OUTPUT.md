Reading the output
==================

The decompiled code is meant to be read one way only: this is that way. It
is python-like, with the conventions below; where the output can't say
something for sure, it says less (a raw `stor[...]`, `mem[...]`, a
`Bytes(n, ...)`) rather than something else.

## Values

- Every value is a 256-bit word, and arithmetic is modulo 2**256: `x - 1`
  with `x` 0 is 2**256 - 1. Division and modulo by 0 give 0.
- Operations are unsigned; a `′` marks the signed ones: `<′ >′ <=′ >=′`,
  `/′ %′ >>′` (arithmetic shift), `*′`. Numbers compared unsigned are printed
  unsigned (`x > 0xffff...ff00`, not `x > -256`); a big number added is
  printed as the word it is (`+ 0xf652...`), not as a negative one.
- Casts keep the low bits: `uint8(x)` is `x % 2**8`, `address(x)` the low 160
  bits; `int8(x)` is the low 8 bits sign-extended; `bool(x)` is 1 when `x`
  isn't 0. `Mask(size, off, x)` is the bits `off` to `off + size` of `x`, in
  place (the others 0). `ceil32(x)` / `floor32(x)` round to a multiple of 32,
  `min(x, y)` / `max(x, y)` are the smaller and the larger.
- `byte(i, x)` is the byte `i` of `x`, the most significant one first (0 for
  an `i` of 32 or more); `clz(x)` the number of leading zero bits of `x`
  (256 for 0: Osaka's CLZ); `signextend(b, x)` the lowest `b + 1` bytes of `x`
  sign-extended (`x` for a `b` of 31 or more); `shift(x, n)` is `x << n` when
  `n` is positive as a signed number (`n >=′ 0`), `x >> -n` when it isn't
  (Vyper's `shift`).
- `ext_code.size(a)` and `ext_code.hash(a)` are the size and the hash of the
  code of the account `a` (EXTCODESIZE, EXTCODEHASH), `ext_code(a).data[s len
  n]` its `n` bytes from `s` (zeroes past its end).

## Operators

Their precedence is python's, from the loosest: `or`, `and`, `not`,
comparisons, `|`, `^`, `&`, `<< >>`, `+ -`, `* / %`, unary `-`, `**`.

- `+ - * / % **` are arithmetic (`**` the power), `<< >>` shifts (by 256 or
  more: 0).
- `& | ^ ~` are bitwise (`^` is xor, `~` not).
- `and`, `or`, `not` are python's, of any values: `a and b` is `b` where `a`
  isn't 0, else `a`; `a or b` is `a` where it isn't 0, else `b` (`b` is
  evaluated only then); `not a` is 1 where `a` is 0, else 0. A comparison
  gives 0 or 1.
- A condition (`if`, `while`, `require`) holds when its value isn't 0;
  `a != b` is also how a xor tested for truth is printed.

## Statements

- `require cond`: reverts (with no data) unless `cond`; `assert cond`: runs an
  invalid opcode (the check of solidity < 0.8, all the gas used) unless
  `cond`.
- `revert with ...`, `return ...`: end the call with that data (see Data);
  `stop`: with none; `invalid`: an invalid opcode (all the gas used).
  `revert with 'text'` is the `Error(string)` of that text, `revert with
  Panic(n)` the `Panic(uint256)` of `n`; `return memory from a len n` (or
  `from a to b`) returns those bytes of memory.
- `while cond:` loops while `cond` holds; the end of its body goes back to
  `cond`. `continue` and `break` are about the innermost loop, or the one they
  name: `loop2: while ...` ... `continue loop2`.
- `x = ...` for a name that isn't a storage variable is a local variable (a
  local variable has no name of a storage variable or of a param).
- `mem[a]` is the word of memory at `a`, `mem[a len n]` its `n` bytes at `a`;
  `mem[a len n] = v` with `v` a number writes its low `n` bytes (zeroes
  above its word), with a list of data (`mem[a] = Bytes(12, 0), mem[b len
  20]`, see Data) those `n` bytes.

## Data

A list of data - what a `return`, a `revert`, a `log`, a `sha3`, the params
of a call, a `concat(...)` or `Array(...)` holds - is the bytes of its
elements, one after the other. An element is a 32-byte word, unless it says
how wide it is:

- `Bytes(n, v)`: the low `n` bytes of the number `v` (zeroes above its
  word) - of bytes `v` (a range, a string...), they're `n` bytes;
- `mem[a len n]`, `call.data[a len n]`, `ext_call.return_data[a len n]`...:
  those `n` bytes;
- `'text'`: its bytes - but when it's all the data of a `return` or a
  `revert` (after a 4-byte selector: `revert with 0x08c379a0, 'text'`), it's
  the ABI-encoded string (offset, length, bytes padded to 32);
- `Array(len=l, data=...)`: an ABI-encoded array;
- the 4-byte selector that starts the data of a revert or a call: printed as
  the error or the function it is when that's known (`Error(string reason)`),
  else as its 8 hex digits - a word that small is printed in decimal.

## Functions and params

- `def f(uint256 a, ...)` is the function whose selector is the hash of that
  signature, when it's known; else `unknown1234abcd(...)` (its selector), its
  params the words of calldata it reads, of the types how it checks them
  suggests.
- `def f(...) payable:` runs its body whatever ether it's sent (`call.value`).
  `def f(...): # not payable` reverts with no data when it's sent any,
  before anything else - `# not payable (invalid)`: runs an invalid opcode
  then (the check of old compilers).
- `const name = ...`: the function `name()` returns that data (read as a
  `return`'s), whatever it's called with - it's not payable.
- A param (`_param1`, or its name) is the word of calldata at its place: its
  type in the signature doesn't clean it. Where it's used cleaned, it's after
  the function checks it (`require _param1 == address(_param1)`), which
  proves the calldata held no more than that.
- The word of a bytes, a string or an array param `p` is where its data is,
  from the byte 4 on: `p.length` is the word of calldata at `4 + p`, `p[all]`
  the `p.length` bytes from `36 + p`. A word of a param that's a struct or
  an array of a fixed size is named after it: `desc.amount`, `amounts[2]`.
- `cd[a]` is the word of calldata at `a`, `calldata.size` the size of the
  calldata, `call.func_hash` its 4 first bytes in place (at the top of a
  word: `call.func_hash >> 224` is the selector).
- A struct in a signature is the types it's made of, `(uint8,bytes32,bytes32)
  sig`: the selector is the hash of the signature with them.
- An external call is `call x.f(...) with:` (CALL), `static call` (STATICCALL),
  `delegate` (DELEGATECALL) or `codecall` (CALLCODE), then `value v wei` (none:
  0), `gas g` and `args ...`: its data is the 4 bytes of its selector, then
  its params as a list of data. The selector is printed as the function it
  calls when that's known, `unknown1234abcd(?)` when it isn't (no params are
  made up), `mem[a len 4]` when it's those bytes of memory, or on a line of
  its own, `funct x` (the low 4 bytes of `x`). `call x with:` has no
  selector: `args` is all of its data, when it may have fewer than 4 bytes
  (`x.call(data)` with a param `data`, say) - no `args`: no data. The address
  called (`x` in `call x.f(...)`, `eth.balance(x)`...) is the low 160 bits
  of `x`, as for the EVM.
  `ext_call.success` is its result (`delegate.return_code`,
  `callcode.return_code` for the others), `ext_call.return_data[a len n]`
  what it returned (zeroes past it), `return_data.size` how much. What a call
  writes to memory is said after it: `mem[a len min(32, return_data.size)] =
  ext_call.return_data[0 len min(32, return_data.size)]`, the rest of the 32
  bytes it gave for it left as it was.
- `x = ecrecover(...) # precompiled` calls a precompiled contract
  (`ecrecover`, `sha256hash`, `identity`...) with that data: `x` is the
  first word it returns (0 if none) - a name that's no other's (`signer_`
  where a param or a storage variable is `signer`), `ecrecover.result` its
  success (`memcopy.success` for `identity`), and `return_data.size` and
  `ext_call.return_data` are what it returned.

## What the EVM gives

- `caller`, `tx.origin`, `call.value`, `this.address`, `tx.gasprice`,
  `gas_remaining` (GAS), `chainid`, `msize` (MSIZE: the bytes of memory used
  so far - by reads and writes the output doesn't all show);
  `block.number`, `block.timestamp`, `block.coinbase`, `block.difficulty`
  (PREVRANDAO), `block.gas_limit`, `block.basefee`, `block.blobbasefee`,
  `blobhash(i)`, `block.hash(n)` (0 but for the 256 blocks before this one).
- `code.data[a len n]` is the contract's own code, `call.data[a len n]` the
  calldata (zeroes past their ends).
- `transient[k]` is the word of the transient storage at `k`.
- `addmod(a, b, n)`, `mulmod(a, b, n)` are the EVM's (0 for an `n` of 0);
  `True` and `False` are 1 and 0.
- `create contract with v wei` (and its `code:`) and `create2 contract with v
  wei` (and its `salt:`) make a contract, `create.new_address` /
  `create2.new_address` its address (0 if it failed); `selfdestruct(x)` ends
  the call, sending the balance to `x`.

## Events

`log Transfer(address from=..., address indexed to=..., ...)`: the event of
that signature, `indexed` params in the topics, the others in the data in
order. An event not known is printed with its whole topic 0:
`log 0xddf2...b3ef: x, indexed y` (`log 0xddf2...b3ef:` with nothing else);
one whose params aren't what the log has (indexed otherwise: ERC-721's
`Approval` is ERC-20's with its three params indexed) with its signature:
`log Approval(address owner, address spender, uint256 value): indexed x,
indexed y, indexed z`. A log of no topics is `log x, y`, its data (`log` for
none).

## Storage

The `def storage:` header names the storage the functions use - `def
storage (vyper):` for Vyper, whose mappings hash the slot, then the key:

- `x is uint8 at storage 3 offset 8`: the bits 8 to 16 of the slot 3. `x`
  reads as that value; a read of fewer of its bits is a cast (`uint4(x)`)
  or a shift of it.
- `m is mapping of T at storage 5`: `m[k]` is at `keccak(k . 5)`, the word
  of the key then the word of the slot. A key of bytes (a range, a
  `concat(...)`, a string) is hashed as those bytes.
- `a is array of T at storage 6`: `a.length` is the word at the slot 6,
  `a[i]` is at `keccak(6) + i` - `keccak(6) + n * i` for a `struct of n
  slots`. Values of 8, 16, 32, 64 or 128 bits are packed: `256 / width` of
  them in a slot, from its lowest bits.
- `f is fixed array of T at storage 7`: `f[i]` is at `7 + i` (packed, and
  of structs, the same way).
- `b is bytes at storage 4`: `b` is the word at the slot 4 (a short bytes or
  string holds its data and twice its length there, a long one twice its
  length plus 1), `b[i]` its data word at `keccak(4) + i`, `b.length` its
  length, `b[all]` its `b.length` bytes (from the word at the slot for a
  short one, from `keccak(4)` on for a long one) - `return Array(len=
  b.length, data=b[all])` is the getter of a `bytes` or a `string`.
- `.field_N` of a struct is its value from the bit `N` on: the slot `N /
  256` after its first one, from the bit `N % 256` of it up to the top of
  the slot, unless a cast says how many bits.
- `stor[e]` is the slot `e` itself.

A name of the header is the one of the getter that returns it when there is
one, else `storN` (`storN_O` for the bits from `O` on of the slot `N`).

## SafeMath

The `def SafeMath:` header has the functions of checked arithmetic the
contract runs, as their code is: the checks that fail before the result is
used - those of OpenZeppelin's SafeMath, of ds-math (a library's internal
functions, solidity < 0.8), of the compiler (solidity's since 0.8, `#
solidity >= 0.8`: an overflow is `Panic(17)`, a division by zero
`Panic(18)`, `inc` and `dec` its `x + 1` and `x - 1`; the check of a
division of solidity < 0.8, `assert b`; Vyper's), a buggy check as it is.
There is a function for each other check, numbered (`add2`), and its
message is a parameter, `errorMessage`, where it fails with different ones.

`SafeMath.add(a, b)` is `a + b`, its checks passed - where the code has
them, and the result after them (in the condition of the next check, or
the next line that may fail): `SafeMath.sub(x, y, 'text')` reverts with
`'text'` where `y > x`. Checks whose result is used only later - after
another check, a call, in a branch - are left as they are: the call there
would fail after them.

## Recursive internal functions

An internal function that calls itself, or calls what calls it, isn't
inlined in its callers as the others are (its calls would unroll until the
decompiler stops): `_1, _2 = internal_0x1a2(a, b)` is a call of the code at
`0x1a2` with those words, `_1, _2` the words it returns (no `=`: none), the
storage and the memory as it may leave them. It's decompiled apart, after
the regular functions: `def internal_0x1a2(_param1, _param2):`, its params
the words it's called with, `return x, y` what it returns, `# recursive, a
call takes K of the 1024 words of the stack` what bounds the recursion -
the EVM's stack has 1024 words, a call past them fails. `cd[a]` in it is
the word of calldata at `a`: the params of the functions that call it
aren't its own. A recursion that never returns but by reverting is
unrolled as before.
