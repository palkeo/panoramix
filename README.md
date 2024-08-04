From 6c5ac012caff6dc5b7dfaa1eb1ee1adae2c7a15c Mon Sep 17 00:00:00 2001
From: ytrezq <ytrezq@sdf-eu.org>
Date: Fri, 8 May 2020 00:38:23 +0200
Subject: [PATCH 1/4] Small performance increase.
MIME-Version: 1.0
Content-Type: text/plain; charset=UTF-8
Content-Transfer-Encoding: 8bit

Test the most used instructions first and the less likely scenarios least. And if an opcode is having a specific value, it also means it can’t have other values at the same time.
Will win a little less than 2 minutes if the timeout limit along node count is raised accordingly.
---
 pano/vm.py | 195 +++++++++++++++++++++++++++--------------------------
 1 file changed, 98 insertions(+), 97 deletions(-)

diff --git a/pano/vm.py b/pano/vm.py
index 1d92ad43..0da47c72 100644
--- a/pano/vm.py
+++ b/pano/vm.py
@@ -438,7 +438,7 @@ def handle_jumps(self, trace, line, condition):
             trace.append(("jump", n))
             return trace
 
-        if op == "jumpi":
+        elif op == "jumpi":
             target = stack.pop()
             if_condition = simplify_bool(stack.pop())
 
@@ -492,19 +492,7 @@ def handle_jumps(self, trace, line, condition):
             logger.debug("jumpi -> if %s", trace[-1])
             return trace
 
-        if op == "selfdestruct":
-            trace.append(("selfdestruct", stack.pop(),))
-            return trace
-
-        if op in ["stop", "assert_fail", "invalid"]:
-            trace.append((op,))
-            return trace
-
-        if op == "UNKNOWN":
-            trace.append(("invalid",))
-            return trace
-
-        if op in ["return", "revert"]:
+        elif op in ["return", "revert"]:
             p = stack.pop()
             n = stack.pop()
 
@@ -516,6 +504,18 @@ def handle_jumps(self, trace, line, condition):
 
             return trace
 
+        elif op in ["stop", "assert_fail", "invalid"]:
+            trace.append((op,))
+            return trace
+
+        elif op == "UNKNOWN":
+            trace.append(("invalid",))
+            return trace
+
+        elif op == "selfdestruct":
+            trace.append(("selfdestruct", stack.pop(),))
+            return trace
+
         return None
 
     def apply_stack(self, ret, line):
@@ -550,16 +550,6 @@ def trace(exp, *format_args):
                 else:
                     trace("[{}] {} {}", line[0], C.asm(op), C.asm(str(line[2])))
 
-        assert op not in [
-            "jump",
-            "jumpi",
-            "revert",
-            "return",
-            "stop",
-            "jumpdest",
-            "UNKNOWN",
-        ]
-
         param = 0
         if len(line) > 2:
             param = line[2]
@@ -581,16 +571,37 @@ def trace(exp, *format_args):
         ]:
             stack.append(arithmetic.eval((op, stack.pop(), stack.pop(),)))
 
-        if op in ["mulmod", "addmod"]:
-            stack.append(("mulmod", stack.pop(), stack.pop(), stack.pop()))
+        elif op[:4] == "push":
+            stack.append(param)
+
+        elif op == "pop":
+            stack.pop()
 
-        if op == "mul":
+        elif op == "dup":
+            stack.dup(param)
+
+        elif op == "mul":
             stack.append(mul_op(stack.pop(), stack.pop()))
 
-        if op == "or":
+        elif op == "or":
             stack.append(or_op(stack.pop(), stack.pop()))
 
-        if op == "shl":
+        elif op == "add":
+            stack.append(add_op(stack.pop(), stack.pop()))
+
+        elif op == "sub":
+            left = stack.pop()
+            right = stack.pop()
+
+            if type(left) == int and type(right) == int:
+                stack.append(arithmetic.sub(left, right))
+            else:
+                stack.append(sub_op(left, right))
+
+        elif op in ["not", "iszero"]:
+            stack.append((op, stack.pop()))
+
+        elif op == "shl":
             off = stack.pop()
             exp = stack.pop()
             if all_concrete(off, exp):
@@ -598,7 +609,7 @@ def trace(exp, *format_args):
             else:
                 stack.append(mask_op(exp, shl=off))
 
-        if op == "shr":
+        elif op == "shr":
             off = stack.pop()
             exp = stack.pop()
             if all_concrete(off, exp):
@@ -606,7 +617,7 @@ def trace(exp, *format_args):
             else:
                 stack.append(mask_op(exp, offset=minus_op(off), shr=off))
 
-        if op == "sar":
+        elif op == "sar":
             off = stack.pop()
             exp = stack.pop()
             if all_concrete(off, exp):
@@ -625,20 +636,25 @@ def trace(exp, *format_args):
                 # FIXME: This won't give the right result...
                 stack.append(mask_op(exp, offset=minus_op(off), shr=off))
 
-        if op == "add":
-            stack.append(add_op(stack.pop(), stack.pop()))
+        elif op == "mstore":
+            memloc = stack.pop()
+            val = stack.pop()
+            trace(("setmem", ("range", memloc, 32), val,))
 
-        if op == "sub":
-            left = stack.pop()
-            right = stack.pop()
+        elif op == "msize":
+            self.counter += 1
+            vname = f"_{self.counter}"
+            trace(("setvar", vname, "msize"))
+            stack.append(("var", vname))
 
-            if type(left) == int and type(right) == int:
-                stack.append(arithmetic.sub(left, right))
-            else:
-                stack.append(sub_op(left, right))
+        elif op == "mload":
+            memloc = stack.pop()
+            loaded = mem_load(memloc)
 
-        elif op in ["not", "iszero"]:
-            stack.append((op, stack.pop()))
+            self.counter += 1
+            vname = f"_{self.counter}"
+            trace(("setvar", vname, ("mem", ("range", memloc, 32))))
+            stack.append(("var", vname))
 
         elif op == "sha3":
             p = stack.pop()
@@ -664,9 +680,6 @@ def trace(exp, *format_args):
             off = sub_op(256, to_bytes(num))
             stack.append(mask_op(val, 8, off, shr=off))
 
-        elif op == "selfbalance":
-            stack.append(("balance", "address",))
-
         elif op == "balance":
             addr = stack.pop()
             if opcode(addr) == "mask_shl" and addr[:4] == ("mask_shl", 160, 0, 0):
@@ -674,9 +687,30 @@ def trace(exp, *format_args):
             else:
                 stack.append(("balance", addr,))
 
+        elif op in [
+            "callvalue",
+            "caller",
+            "address",
+            "number",
+            "gas",
+            "origin",
+            "timestamp",
+            "chainid",
+            "difficulty",
+            "gasprice",
+            "coinbase",
+            "gaslimit",
+            "calldatasize",
+            "returndatasize",
+        ]:
+            stack.append(op)
+
         elif op == "swap":
             stack.swap(param)
 
+        elif op == "selfbalance":
+            stack.append(("balance", "address",))
+
         elif op[:3] == "log":
             p = stack.pop()
             s = stack.pop()
@@ -697,26 +731,6 @@ def trace(exp, *format_args):
             val = stack.pop()
             trace(("store", 256, 0, sloc, val))
 
-        elif op == "mload":
-            memloc = stack.pop()
-            loaded = mem_load(memloc)
-
-            self.counter += 1
-            vname = f"_{self.counter}"
-            trace(("setvar", vname, ("mem", ("range", memloc, 32))))
-            stack.append(("var", vname))
-
-        elif op == "mstore":
-            memloc = stack.pop()
-            val = stack.pop()
-            trace(("setmem", ("range", memloc, 32), val,))
-
-        elif op == "mstore8":
-            memloc = stack.pop()
-            val = stack.pop()
-
-            trace(("setmem", ("range", memloc, 8), val,))
-
         elif op == "extcodecopy":
             addr = stack.pop()
             mem_pos = stack.pop()
@@ -896,44 +910,31 @@ def trace(exp, *format_args):
 
             stack.append("create2.new_address")
 
-        elif op[:4] == "push":
-            stack.append(param)
+        elif op in ("extcodesize", "extcodehash", "blockhash"):
+            stack.append((op, stack.pop(),))
+
+        elif op in ["mulmod", "addmod"]:
+            stack.append(("mulmod", stack.pop(), stack.pop(), stack.pop()))
 
         elif op == "pc":
             stack.append(line[0])
 
-        elif op == "pop":
-            stack.pop()
-
-        elif op == "dup":
-            stack.dup(param)
-
-        elif op == "msize":
-            self.counter += 1
-            vname = f"_{self.counter}"
-            trace(("setvar", vname, "msize"))
-            stack.append(("var", vname))
+        elif op == "mstore8":
+            memloc = stack.pop()
+            val = stack.pop()
 
-        elif op in ("extcodesize", "extcodehash", "blockhash"):
-            stack.append((op, stack.pop(),))
+            trace(("setmem", ("range", memloc, 8), val,))
 
-        elif op in [
-            "callvalue",
-            "caller",
-            "address",
-            "number",
-            "gas",
-            "origin",
-            "timestamp",
-            "chainid",
-            "difficulty",
-            "gasprice",
-            "coinbase",
-            "gaslimit",
-            "calldatasize",
-            "returndatasize",
-        ]:
-            stack.append(op)
+        else:
+            assert op not in [
+                "jump",
+                "jumpi",
+                "revert",
+                "return",
+                "stop",
+                "jumpdest",
+                "UNKNOWN",
+            ]
 
         if stack.len() - previous_len != opcode_dict.stack_diffs[op]:
             logger.error("line: %s", line)

From 09e5bfee127c1b2e82df5116c2676b9ef357ac51 Mon Sep 17 00:00:00 2001
From: Tomasz Kolinko <kolinko@gmail.com>
Date: Mon, 8 Jun 2020 14:05:22 +0200
Subject: [PATCH 2/4] Update README.md

---
 README.md | 94 ++++++++++++++++++++++++++++++++++++++++++++++++++++---
 1 file changed, 90 insertions(+), 4 deletions(-)

diff --git a/README.md b/README.md
index b3d2b31d..9e802c85 100644
--- a/README.md
+++ b/README.md
@@ -1,6 +1,92 @@
-This is a fork of: https://github.com/eveem-org/panoramix.git
+## Installation:
 
-The goal of this fork is to maintain Panoramix in a decent shape, fix some crashes, implement missing opcodes...
+```
+git clone https://github.com/eveem-org/panoramix.git
+pip3 install -r requirements.txt
+```
 
-For now I only plan to implement very minor (but annoying fixes). If something is too complicated to understand or would require
-a non-negligible amount of time it won't be fixed.
+## Running:
+
+You *need* **python3.8** to run Panoramix. Yes, there was no way around it.
+
+```
+python3.8 panoramix.py address [func_name] [--verbose|--silent|--explain]
+```
+
+e.g.
+
+```
+python3.8 panoramix.py 0x06012c8cf97bead5deae237070f9587f8e7a266d
+```
+or
+```
+python3.8 panoramix.py kitties
+```
+
+Output goes to two places:
+- `console`
+- ***`cache_pan/`*** directory - .pan, .json, .asm files
+
+If you want to see how Panoramix works under the hood, try the `--explain` mode:
+
+```
+python3.8 panoramix.py kitties paused --explain
+python3.8 panoramix.py kitties pause --explain
+python3.8 panoramix.py kitties tokenMetadata --explain
+```
+
+### Optional parameters:
+
+func_name -- name of the function to decompile (note: storage names won't be discovered in this mode)
+--verbose -- prints out the assembly and stack as well as regular functions, a good way to try it out is
+by running 'python panoramix.py kitties pause --verbose' - it's a simple function
+
+There are more parameters as well. You can find what they do in panoramix.py.
+
+### Address shortcuts
+Some contract addresses, which are good for testing, have shortcuts, e.g. you can run
+'python panoramix.py kitties' instead of 'python3 panoramix.py 0x06012c8cf97bead5deae237070f9587f8e7a266d'.
+
+See panoramix.py for the list of shortcuts, feel free to add your own.
+
+## Directories & Files
+
+### Code:
+- core - modules for doing abstract/symbolic operations
+- pano - the proper decompiler
+- utils - various helper modules
+- tilde - the library for handling pattern matching in python3.8
+
+### Data:
+- cache_code - cached bytecodes
+- cache_pan - cached decompilation outputs
+- cache_pabi - cached auto-generated p-abi files
+- supplement.db - sqlite3 database of function definitions
+- supp2.db - a lightweight variant o the above
+
+Cache directories are split into subdirectories, so the filesystem doesn't break down with large amounts
+of cached contracts (important when running bulk_decompile on all 2.2M contracts on the chain)
+
+All of the above generated after the first run.
+
+## Utilities
+bulk_decompile.py - batch-decompiles contracts, with multi-processing support
+bulk_compare.py - decompiles a set of test contracts, fetches the current decompiled from Eveem, and prepares two files, so you can diff them and see what changes were made
+
+## Why **python3.8** and **Tilde**
+Panoramix uses a ton of pattern matching operations, and python doesn't support those as a language.
+
+There are some pattern-matching libraries for older python versions, but none of them seemed good enough.
+Because of that, I built Tilde, which is a language extension adding a new operator.
+
+Tilde replaces '~' pattern matching operator with a series of ':=' operators underneath.
+Because of that, python3.8 is a must.
+
+Believe me, I spent a lot of time looking for some other way to make pattern matching readable.
+Nothing was close to this good.
+
+But if you manage to figure out a way to do it without Tilde (and maintain readability), I'll gladly accept a PR :)
+
+# How Panoramix works
+
+See the source code comments, starting with panoramix.py. Also, those slides[tbd].

From a75744cc2e7d1ad4cb3514d172d1872233f645fd Mon Sep 17 00:00:00 2001
From: Tomasz Kolinko <kolinko@gmail.com>
Date: Mon, 8 Jun 2020 14:13:24 +0200
Subject: [PATCH 3/4] Update README.md

---
 README.md | 7 +++++++
 1 file changed, 7 insertions(+)

diff --git a/README.md b/README.md
index 9e802c85..a31e279d 100644
--- a/README.md
+++ b/README.md
@@ -1,3 +1,10 @@
+## Important
+
+Palkeo is maintaining a more up to date for of Panoramix. Be sure to check it out:
+
+https://github.com/palkeo/panoramix
+
+
 ## Installation:
 
 ```

From baa4cdeff9565ff95a86f282d2c4aa99c0da2013 Mon Sep 17 00:00:00 2001
From: Recep <148443421+zxramozx@users.noreply.github.com>
Date: Sun, 4 Aug 2024 04:25:55 +0300
Subject: [PATCH 4/4] Update README.md

---
 README.md | 4 ++--
 1 file changed, 2 insertions(+), 2 deletions(-)

diff --git a/README.md b/README.md
index a31e279d..9a686ff0 100644
--- a/README.md
+++ b/README.md
@@ -23,7 +23,7 @@ python3.8 panoramix.py address [func_name] [--verbose|--silent|--explain]
 e.g.
 
 ```
-python3.8 panoramix.py 0x06012c8cf97bead5deae237070f9587f8e7a266d
+python3.8 panoramix.py 0xbd0B121a1D68Ed7AEE8bD1C988e923EB8528b6EA
 ```
 or
 ```
@@ -52,7 +52,7 @@ There are more parameters as well. You can find what they do in panoramix.py.
 
 ### Address shortcuts
 Some contract addresses, which are good for testing, have shortcuts, e.g. you can run
-'python panoramix.py kitties' instead of 'python3 panoramix.py 0x06012c8cf97bead5deae237070f9587f8e7a266d'.
+'python panoramix.py kitties' instead of 'python3 panoramix.py 0xbd0B121a1D68Ed7AEE8bD1C988e923EB8528b6EA'.
 
 See panoramix.py for the list of shortcuts, feel free to add your own.
 
