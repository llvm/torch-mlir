Utilities for compiling and running on the npcomp runtime with a JIT.

The runtime itself lives in {include,lib}/runtime/, but since it is totally
firewalled from the compiler codebase, it presents a fairly bare-bones
interface (e.g. it doesn't use libSupport, can't use LLVM's JIT interfaces,
etc.).

The interface provided in this directory uses standard LLVM conventions and
freely relies on libSupport, JIT utilities, etc.
