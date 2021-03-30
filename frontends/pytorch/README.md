# NPComp - PyTorch frontend integration

This directory contains optional components for interfacing PyTorch to NPComp.
Integration is targeted at multiple levels:

* Via program capture with a ATen pseudo-device.
* Via IR-level integration with PyTorch (via tracing or scripting interfaces).
* Interfaces to facilitate checking against reference implementations and
  verification.

In all situations, the target dialects are maintained in the outer project,
along with their lowerings to common intermediate dialects and backends. This
directory should be purely about interfacing with the PyTorch/LibTorch
components for extracting and executing programs.

The code in this directory is intended to integrate tightly with pytorch, and
follows the code style for pytorch.  See the [overall documentation for
frontends](../README.md) for further details about code layout and integration
philosophy.  In particular, this directory exists to provide a working
frontend to an MLIR based pytorch compilation flow and is not intended to be
contributed to the LLVM monorepo. If the project is successful, it makes more
sense to either break it out as an independent project that depends on
LLVM/MLIR/npcomp or contribute it upstream to PyTorch. However, as it will be
quite some time before the components are in a state to support such a
dependency, it is being carried in-tree in the interim.

### Program capture with a ATen dispatch capture.

Integration with a pseudo-device is typified by code like the following:

```
import torch
import torch_mlir

lhs = torch.rand(2, 3)
rhs = torch.rand(3, 4)

mb = torch_mlir.ModuleBuilder()
with mb.capture_function("mm", [lhs, rhs]) as f:
  result = torch.mm(lhs, rhs)
  f.returns([result])

mb.module.operation.print()
```

All operations that happen under the `mb.capture_function` context manager are
intercepted via PyTorch's
[dispatcher](http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/),
and an IR graph is constructed into the module held by the ModuleBuilder.

This technique has several advantages and disadvantages. For training use
cases, this technique generates a backward path automatically using the same
method that pytorch natively uses. The resulting graph also tends to be
simpler, since it will not reflect conditionals in the original python
code. Lastly, it is natural if MLIR is being used as a frontend target for an
actual device of some sort.  In this case, the MLIR could go through a
device-specific lowering path and the resulting code run on a device.
The implementation of this technique is largely modeled after `pytorch/xla`.
