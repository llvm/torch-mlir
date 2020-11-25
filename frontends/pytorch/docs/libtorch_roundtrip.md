# Design sketch: libtorch code generation round-trip

It has been brought up a couple of times that having a dynamic fallback to
libtorch for kernel calls that the compiler does not recognize could be
advantageous. This is a sketch of how such a facility could work.

## Op background

When programs are imported from Torch (either via acap/driver capture or
from TorchScript), kernel calls are mapped to a `torch.kernel_call` op, which
it is useful to visualize:

```mlir
%0 = torch.kernel_call "aten::mm" %arg0, %arg1 :
    (!numpy.ndarray<[2,3]:f32>, !numpy.ndarray<[3,4]:f32>) ->
      !numpy.ndarray<[2,4]:f32>
    {
      sigArgTypes = ["Tensor", "Tensor"],
      sigIsMutable = false,
      sigIsVararg = false,
      sigIsVarret = false,
      sigRetTypes = ["Tensor"]
    }
```

A couple of things to note at this level:

* Tensor operands/results are all represented by mutable `ndarray` types.
* The kernel call name ("aten::mm" above) is the `c10::OperatorName`.
* `sigArgTypes` and `sigRetTypes` correspond to the rest of a signature.
  Together with the kernel name, it is sufficient to find a precise `OpHandle`
  that can be used for making calls.
* The `torch.kernel_call` implements the `TorchKernelOpInterface` which
  provides structured access to this metadata.

From here, one typically uses the pass `aten-recognize-kernels` to promote
`torch.kernel_call` ops that the compiler has concretely modeled into
corresponding `aten` dialect ops. Here is an example of a function containing
the above, with aten kernels recognized:

```mlir
  func @mm(%arg0: !numpy.ndarray<[2,3]:f32>, %arg1: !numpy.ndarray<[3,4]:f32>) -> !numpy.ndarray<[2,4]:f32> {
    %0 = numpy.copy_to_tensor %arg0 : (!numpy.ndarray<[2,3]:f32>) -> tensor<2x3xf32>
    %1 = numpy.copy_to_tensor %arg1 : (!numpy.ndarray<[3,4]:f32>) -> tensor<3x4xf32>
    %2 = "aten.mm"(%0, %1) : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    %3 = numpy.create_array_from_tensor %2 : (tensor<2x4xf32>) -> !numpy.ndarray<[2,4]:f32>
    return %3 : !numpy.ndarray<[2,4]:f32>
  }
```

A few things to note about this form:

* These recognized kernels are generated from the `torch_signature_ods_gen.py`
  script, which imposes some mapping policy on them.
* Most kernels are aggressively converted to operate on ssa tensor values via
  `copy_to_tensor`/`create_array_from_tensor` ops, making the majority of ops
  in the `aten` dialect which are purely functional operate just on value
  semantic types.
* The metadata is stripped off of the originating `torch.kernel_call` but each
  `aten` op implements `TorchKernelOpInterface`, giving it access to the kernel
  name and a signature matching its operands/results of a Torch kernel that
  implements the computation.
* There is some information loss here but there should be enough retained to
  perform the correct calculation, if not execute it exactly as the original
  program specified (i.e. `out=` and other "ergonomic" aliases will be
  rewritten into dedicated stores, etc).

## General fallback flow

The most straight-forward way to execute a `torch.kernel_call` or `aten` op
supporting the `TorchKernelOpInterface` would be to rewrite it into code that
invokes the ATen boxed dispatch mechanism:

* Looking up a corresponding kernel based on a signature known at compile time
  (constructed from `TorchKernelOpInterface` metadata).
* For each operand, scribble into a `Stack` (of `IValue`) list.
* Invoking `c10::Dispatcher::callBoxed()` with the stack.
* Marshaling returned results back out of the return `Stack`.
* Performing error and type constraint checking.

The "inside" of such a dispatch function would be somewhat "switchy" but is
not all that complicated.

## Runtime library

`libtorch` on its own is not particularly amenable to be invoked from such
a low level. It would be better if there were a shared library that provided
the above facility as simple C functions that the compiler could emit calls
to. It would then be trivial to load/link this shared library in for JIT'ing,
AOT compilation, etc.

Example:

```c
/// Looks up a Torch op given a signature.
void *refbackFindTorchOp(const char *signature);

/// Creates a 'call' struct from an op returned by `refbackFindTorchOp`.
/// Must be de-allocated via refbackDestroyTorchCall() when done.
void *refbackCreateTorchCall(void *torchOp);

/// Adds IValues to the call stack.
void refbackTorchCallAddTensor(void *call, void *data, int64_t *sizes, int rank);
void refbackTorchCallAddScalar(void *call, int64_t scalar);
// ...

/// Invokes the kernel.
/// After invocation, results can be read out with below methods.
bool refbackTorchInvoke(void *call);

/// Gets IValues from the result stack.
bool refbackTorchGetTensor(void *call, size_t index, void **data, int64_t **sizes, int *rank);
bool refbackTorchGetScalar(void *call, size_t index, int64_t *scalar);

/// Frees any resources associated with the call.
void refbackTorchCallDestroy(void *call);
```

## Generating code

A pass could be written to transform ops implementing `TorchKernelOpInterface`
into `llvm` calls into the above functions. Details will be a bit thick and
depend on precise representations, but it should be fully generic. It should
be possible to prototype the whole thing with nothing but command line tools
and the existing `torch_mlir` paths for extracting programs.

## Code location recommendations:

* C-runtime library: `frontends/pytorch/csrc/kernelcrt`
* Code generation pass: `include/npcomp/Dialects/Torch/Transforms/TorchKernelToLLVMPass.cpp`

## Gotchas

This facility should work well for Torch kernels that are wholly unknown to
the compiler. However, kernels that the compiler fails to lower completely (i.e.
due to some unsupported, and unknown at the outset dynamism) way end up as
`tcf` ops or others that cannot be natively lowered via the
`TorchKernelOpInterface` facility. We can deal with this phase ordering in
a couple of ways:

* When converting into `tcf` be more precise about when certain dynamic
  constructs are wholly unsupported. Not likely to scale really well unless
  if just being used as a stop-gap. In that case, possibly having a pass
  early that marks ops to not lower because we know we want to retain them
  at the higher level may be fine.
* Treat `aten` as both a source and a target dialect for `tcf`: implement
  lowerings *to* `aten` that run after the rest of `tcf` has been lowered.
* Implement `TorchKernelOpInterface` on the `tcf` ops (or have some other
  interface for mapping them back).

