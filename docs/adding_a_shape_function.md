# Adding a Shape Function

## Overview

As part of adding support for a Torch operator in Torch-MLIR, it is usually
necessary to define a shape function so that the compiler can infer the shapes
of result tensors for the operator. We use the [shape library](shape_lib.md) for this process.

## Step-by-step guide

We will use the example of adding support for the `torch.aten.tanh` op.

1. First, you need to find the shape function signature for the operator you are
   implementing a shape function for. This can be found in
   `include/torch-mlir/Dialect/Torch/IR/JITOperatorRegistryDump.txt` generated
   by the `build_tools/update_torch_ods.sh` script. That file is the "rosetta
   stone" that allows translating between e.g. `torch.aten.tanh`, `AtenTanhOp`,
   and the shape function signature
   `def aten〇tanh(self: List[int]) -> List[int]:`. Note the use of `〇` as a
   separator since `.` or `::` aren't legal in a Python identifier.

2. Paste the shape function signature into `shape_lib_gen.py` in an appropriate
   place (ideally near other functions with a similar shape function). Note that
   `shape_lib_gen.py` will check that this signature is verbatim identical with
   the one given in `JITOperatorRegistryDump.txt` -- this ensures that the shape
   functions don't get outdated if Torch changes an operator signature.

3. Fill in the body of the shape function. Ideally this will just be a call into
   a helper function from
   [`torch/jit/_shape_functions.py`](https://github.com/pytorch/pytorch/blob/279634f384662b7c3a9f8bf7ccc3a6afd2f05657/torch/jit/_shape_functions.py#L1).
   But in general, you will need to write the shape function and test it (see
   the comments about "Shape function testing infrastructure" in
   `shape_lib_gen.py`). New shape functions should be added upstream following
   the example of [this PR](https://github.com/pytorch/pytorch/pull/76889),
   though it can be useful to iterate locally in `shape_lib_gen.py` first.

4. Re-run the `build_tools/update_shape_lib.sh` script to update the shape
   library. After this step happens, ideally everything "just works" and the
   shape is now correctly inferred for the operator.

## When things go wrong

It is possible that the shape refinement pipeline (see
[Shape Refinement Pipeline Architecture](shape_lib.md#shape-refinement-pipeline-architecture))
is not able to infer the shape of a tensor with a given shape function. This
usually means that there is something about the shape function which the
optimizations in `torch-simplify-shape-functions`
(`lib/Dialect/Torch/Transforms/SimplifyShapeCalculations.cpp`) cannot handle.

To debug this, the overall goal is to pinpoint the IR construct that is not
being simplified. This is usually accomplished by a combination of looking at
the Python code for the shape function and the IR dumps. The best IR dump to
look at varies, but frequently the IR dump right before `DropShapeCalculations`
is the most useful, because it has already been simplified as much as possible,
making it is easy to see what is blocking further simplification. Examples of
issues you might see:

- You might find that there is a loop with a non-constant trip count, but based
  on your understanding of the shape function, you would expect it to be
  simplified to a constant trip count -- you can then look at the trip count
  calculation and see if there is a missing fold or canonicalization.

- You might find that there is a list operation that is not currently understood
  by the optimizations. You can then teach the optimizations about that
  operation.

- You might find that there is an `Optional` value that you would expect to be
  resolved to either a concrete value or `None`. You can then look at the calculation that produces the optional value and see what folds or canonicalizations are missing.

See [this video](https://www.youtube.com/watch?v=E5epCJOtrf8) for general
guidance on debugging Torch-MLIR.

As a last resort, you can rewrite the shape function using constructs that
`torch-simplify-shape-functions` can handle (look at other shape functions for
examples, sometimes it requires writing things a little awkwardly).
