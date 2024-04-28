# Torch-MLIR Abstract Interpretation Library Infrastructure

## Overview

The Torch-MLIR project has an infrastructure for maintaining a library of
calculation functions for different Torch operators, which supply extra
information such as result dtypes and shapes as well as decompositions. These
functions are fully executable specifications of the shape/dtype/decomposition
functions for each operator and can be authored and tested from Python for
convenience. These are then brought into the compiler and can be manipulated /
transformed for various purposes.  Additionally, in the case of shape functions,
this effort is synergistic with upstream PyTorch efforts to maintain a library
of shape functions.

The two main use cases are:

- Refinement / inference. The `torch-shape-refinement-pipeline` and
  `torch-dtype-refinement-pipeline` pass pipelines orchestrate a series of
  passes that use the available information in the program to further refine the
  types in the program.

- Error guard insertion for backends (Not Yet Implemented). The executable
  functions can include error guards / assertions that abort the program in case
  of invalid input (such as a matmul with a mismatching contracting dimension).

## Architecture

Functions are defined as TorchScript-able Python functions in
`python/torch_mlir/jit_ir_importer/build_tools/abstract_interp_lib_gen.py`.
The signatures of the functions are systematically derived from Torch JIT
operator registry. Most shape functions are expected to reuse the upstream
helper functions
[`torch/jit/_shape_functions.py`](https://github.com/pytorch/pytorch/blob/279634f384662b7c3a9f8bf7ccc3a6afd2f05657/torch/jit/_shape_functions.py#L1),
and any new shape functions should be added there.

The `build_tools/update_abstract_interp_lib.sh` script invokes
`abstract_interp_lib_gen.py` to generate an MLIR module containing the functions,
which is currently embedded as a string literal in
`lib/Dialect/Torch/Transforms/AbstractInterpLibrary.cpp`.

The function `StringRef mlir::torch::Torch::getAbstractInterpLibrary()` is
available for use inside the compiler any time that the library is needed.

## Shape and Dtype Refinement Pipeline Architecture

One of the main services that Torch-MLIR provides for backends is to normalize
all Torch frontends into a common form which includes tensor shapes and dtypes
that are as precise as possible. This alleviates the need for backends to solve
this problem themselves. This process of shape and dtype refinement is
accomplished in Torch-MLIR through a pipeline of passes which uses the abstract
interpretation library combined with abstract interpretation of the calculation
functions to calculate shapes and dtypes that are as precise as possible.

The pipeline works as follows:

1. Calculations are reified. The `torch-reify-shape-calculations` and
   `torch-reify-dtype-calculations` passes reify (i.e., materializes into the
   IR) the functions for each op with a function in the calculation library. To
   do this, the passes wrap those ops in a `torch.shape.calculate` or
   `torch.dtype.calculate` op, respectively, which has two regions: 1) a body
   with the op itself, and 2) the shape or dtype calculation, which calculates
   the shapes or dtypes of the tensors yielded by the body.

2. Simplifying the functions and propagating the shapes and dtypes. After the
   functions are reified, we then attempt to "optimize hard enough" until the
   shapes and dtypes yielded by the calculation regions become obvious in the IR.
   Those results are propagated through the IR, which usually reveals more
   opportunities for simplification.

   a. After reification, the functions are just a loose collection of
   functions, which are difficult to analyze. The first step is to inline them.

   b. After inlining, the `torch-simplify-shape-calculations` and
   `torch-simplify-dtype-calculations` passes are used to simplify the
   calculations. These passes bring in a number of targeted canonicalization
   patterns and folds, along with a few specific patterns such as unrolling
   fixed-trip-count loops and abstractly interpreting list operations (an
   example is turning a series of "append" operations into a list
   literal). These passes also look at the values yielded by the calculation
   regions, and if the resulting shape or dtype can be deduced by looking at the
   IR (for example, the shape is the list literal `[1, 2, 3]`), it will refine
   the types of the `torch.shape.calculate` and `torch.dtype.calculate`
   ops. This usually yields more opportunities for simplification. This process
   runs to a fixed-point.

3. Dropping the calculations. Once all the types in the program have been
   refined as much as possible, the ops that were originally wrapped in
   `torch.shape.calculate` and `torch.dtype.calculate` are unwrapped by the
   `torch-drop-abstract-interp-calculations` pass which drops the reified
   calculations, leaving behind the shape and dtype refined program.

Inferring precise shapes and dtypes often is needed for correctness by
backends. That said, requiring "optimizing hard enough" for correctness is
usually considered quite brittle in a compiler flow. In this case, the saving
grace is that we are only optimizing the functions, which are authored by
compiler developers (not users), and thus there is some give-and-take in terms
of understanding the optimizable constructs while authoring the functions, or
improving the optimizations to enable easier authoring. Some brittleness is
likely to escape to users, unfortunately, since there will always be situations
where, for example, a statically shaped program allows the shape functions to be
simplified to a greater extent than in a dynamically shaped program (for
example, if the shape function checks "is this dimension of size 1"). We hope
that this is minimal.

## Adding to the abstract interpretation library

See [Adding Abstract Interpretation Functions](adding_abstract_interpretation_functions.md)
for details on how to add a shape and dtype function for an operator.

## Rationale

### Use of full operator signatures

The use of the full operator signature such as
`def aten〇add〇Tensor(self: List[int], other: List[int], alpha: float = 1) -> List[int]:`
for defining calculation functions is somewhat verbose and repetitive, especially when
there are multiple identical functions. Upstream uses a map with key-value
pairs like `"aten.add.Tensor": upstream_shape_functions.broadcast`, which is
more compact and less repetitive in some ways (upstream also allows trailing
arguments beyond those accepted by the shape function to be ignored, allowing
further deduplication). The decision to do it the more verbose way in Torch-MLIR
was based on the following goals:

- To make the system very easy to debug and test.

- To make the system maximally consistent between functions that are
  implemented with the upstream shape helpers and the ones that are manually
  written, which are still a fairly large and non-trivial set.

- To make it as mechanical as possible to add a new function.
