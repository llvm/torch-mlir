# Torch-MLIR Shape Library Infrastructure

## Overview

The Torch-MLIR project has an infrastructure for maintaining a library of shape
functions for different Torch operators. These shape functions are fully
executable specifications of the shape functions for each operator and can be
authored and tested from Python for convenience. These are then brought into the
compiler and can be manipulated / transformed for various purposes.
Additionally, this effort is synergistic with upstream PyTorch efforts to
maintain a library of shape functions.

The two main use cases are:

- Shape refinement / shape inference. The `torch-shape-refinement-pipeline` pass
  pipeline orchestrates a series of passes that use the available shape information in the program to further refine the types in the program.

- Error guard insertion for backends (Not Yet Implemented). The executable shape
  functions can include error guards / assertions that abort the program in case
  of invalid input (such as a matmul with a mismatching contracting dimension).

## Architecture

Shape functions are defined as TorchScript-able Python functions in
`python/torch_mlir/dialects/torch/importer/jit_ir/build_tools/shape_lib_gen.py`.
The signatures of the shape functions are systematically derived from Torch JIT
operator registry (mainly by replacing `Tensor` with `List[int]` in the operator
signatures). Most shape functions are expected to reuse the upstream helper
functions in
`python/torch_mlir/dialects/torch/importer/jit_ir/build_tools/upstream_shape_helpers.py`.

The `build_tools/update_shape_lib.sh` script invokes `shape_lib_gen.py` to
generate an MLIR module containing the shape functions, which is currently
embedded as a string literal in `lib/Dialect/Torch/Transforms/ShapeLibrary.cpp`.

The function `StringRef mlir::torch::Torch::getShapeLibrary()` is available for
use inside the compiler any time that the shape library is needed.

## Shape Refinement Pipeline Architecture

One of the main services that Torch-MLIR provides for backends is to normalize
all Torch frontends into a common form which includes tensor shapes that are as
precise as possible. This alleviates the need for backends to solve this problem
themselves. This process of shape refinement is accomplished in Torch-MLIR
through a pipeline of passes which uses the shape library combined with abstract
interpretation of the shape functions to calculate shapes that are as precise as
possible.

The pipeline works as follows:

1. Shape calculations are reified. The `torch-reify-shape-calculations` reifies
   (i.e., materializes into the IR) the shape functions for each op with a shape
   function in the shape library. To do this, it wraps those ops in a
   `torch.shape.calculate` op, which has two regions: 1) a body with the op
   itself, and 2) the shape calculation, which calculates the shapes of the
   tensors yielded by the body.

2. Simplifying the shape functions and propagating the shapes. After the shape
   functions are reified, we then attempt to "optimize hard enough" until the
   shapes yielded by the shape calculation regions become obvious in the IR.
   Those shapes are propagated through the IR, which usually reveals more
   opportunities for simplification.

   a. After reification, the shape functions are just a loose collection of
   functions, which are difficult to analyze. The first step is to inline them.

   b. After inlining, the `torch-simplify-shape-calculations` pass is used to
   simplify the shape calculations. This pass brings in a number of targeted
   canonicalization patterns and folds, along with a few specific patterns such
   as unrolling fixed-trip-count loops and abstractly interpreting list
   operations (an example is turning a series of "append" operations into a list
   literal). This pass also looks at the values yielded by the shape calculation
   regions, and if the resulting shape can be deduced by looking at the IR (for
   example, the shape is the list literal `[1, 2, 3]`), it will refine the types
   of the `torch.shape.calculate` op. This usually yields more opportunities for
   simplification. This process runs to a fixed-point.

3. Dropping the shape calculations. Once all the types in the program have been
   refined as much as possible, the ops that were originally wrapped in
   `torch.shape.calculate` are unwrapped by the `torch-drop-shape-calculations`
   pass which drops the reified shape calculations, leaving behind the shape-refined program.

Inferring precise shape often is needed for correctness by backends. That said,
requiring "optimizing hard enough" for correctness is usually considered quite
brittle in a compiler flow. In this case, the saving grace is that we are only
optimizing the shape functions, which are authored by compiler developers (not
users), and thus there is some give-and-take in terms of understanding the
optimizable constructs while authoring the shape functions, or improving the
optimizations to enable easier authoring. Some brittleness is likely to escape
to users, unfortunately, since there will always be situations where, for
example, a statically shaped program allows the shape functions to be simplified
to a greater extent than in a dynamically shaped program (for example, if the
shape function checks "is this dimension of size 1"). We hope that this is
minimal.

## Adding to the shape library

See [Adding a Shape Function](adding_a_shape_function.md) for details on how to
add a shpae function for an operator.

## Rationale

### Use of full operator signatures

The use of the full operator signature such as
`def aten〇add〇Tensor(self: List[int], other: List[int], alpha: float = 1) -> List[int]:`
for defining shape functions is somewhat verbose and repetitive, especially when
there are multiple identical shape functions. Upstream uses a map with key-value
pairs like `"aten.add.Tensor": upstream_shape_helpers.broadcast`, which is more
compact and less repetitive in some ways (upstream also allows trailing
arguments beyond those accepted by the shape function to be ignored, allowing
further deduplication). The decision to do it the more verbose way in Torch-MLIR
was based on the following goals:

- To make the system very easy to debug and test.

- To make the system maximally consistent between shape functions that are
  implemented with the upstream shape helpers and the ones that are manually
  written, which are still a fairly large and non-trivial set.

- To make it as mechanical as possible to add a new shape function.

## TODO

We should develop a workflow with upstream to push our manually-authored shape
functions to live and be tested there. We should also find a way to share with
upstream the mapping between operators and their shape functions. We will be
able to simplify this infrastructure quite a bit once that happens.
