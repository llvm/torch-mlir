# node_import

Most of the tests in this directory test the importing of TorchScript
`torch::jit::Graph`'s.

However, TorchScript graphs don't really correspond directly to anything on
the MLIR side. They are a weird combination of a context, builder, and
function and just holds a `torch::jit::Block`. It is `torch::jit::Node`
and `torch::jit::Block` which form the recursive structure analogous to
MLIR's operation/region/block.

- `torch::jit::Node` == `mlir::Operation`,
- `torch::jit::Block` == `mlir::Region` containing single `mlir::Block`

Hence the name of this directory and the corresponding code in
node_importer.h/cpp.
