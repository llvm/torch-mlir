// For writing an extension like this one, see:
// https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html

#include <torch/script.h> // One-stop header for PyTorch

torch::Tensor identity(torch::Tensor t) {
  // Do literally nothing.
  return t;
}

TORCH_LIBRARY(_torch_mlir_custom_op_example, m) {
  m.def("identity(Tensor t) -> Tensor");
  m.impl("identity", &identity);
}
