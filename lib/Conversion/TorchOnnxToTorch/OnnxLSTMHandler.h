#include "torch-mlir/Conversion/TorchOnnxToTorch/Patterns.h"
#include "torch-mlir/Conversion/TorchOnnxToTorch/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

// debug
#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "torch-onnx-to-torch-patterns"
namespace mlir::torch::onnx_c {

LogicalResult OnnxLSTMHandler(OpBinder binder,
                              ConversionPatternRewriter &rewriter);

} // namespace mlir::torch::onnx_c