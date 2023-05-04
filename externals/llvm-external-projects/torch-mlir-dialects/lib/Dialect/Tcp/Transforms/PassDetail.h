#pragma once

#include "mlir/Pass/Pass.h"

#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpOps.h"

namespace mlir::tcp {

using namespace mlir;
#define GEN_PASS_CLASSES
#include "torch-mlir-dialects/Dialect/Tcp/Transforms/Passes.h.inc"

} // end namespace mlir::tcp
