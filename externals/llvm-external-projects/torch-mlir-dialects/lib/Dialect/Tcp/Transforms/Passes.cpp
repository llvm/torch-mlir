#include "torch-mlir-dialects/Dialect/Tcp/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "torch-mlir-dialects/Dialect/Tcp/Transforms/FuseTcpOpsPass.h"
#include "torch-mlir-dialects/Dialect/Tcp/Transforms/IsolateGroupOpsPass.h"
#include <memory>

namespace {
#define GEN_PASS_REGISTRATION
#include "torch-mlir-dialects/Dialect/Tcp/Transforms/Passes.h.inc"
} // end namespace

void mlir::tcp::registerTcpPasses() { ::registerPasses(); }
