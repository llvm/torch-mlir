#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tcp {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createTcpFuseElementwiseOpsPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createTcpIsolateGroupOpsPass();

/// Registers all Tcp related passes.
void registerTcpPasses();

} // namespace mlir::tcp
