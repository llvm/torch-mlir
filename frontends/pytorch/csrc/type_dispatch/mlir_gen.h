//===- mlir_gen.h -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/MLIRContext.h"

#include "ir.h"

namespace torch_mlir {

/// This class generates MLIR from a pytorch graph
class MLIRGen {

public:
  MLIRGen(mlir::MLIRContext &context);

  // Generate an MLIR model that computes the given outputs.
  std::tuple<mlir::OwningModuleRef, std::vector<at::Tensor>>
  genModule(std::vector<ir::Value> &v);

private:
  mlir::Value genValue(const ir::Value &v);

  void genParameters(const ir::Value &v, std::set<ir::Value> &visited);

  mlir::FuncOp genFunction(std::vector<ir::Value> &v);

  bool declareSymbol(const ir::Value &irValue, mlir::Value mlirValue);

private:
  mlir::MLIRContext &context;
  mlir::OwningModuleRef module;
  std::unique_ptr<mlir::OpBuilder> builder;
  std::map<const ir::Value, mlir::Value> symbolTable;
  std::map<const ir::NodePtr, mlir::Operation *> opTable;
  std::vector<ir::Value> parameters;
  std::vector<at::Tensor> arguments;
};

} // namespace torch_mlir
