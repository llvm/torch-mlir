//===- mlir_gen.cpp ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Verifier.h"

#include "llvm/Support/Debug.h"

#include "ATen/ArrayRef.h"
namespace at {
template <typename T> using ArrayRef = c10::ArrayRef<T>;
}
#include "ATen/Tensor.h"

#include "ir.h"
#include "mlir_gen.h"

#include <set>
#include <vector>

#define DEBUG_TYPE "torch_mlir"

namespace torch_mlir {

std::tuple<mlir::OwningModuleRef, std::vector<at::Tensor>>
MLIRGen::genModule(std::vector<ir::Value> &v) {
  // the module
  module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));

  auto fn = genFunction(v);
  if (fn) {
    module->push_back(fn);
    if (failed(mlir::verify(*module))) {
      emitError(mlir::UnknownLoc::get(&context), "module verification error");
    }
  }
  return std::make_tuple(std::move(module), arguments);
}

mlir::Value MLIRGen::genValue(const ir::Value &v) {

  if (symbolTable.count(v))
    return symbolTable[v];

  LLVM_DEBUG(llvm::dbgs() << "genValue node: " << v.node->op() << "\n");

  ir::NodePtr node = v.node;
  auto loc = mlir::UnknownLoc::get(&context);

  for (auto &operand : node->operands())
    genValue(operand);

  mlir::Value mlirValue = nullptr;
  if (opTable.count(v.node)) {
    mlirValue = opTable[v.node]->getResult(v.index);
  } else {
    mlir::Operation *mlirOp = node->genMLIR(builder, context, symbolTable);
    opTable.insert({v.node, mlirOp});
    assert(mlirOp && "failed to generate mlir op");
    mlirValue = mlirOp->getResult(v.index);
  }

  declareSymbol(v, mlirValue);

  return mlirValue;
}

// generate function parameters for the IR rooted at v
void MLIRGen::genParameters(const ir::Value &v, std::set<ir::Value> &visited) {
  ir::NodePtr node = v.node;
  if (visited.count(v))
    return;
  visited.insert(v);
  for (const ir::Value &operand : node->operands()) {
    // if the operand is a leaf
    if (operand.node->op() == ir::OpKind::Get("aten::torch_data")) {
      parameters.push_back(operand);
    } else {
      genParameters(operand, visited);
    }
  }
}

mlir::FuncOp MLIRGen::genFunction(std::vector<ir::Value> &vs) {

  auto loc = mlir::UnknownLoc::get(&context);

  auto gen_tensor_ty = [&](const ir::Value &v) {
    auto shape = v.sizes();
    auto tdn = dynamic_cast<ir::TorchDataNode *>(v.node.get());
    mlir::Type elemTy;
    if (tdn) {
      auto dtype = tdn->tensor().dtype();
      if (dtype == at::kFloat)
        elemTy = mlir::FloatType::getF32(&context);
      else if (dtype == at::kDouble)
        elemTy = mlir::FloatType::getF64(&context);
      else if (dtype == at::kLong)
        elemTy = mlir::IntegerType::get(64, &context);
      else if (dtype == at::kInt)
        elemTy = mlir::IntegerType::get(32, &context);
      else if (dtype == at::kShort)
        elemTy = mlir::IntegerType::get(16, &context);
      else if (dtype == at::kChar || dtype == at::kByte)
        elemTy = mlir::IntegerType::get(8, &context);
      else {
        std::cout << tdn->tensor().dtype() << "\n";
        assert(0 && "bad type");
      }
    } else {
      elemTy = mlir::FloatType::getF32(&context);
    }
    return mlir::RankedTensorType::get(shape, elemTy);
  };

  std::set<ir::Value> visited;
  for (auto &v : vs)
    genParameters(v, visited);

  std::map<ir::Value, ir::Value> parameter_map;
  std::vector<ir::Value> unique_parameters;

  for (const ir::Value &p : parameters) {
    bool found = false;
    for (const ir::Value &q : unique_parameters) {
      if (p.node->op() == ir::OpKind::Get("aten::torch_data") &&
          q.node->op() == ir::OpKind::Get("aten::torch_data")) {
        auto &ptd = *dynamic_cast<ir::TorchDataNode *>(p.node.get());
        auto &qtd = *dynamic_cast<ir::TorchDataNode *>(q.node.get());
        if (ptd.tensor().is_same(qtd.tensor())) {
          found = true;
          parameter_map.insert({p, q});
          break;
        }
      }
    }
    if (!found) {
      unique_parameters.push_back(p);
    }
  }

  // collect the argument types and tensors
  std::vector<mlir::Type> arg_types;
  for (const ir::Value &p : unique_parameters) {
    // tensor type for the function signature
    arg_types.push_back(gen_tensor_ty(p));

    // tensor itself for actually calling the graph
    auto tdn = dynamic_cast<ir::TorchDataNode *>(p.node.get());
    arguments.push_back(tdn->tensor());
  }

  // construct return type
  std::vector<mlir::Type> ret_types;
  for (auto &v : vs)
    ret_types.push_back(gen_tensor_ty(v));

  // create the function type and the function itself
  auto func_type = mlir::FunctionType::get(arg_types, ret_types, &context);
  auto function =
      mlir::FuncOp::create(loc, "graph", func_type, /* attrs = */ {});

  // entry
  auto &entryBlock = *function.addEntryBlock();

  // Declare all the function arguments in the symbol table.
  for (const auto &i :
       llvm::zip(unique_parameters, entryBlock.getArguments())) {
    declareSymbol(std::get<0>(i), std::get<1>(i));
  }
  // Declare all the duplicates from the original
  // parameter list in the symbol table
  for (auto &k_v : parameter_map) {
    assert(symbolTable.count(k_v.second));
    declareSymbol(k_v.first, symbolTable[k_v.second]);
  }

  builder = std::make_unique<mlir::OpBuilder>(function.getBody());

  std::vector<mlir::Value> rets;
  for (auto &v : vs)
    rets.push_back(genValue(v));

  builder->create<mlir::ReturnOp>(loc, rets);
  return function;
}

bool MLIRGen::declareSymbol(const ir::Value &irValue, mlir::Value mlirValue) {
  if (symbolTable.count(irValue)) {
    return false;
  }
  symbolTable.insert({irValue, mlirValue});
  return true;
}

} // namespace torch_mlir
