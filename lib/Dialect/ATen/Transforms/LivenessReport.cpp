//===- LivenessReport.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "npcomp/Dialect/ATen/IR/ATenDialect.h"
#include "npcomp/Dialect/ATen/Transforms/LivenessReport.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"

#include <iostream>
#include <sstream>
#include <vector>

#define DEBUG_TYPE "liveness-report"

using namespace mlir;

namespace mlir {
namespace NPCOMP {
namespace aten {

std::string LivenessReport::generateTextReport() {
  resolveLiveness();
  std::string output;
  for (auto &p : livenessIntervals) {
    Value v = p.first;
    std::vector<Operation *> &oplist = p.second;
    llvm::outs() << "// begin\n";
    v.print(llvm::outs());
    llvm::outs() << "\n";
    for (auto *o : oplist) {
      o->print(llvm::outs());
      llvm::outs() << "\n";
    }
    llvm::outs() << "// end \n";
  }
  return output;
}

std::string LivenessReport::emitJSONReport() {
  resolveLiveness();
  llvm::json::Object top;
  auto graph = module.lookupSymbol<mlir::FuncOp>("graph");

  std::map<Operation *, std::vector<Value>> liveAt;

  graph.walk([&](Operation *op) {
    for (Value result : op->getResults()) {
      for (auto &p : livenessIntervals) {
        Value v = p.first;
        if (v == result) {
          std::vector<Operation *> &oplist = p.second;
          for (auto *o : oplist)
            liveAt[o].push_back(result);
        }
      }
    }
  });

  auto argList = graph.getBody().getBlocks().front().getArguments();
  for (Value &arg : argList) {
    for (auto &p : livenessIntervals) {
      Value v = p.first;
      if (v == arg) {
        std::vector<Operation *> &oplist = p.second;
        for (auto *o : oplist)
          liveAt[o].push_back(arg);
      }
    }
  }

  graph.walk([&](Operation *op) {
    llvm::json::Object layerDetail;
    auto attr = op->getAttrOfType<StringAttr>("layer_name");
    if (!attr)
      return;
    std::vector<Value> &vlist = liveAt[op];
    int64_t parameterVol = 0;
    int64_t returnVol = 0;
    for (auto v : vlist) {
      int64_t vol = getTensorVolume(v.getType());
      if (v.getDefiningOp()) {
        if (auto a =
                v.getDefiningOp()->getAttrOfType<StringAttr>("layer_name")) {
          auto ld = layerDetail.getInteger(a.getValue().str());
          if (ld)
            layerDetail[a.getValue().str()] = *ld + vol;
          else
            layerDetail[a.getValue().str()] = vol;
        } else {
          llvm_unreachable("unknown type");
        }
      } else if (std::find(argList.begin(), argList.end(), v) !=
                 argList.end()) {
        parameterVol += vol;
      } else {
        llvm_unreachable("unknown type");
      }
      auto ret = cast<ReturnOp>(op->getBlock()->getTerminator());
      for (auto oper : ret.getOperands()) {
        if (oper == v) {
          returnVol += vol;
          break;
        }
      }
    }
    if (parameterVol) {
      layerDetail["parameters"] = parameterVol;
    }
    if (returnVol) {
      layerDetail["returns"] = returnVol;
    }
    top[attr.getValue().str()] = llvm::json::Value(std::move(layerDetail));
  });

  llvm::json::Value topv(std::move(top));
  std::string ret;
  llvm::raw_string_ostream ss(ret);
  ss << llvm::formatv("{0:2}", topv) << "\n";
  return ss.str();
}

void LivenessReport::resolveLiveness() {

  auto context = module.getContext();
  auto loc = mlir::UnknownLoc::get(context);

  // check that a function called "graph" exists
  auto graph = module.lookupSymbol<mlir::FuncOp>("graph");
  if (!graph) {
    emitError(mlir::UnknownLoc::get(module.getContext()),
              "LivenessReport failed: can't find a graph function\n");
    return;
  }

  // put each aten operation into its own basic block,
  // so that we can use standard liveness
  Region &bodyRegion = graph.getBody();
  Block *entryBB = &bodyRegion.getBlocks().front();
  Block *BB = entryBB;
  std::vector<Block *> new_blocks;
  while (true) {
    std::vector<Operation *> ops;
    for (Operation &op : BB->getOperations())
      ops.push_back(&op);

    // skip over constant ops
    int idx = 0;
    while (dyn_cast<mlir::NPCOMP::aten::ConstantOp>(ops[idx]))
      idx++;

    if (dyn_cast<ReturnOp>(ops[idx]))
      break;

    Block *newBB = BB->splitBlock(ops[idx + 1]);
    new_blocks.push_back(newBB);

    mlir::OpBuilder builder = mlir::OpBuilder::atBlockBegin(BB);
    builder.create<BranchOp>(loc, newBB);
    BB = newBB;
  }

  // dump transformed function
  // graph.dump();

  // run MLIR Liveness analysis
  auto liveness = Liveness(graph);

  for (BlockArgument &arg :
       graph.getBody().getBlocks().front().getArguments()) {
    auto liveOps = liveness.resolveLiveness(arg);
    for (Operation *o : liveOps) {
      livenessIntervals[arg].push_back(o);
    }
  }

  graph.walk([&](Operation *op) {
    auto attr = op->getAttrOfType<StringAttr>("layer_name");
    if (!attr)
      return;

    for (Value v : op->getResults()) {
      auto liveOps = liveness.resolveLiveness(v);
      for (Operation *o : liveOps)
        if (auto a = o->getAttrOfType<StringAttr>("layer_name"))
          livenessIntervals[v].push_back(o);
    }
  });

  // undo the BB insert
  auto *deadBr = bodyRegion.getBlocks().front().getTerminator();
  for (Block *b : new_blocks) {

    auto *br = b->getTerminator();
    std::vector<Operation *> ops;
    for (auto &op : b->getOperations())
      ops.push_back(&op);

    for (auto *op : ops) {
      if (op == br && !dyn_cast<ReturnOp>(op)) {
        op->erase();
        continue;
      }
      op->moveBefore(deadBr);
    }
  }
  deadBr->erase();

  for (Block *b : new_blocks)
    b->erase();

  // graph.dump();
}

} // namespace aten
} // namespace NPCOMP
} // namespace mlir
