//===- ATenOpReport.cpp -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/ATen/ATenOpReport.h"
#include "npcomp/Dialect/ATen/ATenDialect.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Pass/Pass.h"

#include <iostream>
#include <vector>

#define DEBUG_TYPE "aten-op-stats"

using namespace mlir;

namespace {

std::string getAsString(std::map<std::string, uint64_t> &m, std::string &e) {
  return m.count(e) ? std::to_string(m[e]) : " ";
}

/// Query operations through the StatisticsOpInterface and print the result
/// in a human-readable way.  This replicates the functionality in various
/// network analysis tools and is a stepping stone toward using the information
/// as an analysis to drive optimization.
struct ATenOpReportPass
    : public PassWrapper<ATenOpReportPass, OperationPass<ModuleOp>> {

private:
  std::string *output;
  std::vector<std::string> tableFields;
  std::map<Operation *, std::string> opToName;

public:
  ATenOpReportPass()
      : output(nullptr),
        tableFields({"reads", "writes", "activation_in", "activation_out",
                     "parameters_in", "ops:MAC", "ops:==", "ops:>", "ops:*",
                     "ops:+", "ops:/", "ops:sqrt", "ops:-", "grad"}) {}

  ATenOpReportPass(std::string *output)
      : output(output),
        tableFields({"reads", "writes", "activation_in", "activation_out",
                     "parameters_in", "ops:MAC", "ops:==", "ops:>", "ops:*",
                     "ops:+", "ops:/", "ops:sqrt", "ops:-", "grad"}) {}

  std::string emitJSONReport() {

    llvm::json::Object top;

    auto graph = getOperation().lookupSymbol<mlir::FuncOp>("graph");
    graph.walk([&](Operation *op) {
      if (auto stats =
              mlir::dyn_cast<mlir::NPCOMP::StatisticsOpInterface>(op)) {

        // name for this layer
        std::string layerName = opToName[op];

        // raw stats for this layer
        std::map<std::string, uint64_t> layerStatsMap = stats.getStatistics();

        // JSON version of the stats we are building
        llvm::json::Object layerStatsJSON;

        // foreach string f in tableField,
        // get the sum of all entries in layerStatsMap containing f
        for (auto &f : tableFields) {
          for (auto &p : layerStatsMap) {
            if (p.first.find(f) != std::string::npos) {
              if (auto count = layerStatsJSON[f].getAsInteger())
                layerStatsJSON[f] = (int64_t)p.second + *count;
              else
                layerStatsJSON[f] = (int64_t)p.second;
            }
          }
        }
        top[layerName] = llvm::json::Value(std::move(layerStatsJSON));
      }
    });

    llvm::json::Value topv(std::move(top));
    std::string ret;
    llvm::raw_string_ostream ss(ret);
    ss << llvm::formatv("{0:2}", topv) << "\n";
    return ss.str();
  }

  void runOnOperation() override {

    // I don't change anything
    markAllAnalysesPreserved();

    auto module = getOperation();

    // check that a function called "graph" exists
    auto graph = module.lookupSymbol<mlir::FuncOp>("graph");
    if (!graph) {
      emitError(mlir::UnknownLoc::get(module.getContext()),
                "OpReportPass failed: can't find a graph function\n");
      signalPassFailure();
      return;
    }

    unsigned currentLayer = 0;
    opToName.clear();
    graph.walk([&](Operation *op) {
      auto attr = op->getAttrOfType<StringAttr>("layer_name");
      if (attr)
        opToName[op] = attr.getValue().str();
      else
        opToName[op] = "unknown-layer-" + std::to_string(currentLayer);
      currentLayer++;
    });

    std::string report = emitJSONReport();
    if (output) {
      *output = report;
    } else {
      graph.emitWarning(report);
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::NPCOMP::aten::createATenOpReportPass() {
  return std::make_unique<ATenOpReportPass>();
}

std::unique_ptr<mlir::Pass>
mlir::NPCOMP::aten::createATenOpReportPass(std::string &report) {
  return std::make_unique<ATenOpReportPass>(&report);
}

void mlir::NPCOMP::aten::registerATenOpReportPass() {
  PassRegistration<ATenOpReportPass>("aten-op-report",
                                     "Generate ATen operation report");
}
