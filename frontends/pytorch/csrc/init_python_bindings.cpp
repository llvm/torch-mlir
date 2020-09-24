//===- init_python_bindings.cpp ---------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

// This is the top-level entry point for the MLIR/NPCOMP <-> PyTorch bridge.
// It provides several mechanisms for extracting programs from PyTorch via:
//   a) A pseudo-device which captures the operations to an MLIR module
//      (implemented via the legacy type_dispatch mechanism for PyTorch 1.3).
//   b) Direct IR translation from PyTorch Graphs (not implemented).
//   c) Using the PyTorch JIT facility (not implemented).

#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "npcomp/Dialect/ATen/ATenDialect.h"
#include "npcomp/Dialect/ATen/ATenOpReport.h"
#include "npcomp/Dialect/ATen/ATenPasses.h"
#include "npcomp/Dialect/ATen/LivenessReport.h"

#include "init_python_bindings.h"

#include <string>

namespace py = pybind11;
using namespace mlir;

namespace llvm {
extern bool DebugFlag;
}

namespace torch_mlir {
namespace {

mlir::OwningModuleRef LoadModule(mlir::MLIRContext &context, std::string mlir) {

  mlir::OwningModuleRef module;

  std::unique_ptr<llvm::MemoryBuffer> membuf =
      llvm::MemoryBuffer::getMemBuffer(mlir);

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(membuf), llvm::SMLoc());
  module = mlir::parseSourceFile(sourceMgr, &context);

  if (!module) {
    llvm::errs() << "Error can't parse mlir module\n";
    return nullptr;
  }
  if (failed(mlir::verify(*module))) {
    llvm::errs() << "Error verifying MLIR module\n";
    return nullptr;
  }
  if (!module)
    return nullptr;
  return module;
}

void InitModuleBindings(py::module &m) {
  m.def(
      "_op_report",
      [](std::string mlir) -> std::string {
        mlir::MLIRContext context;
        auto module = LoadModule(context, mlir);
        mlir::PassManager pm(module->getContext());

        // our pass
        std::string report;
        pm.addPass(mlir::NPCOMP::aten::createATenLayerNamePass());
        pm.addPass(mlir::NPCOMP::aten::createATenOpReportPass(report));

        if (failed(pm.run(*module))) {
          llvm::errs() << "ATenOpReportPass failed";
          return "<error>";
        }
        return report;
      },
      "run ATenOpReportPass");

  m.def(
      "_liveness_report",
      [](std::string mlir) -> std::string {
        mlir::MLIRContext context;
        auto module = LoadModule(context, mlir);

        mlir::PassManager pm(module->getContext());

        pm.addPass(mlir::NPCOMP::aten::createATenLayerNamePass());
        if (failed(pm.run(*module))) {
          llvm::errs() << "ATen generate liveness report failed";
          return "<error>";
        }

        auto mOp = module.get();
        auto liveness = mlir::NPCOMP::aten::LivenessReport(mOp);
        std::string report = liveness.emitJSONReport();
        return report;
      },
      "generate liveness report");

  // TODO: Could this be implemented with MLIR python bindings?
  m.def(
      "lower_to_std",
      [](std::string mlir) -> std::string {
        mlir::MLIRContext context;
        auto module = LoadModule(context, mlir);

        PassManager pm0(module->getContext());
        pm0.addPass(mlir::NPCOMP::aten::createATenLoweringPass());
        pm0.addPass(mlir::NPCOMP::aten::createReturnEliminationPass());
        pm0.addPass(mlir::createCSEPass());

        if (failed(pm0.run(*module))) {
          llvm::errs() << "aten to loops conversion failed ";
          return "";
        }

        // dump MLIR to string and return
        std::string s;
        llvm::raw_string_ostream ss(s);
        ss << "# Lowered to Std\n";
        module->print(ss);
        return ss.str();
      },
      "lower aten to std dialect");

  m.def(
      "set_debug",
      [](bool b, std::string type) -> void {
        llvm::setCurrentDebugType(type.c_str());
        llvm::DebugFlag = b;
      },
      "enable/disable debug messages");
}

} // namespace

void InitBindings(py::module &m) {
  InitModuleBindings(m);

#if defined(NPCOMP_ENABLE_TORCH_TYPE_DISPATCH)
  InitTypeDispatchBindings(m);
#else
  auto c10_m = m.def_submodule(
      "c10", "Experimental support for c10 dispatch integration");
  InitC10DispatchBindings(c10_m);
#endif
}

} // namespace torch_mlir

PYBIND11_MODULE(_torch_mlir, m) { torch_mlir::InitBindings(m); }
