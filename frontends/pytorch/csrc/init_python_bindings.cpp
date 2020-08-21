//===- init_python_bindings.cpp ---------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

// This file implements Python bindings to the MLIR/NPCOMP ATen dialect.
// Roughly speaking, it enables something like this:
//
//  dev = torch_mlir.mlir_device()
//  t0 = torch.randn((4,4), device=dev)
//  t1 = torch.randn((4,4), device=dev)
//  t2 = t0 + t1
//  t2_mlir = torch_mlir.get_mlir( t2 )
//  t2_cpu = t2.to('cpu')
//
// In this case t2_cpu contains the result of the computation, and t2_mlir
// contains the mlir description of the computation.

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

// Then ATen headers with workarounds
#include "ATen/ArrayRef.h"
namespace at {
template <typename T> using ArrayRef = c10::ArrayRef<T>;
}
#include "ATen/SmallVector.h"
namespace at {
template <typename T, int S> using SmallVector = c10::SmallVector<T, S>;
}
#include <ATen/Tensor.h>

// other headers

#include "aten_mlir_bridge.h"
#include "aten_mlir_type.h"
#include "init_python_bindings.h"
#include "mlir_gen.h"

#include <string>

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

void InitModuleBindings(py::module m) {
  m.def("_initialize_aten_bindings",
        []() { ATenMLIRType::InitializeAtenBindings(); });
  m.def("_set_default_device", []() {});

  m.def("_get_mlir", [](std::vector<at::Tensor> &ts) -> std::string {
    if (ts.size() == 0)
      return std::string();

    mlir::MLIRContext context;

    // gather IR for all the tensors
    std::vector<ir::Value> recorded_ir;
    for (auto &t : ts)
      if (c10::optional<MLIRTensor> at = bridge::TryGetMLIRTensor(t))
        recorded_ir.push_back(at->GetIrValue());

    // generate MLIR from IR
    auto mlir_gen = MLIRGen(context).genModule(recorded_ir);
    mlir::OwningModuleRef module = std::move(std::get<0>(mlir_gen));

    mlir::PassManager pm(module->getContext());

    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::NPCOMP::aten::createATenLayerNamePass());
    if (failed(pm.run(*module))) {
      llvm::errs() << "ATenLayerNamePass failed";
      return "<error>";
    }

    // dump MLIR to string and return
    std::string s;
    llvm::raw_string_ostream ss(s);
    module->print(ss);
    return ss.str();
  });

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

void InitBindings(py::module m) { InitModuleBindings(m); }

} // namespace torch_mlir

PYBIND11_MODULE(_torch_mlir, m) { torch_mlir::InitBindings(m); }
