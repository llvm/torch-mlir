//===- PythonModule.cpp - IREE python bindings ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Backend/IREE/PythonModule.h"

#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/VM/Target/Bytecode/BytecodeModuleTarget.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "npcomp/Python/MlirIr.h"
#include "npcomp/Python/MlirPass.h"

using namespace mlir;

namespace {

class Blob {
public:
  Blob(std::string contents) : contents(contents) {}
  std::string contents;
};

} // namespace

/// Defines an "iree" module with backend support definitions.
void mlir::npcomp::python::defineBackendIREEModule(py::module m) {
  py::class_<Blob>(m, "Blob", py::buffer_protocol())
      .def_buffer([](Blob &self) -> py::buffer_info {
        return py::buffer_info(
            static_cast<void *>(&self.contents.front()), // Pointer to buffer
            sizeof(uint8_t),                             // Size of one scalar
            py::format_descriptor<uint8_t>::value,       // Python struct-style
                                                         // format
            1,                                           // Number of dimensions
            {self.contents.size()},                      // Buffer dimensions
            {self.contents.size()}                       // Strides
        );
      });

  m.def("build_flow_transform_pass_pipeline",
        [](PyPassManager &pm) {
          mlir::iree_compiler::IREE::Flow::buildFlowTransformPassPipeline(
              pm.passManager);
        },
        py::arg("pm"),
        py::doc("Builds a pass pipeline for top-level Flow import"));
  m.def("build_hal_transform_pass_pipeline",
        [](PyPassManager &pm, std::vector<std::string> targetBackends) {
          mlir::iree_compiler::IREE::HAL::TargetOptions options;
          if (targetBackends.empty()) {
            options.targets =
                mlir::iree_compiler::IREE::HAL::getRegisteredTargetBackends();
          } else {
            options.targets = std::move(targetBackends);
          }
          iree_compiler::IREE::HAL::buildHALTransformPassPipeline(
              pm.passManager, options);
        },
        py::arg("pm"), py::arg("target_backends") = std::vector<std::string>(),
        py::doc("Builds a pass pipeline for top-level Flow import"));
  m.def("build_vm_transform_pass_pipeline",
        [](PyPassManager &pm) {
          mlir::iree_compiler::IREE::VM::buildVMTransformPassPipeline(
              pm.passManager);
        },
        py::arg("pm"), py::doc("Builds the VM transformation pipeline"));
  m.def("translate_to_vm_bytecode", [](PyModuleOp &module) {
    // TODO: Make the options parameterizable.
    mlir::iree_compiler::IREE::VM::BytecodeTargetOptions options;
    std::string contents;
    llvm::raw_string_ostream out(contents);
    if (failed(mlir::iree_compiler::IREE::VM::translateModuleToBytecode(
            module.moduleOp, options, out))) {
      // TODO: Merge diagnostic captures in.
      throw py::raisePyError(PyExc_RuntimeError,
                             "Error translating module (see stderr)");
    }

    out.flush();
    return Blob(std::move(out.str()));
  });
}
