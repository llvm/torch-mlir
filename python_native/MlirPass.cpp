//===- MlirIr.cpp - MLIR IR Bindings --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MlirPass.h"
#include "MlirInit.h"
#include "NpcompModule.h"

#include "mlir/Pass/PassRegistry.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// Module initialization
//===----------------------------------------------------------------------===//

void defineMlirPassModule(py::module m) {
  m.doc() = "Python bindings for mlir pass infra";

  PyPassManager::bind(m);
}

//===----------------------------------------------------------------------===//
// PassManager
//===----------------------------------------------------------------------===//

void PyPassManager::bind(py::module m) {
  py::class_<PyPassManager>(m, "PassManager")
      .def(py::init<std::shared_ptr<PyContext>, bool>(), py::arg("context"),
           py::arg("verifyModules") = true)
      .def("enableCrashReproducerGeneration",
           [](PyPassManager &self, std::string outputFile,
              bool genLocalReproducer) {
             self.passManager.enableCrashReproducerGeneration(
                 outputFile, genLocalReproducer);
           },
           py::arg("outputFile"), py::arg("genLocalReproducer") = false)
      .def("__len__",
           [](PyPassManager &self) { return self.passManager.size(); })
      .def("__str__",
           [](PyPassManager &self) {
             std::string spec;
             llvm::raw_string_ostream stream(spec);
             self.passManager.printAsTextualPipeline(stream);
             return spec;
           })
      .def("run",
           [](PyPassManager &self, PyModuleOp &module) {
             if (module.context.get() != self.context.get()) {
               throw py::raiseValueError(
                   "Expected a module with the same context "
                   "as the PassManager");
             }
             if (failed(self.passManager.run(module.moduleOp))) {
               // TODO: Wrap propagate context diagnostics
               throw py::raisePyError(PyExc_RuntimeError,
                                      "Could not run passes");
             }
           })
      .def("addPassPipelines", [](PyPassManager &self, py::args passPipelines) {
        std::string error;
        llvm::raw_string_ostream error_stream(error);
        for (auto pyPassPipeline : passPipelines) {
          auto passPipeline = pyPassPipeline.cast<std::string>();
          if (failed(mlir::parsePassPipeline(passPipeline, self.passManager,
                                             error_stream))) {
            std::string message = "failed to parse pass pipeline '";
            message.append(passPipeline);
            message.append("': ");
            message.append(error);
            throw py::raiseValueError(message);
          }
        }
      });
}

} // namespace mlir
