//===- native.cpp - MLIR Python bindings ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <unordered_map>

#include "native.h"
#include "pybind_utils.h"

#include "llvm/Support/CommandLine.h"

namespace mlir {
namespace npcomp {
namespace python {

void defineLLVMModule(pybind11::module m) {
  m.def("print_help_message", []() { llvm::cl::PrintHelpMessage(); });
  m.def("add_option",
        [](std::string name, llvm::Optional<std::string> value) {
          auto options_map = llvm::cl::getRegisteredOptions();
          auto found_it = options_map.find(name);
          if (found_it == options_map.end()) {
            std::string message = "Unknown LLVM option: ";
            message.append(name);
            throw py::raiseValueError(message.c_str());
          }

          std::string value_sr = value ? *value : "";
          found_it->getValue()->addOccurrence(1, name, value_sr);
        },
        py::arg("name"), py::arg("value") = llvm::Optional<std::string>());
  m.def("reset_option",
        [](std::string name) {
          auto options_map = llvm::cl::getRegisteredOptions();
          auto found_it = options_map.find(name);
          if (found_it == options_map.end()) {
            std::string message = "Unknown LLVM option: ";
            message.append(name);
            throw py::raiseValueError(message.c_str());
          }
          found_it->getValue()->setDefault();
        },
        py::arg("name"));
}

PYBIND11_MODULE(native, m) {
  // Guard the once init to happen once per process (vs module, which in
  // mondo builds can happen multiple times).
  static bool llvm_init_baton = ([]() { return npcompMlirInitialize(); })();
  (void)(llvm_init_baton);

  m.doc() = "Npcomp native python bindings";

  auto llvm_m = m.def_submodule("llvm", "LLVM interop");
  defineLLVMModule(llvm_m);

  auto mlir_m = m.def_submodule("mlir", "MLIR interop");
  auto mlir_ir_m = mlir_m.def_submodule("ir");
  defineMlirIrModule(mlir_ir_m);

  auto npcomp_dialect = m.def_submodule("dialect", "NPComp custom dialects");
  defineNpcompDialect(npcomp_dialect);
}

} // namespace python
} // namespace npcomp
} // namespace mlir
