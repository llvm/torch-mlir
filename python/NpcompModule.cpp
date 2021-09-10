//===- NpcompModule.cpp - MLIR Python bindings ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <unordered_map>

#include "./NpcompModule.h"
#include "./NpcompPybindUtils.h"

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"
#include "npcomp-c/InitLLVM.h"
#include "npcomp-c/Registration.h"

namespace {

// TODO: Move this upstream.
void emitError(MlirLocation loc, std::string message) {
  ::mlirEmitError(loc, message.c_str());
}

} // namespace

PYBIND11_MODULE(_npcomp, m) {
  m.doc() = "Npcomp native python bindings";
  ::npcompRegisterAllPasses();
  ::npcompInitializeLLVMCodegen();

  m.def("register_all_dialects", ::npcompRegisterAllDialects);
  m.def("emit_error", emitError);

  // Optional backend modules.
  auto backend_m = m.def_submodule("backend", "Backend support");
  (void)backend_m;

#ifdef NPCOMP_ENABLE_REFJIT
  auto refjit_m =
      backend_m.def_submodule("refjit", "Reference CPU Jit Backend");
  ::npcomp::python::defineBackendRefJitModule(refjit_m);
#endif
}
