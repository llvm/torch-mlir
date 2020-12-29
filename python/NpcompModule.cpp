//===- NpcompModule.cpp - MLIR Python bindings ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <unordered_map>

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"
#include "npcomp-c/InitLLVM.h"
#include "npcomp-c/Registration.h"
#include "npcomp-c/Types.h"
#include "npcomp/Python/PybindUtils.h"

#ifdef NPCOMP_ENABLE_REFJIT
#include "npcomp/Backend/RefJIT/PythonModule.h"
#endif

namespace {

MlirType shapedToNdArrayArrayType(MlirType shaped_type) {
  if (!mlirTypeIsAShaped(shaped_type)) {
    throw py::raiseValueError("type is not a shaped type");
  }
  return npcompNdArrayTypeGetFromShaped(shaped_type);
}

MlirType ndarrayToTensorType(MlirType ndarray_type) {
  if (!npcompTypeIsANdArray(ndarray_type)) {
    throw py::raiseValueError("type is not an ndarray type");
  }
  return npcompNdArrayTypeToTensor(ndarray_type);
}

MlirType slotObjectType(MlirContext context, const std::string &className,
                        const std::vector<MlirType> &slotTypes) {
  MlirStringRef classNameSr{className.data(), className.size()};
  return ::npcompSlotObjectTypeGet(context, classNameSr, slotTypes.size(),
                                   slotTypes.data());
}

// TODO: Move this upstream.
void emitError(MlirLocation loc, std::string message) {
  ::mlirEmitError(loc, message.c_str());
}

} // namespace

PYBIND11_MODULE(_npcomp, m) {
  m.doc() = "Npcomp native python bindings";

  m.def("register_all_dialects", ::npcompRegisterAllDialects);
  m.def("_register_all_passes", ::npcompRegisterAllPasses);
  m.def("_initialize_llvm_codegen", ::npcompInitializeLLVMCodegen);
  m.def("shaped_to_ndarray_type", shapedToNdArrayArrayType);
  m.def("ndarray_to_tensor_type", ndarrayToTensorType);
  m.def("slot_object_type", slotObjectType);
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
