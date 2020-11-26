//===- npcomp_py_interop.h --------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//
// Helper methods for interfacing-by-string with npcomp and MLIR Python APIs.
// Keeping these roughly in one place facilitates top-level sanity checks
// at load time and eases string find-replace maintenance.
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_FRONTENDS_PYTORCH_CSRC_NPCOMP_PY_INTEROP_H
#define NPCOMP_FRONTENDS_PYTORCH_CSRC_NPCOMP_PY_INTEROP_H

#include "pybind.h"

#include "mlir-c/Bindings/Python/Interop.h"

namespace torch_mlir {

//------------------------------------------------------------------------------
// MLIR Python accessors.
//------------------------------------------------------------------------------

inline py::object getMlirIrClass(const char *className) {
  return py::module::import("mlir.ir").attr(className);
}

inline py::object getMlirContextClass() { return getMlirIrClass("Context"); }

inline py::object getMlirModuleClass() { return getMlirIrClass("Module"); }

//------------------------------------------------------------------------------
// NPCOMP Python accessors.
//------------------------------------------------------------------------------

inline py::object createNpcompMetaModuleClass(py::object mlirModule) {
  return py::module::import("npcomp.meta.meta_module")
      .attr("MetaModule")(std::move(mlirModule));
}

inline py::object createNpcompMetaSignatureClass(int arity) {
  return py::module::import("npcomp.meta.types").attr("Signature")(arity);
}

inline py::object
createNpcompMetaExportGenericFunction(py::str irSymbolName,
                                      py::object signatureObj) {
  return py::module::import("npcomp.meta.meta_module")
      .attr("ExportGenericFunction")(std::move(irSymbolName),
                                     std::move(signatureObj));
}

inline py::object
createNpcompMetaExportSpecializedFunction(py::str irSymbolName,
                                          py::object signatureObj) {
  return py::module::import("npcomp.meta.meta_module")
      .attr("ExportSpecializedFunction")(std::move(irSymbolName),
                                         std::move(signatureObj));
}

inline py::object mapMlirTypeToMetaType(MlirType t) {
  PyObject *capsule = mlirPythonTypeToCapsule(t);
  return py::module::import("_npcomp.types")
      .attr("map_mlir_type_to_meta_type")(
          py::reinterpret_steal<py::object>(capsule));
}

} // namespace torch_mlir

#endif // NPCOMP_FRONTENDS_PYTORCH_CSRC_NPCOMP_PY_INTEROP_H
