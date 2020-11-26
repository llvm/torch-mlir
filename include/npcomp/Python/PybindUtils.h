//===- PybindUtils.h - Utilities for interop with python ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_PYTHON_PYBIND_UTILS_H
#define NPCOMP_PYTHON_PYBIND_UTILS_H

#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "llvm/ADT/Optional.h"

namespace py = pybind11;

namespace pybind11 {
namespace detail {

template <typename T>
struct type_caster<llvm::Optional<T>> : optional_caster<llvm::Optional<T>> {};

/// Helper to convert a presumed MLIR API object to a capsule, accepting either
/// an explicit Capsule (which can happen when two C APIs are communicating
/// directly via Python) or indirectly by querying the MLIR_PYTHON_CAPI_PTR_ATTR
/// attribute (through which supported MLIR Python API objects export their
/// contained API pointer as a capsule). This is intended to be used from
/// type casters, which are invoked with a raw handle (unowned). The returned
/// object's lifetime may not extend beyond the apiObject handle without
/// explicitly having its refcount increased (i.e. on return).
static py::object mlirApiObjectToCapsule(py::handle apiObject) {
  if (PyCapsule_CheckExact(apiObject.ptr()))
    return py::reinterpret_borrow<py::object>(apiObject);
  return apiObject.attr(MLIR_PYTHON_CAPI_PTR_ATTR);
}

/// Casts object -> MlirAttribute.
template <> struct type_caster<MlirAttribute> {
  PYBIND11_TYPE_CASTER(MlirAttribute, _("MlirAttribute"));
  bool load(handle src, bool) {
    auto capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToAttribute(capsule.ptr());
    if (mlirAttributeIsNull(value)) {
      return false;
    }
    return true;
  }
};

/// Casts object -> MlirContext.
template <> struct type_caster<MlirContext> {
  PYBIND11_TYPE_CASTER(MlirContext, _("MlirContext"));
  bool load(handle src, bool) {
    auto capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToContext(capsule.ptr());
    if (mlirContextIsNull(value)) {
      return false;
    }
    return true;
  }
};

/// Casts object -> MlirModule.
template <> struct type_caster<MlirModule> {
  PYBIND11_TYPE_CASTER(MlirModule, _("MlirModule"));
  bool load(handle src, bool) {
    auto capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToModule(capsule.ptr());
    if (mlirModuleIsNull(value)) {
      return false;
    }
    return true;
  }
};

/// Casts object -> MlirOperation.
template <> struct type_caster<MlirOperation> {
  PYBIND11_TYPE_CASTER(MlirOperation, _("MlirOperation"));
  bool load(handle src, bool) {
    auto capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToOperation(capsule.ptr());
    if (mlirOperationIsNull(value)) {
      return false;
    }
    return true;
  }
};

/// Casts object -> MlirPassManager.
template <> struct type_caster<MlirPassManager> {
  PYBIND11_TYPE_CASTER(MlirPassManager, _("MlirPassManager"));
  bool load(handle src, bool) {
    auto capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToPassManager(capsule.ptr());
    if (mlirPassManagerIsNull(value)) {
      return false;
    }
    return true;
  }
};

/// Casts object -> MlirType.
template <> struct type_caster<MlirType> {
  PYBIND11_TYPE_CASTER(MlirType, _("MlirType"));
  bool load(handle src, bool) {
    auto capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToType(capsule.ptr());
    if (mlirTypeIsNull(value)) {
      return false;
    }
    return true;
  }
};

} // namespace detail
} // namespace pybind11

namespace pybind11 {

/// Raises a python exception with the given message.
/// Correct usage:
//   throw RaiseValueError(PyExc_ValueError, "Foobar'd");
inline pybind11::error_already_set raisePyError(PyObject *exc_class,
                                                const char *message) {
  PyErr_SetString(exc_class, message);
  return pybind11::error_already_set();
}

/// Raises a value error with the given message.
/// Correct usage:
///   throw RaiseValueError("Foobar'd");
inline pybind11::error_already_set raiseValueError(const char *message) {
  return raisePyError(PyExc_ValueError, message);
}

/// Raises a value error with the given message.
/// Correct usage:
///   throw RaiseValueError(message);
inline pybind11::error_already_set raiseValueError(const std::string &message) {
  return raisePyError(PyExc_ValueError, message.c_str());
}

} // namespace pybind11

#endif // NPCOMP_PYTHON_PYBIND_UTILS_H
