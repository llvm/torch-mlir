//===- PythonModule.cpp - RefJIT python bindings --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "./NpcompModule.h"

#include <cstdlib>

#include "pybind11/numpy.h"

#include "npcomp-c/RefJITBackend.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

static NpcompRefJitElementType
mapBufferFormatToElementType(const std::string &format, py::ssize_t itemSize) {
  if (format == "f")
    return NPCOMP_REFJIT_F32;

  std::string message("unsupported buffer format: ");
  message.append(format);
  throw py::raiseValueError(message);
}

namespace {

struct PyRefJitModule {
  PyRefJitModule(NpcompRefJitModule instance) : instance(instance) {}
  ~PyRefJitModule() {
    if (instance.ptr)
      npcompRefJitModuleDestroy(instance);
  }
  PyRefJitModule(const PyRefJitModule &) = delete;
  void operator=(const PyRefJitModule &) = delete;
  PyRefJitModule(PyRefJitModule &&other) : instance(other.instance) {
    other.instance.ptr = nullptr;
  }

  NpcompRefJitModule instance = {nullptr};
};

struct PyRefJitValueList {
  PyRefJitValueList(NpcompRefJitValueList instance) : instance(instance) {}
  ~PyRefJitValueList() {
    if (instance.ptr)
      npcompRefJitValueListDestroy(instance);
  }
  PyRefJitValueList(const PyRefJitValueList &) = delete;
  void operator=(const PyRefJitValueList &) = delete;
  PyRefJitValueList(PyRefJitValueList &&other) : instance(other.instance) {
    other.instance.ptr = nullptr;
  }

  NpcompRefJitValueList instance = {nullptr};
};

} // namespace

void npcomp::python::defineBackendRefJitModule(py::module &m) {
  m.def("build_backend_compilation_pipeline", [](MlirPassManager capiPm) {
    npcompRefJitBuildBackendCompilationPipeline(capiPm, /*optimize=*/true);
  });
  py::class_<PyRefJitValueList>(m, "ValueList");
  py::class_<PyRefJitModule>(m, "JITModule")
      .def_static(
          "from_compiled_module",
          [](MlirModule capiModule,
             std::vector<std::string> pySharedLibs) -> PyRefJitModule {
            SmallVector<MlirStringRef, 4> sharedLibs;
            for (auto &s : pySharedLibs)
              sharedLibs.push_back(MlirStringRef{s.data(), s.size()});
            char *errorMessageCstr;
            NpcompRefJitModule m =
                npcompRefJitModuleCreate(capiModule, &sharedLibs[0],
                                         sharedLibs.size(), &errorMessageCstr);
            if (npcompRefJitModuleIsNull(m)) {
              std::string errorMessage(errorMessageCstr);
              std::free(errorMessageCstr);
              throw py::raisePyError(PyExc_RuntimeError, errorMessage.c_str());
            }
            return PyRefJitModule(m);
          },
          py::arg("module"), py::arg("shared_libs"))
      .def(
          "invoke",
          [](PyRefJitModule &self, std::string functionName,
             std::vector<py::buffer> inputs) {
            py::object ioListObject =
                py::cast(PyRefJitValueList(npcompRefJitValueListCreate()));
            PyRefJitValueList &ioList =
                py::cast<PyRefJitValueList &>(ioListObject);

            // Prepare inputs.
            for (auto &buffer : inputs) {
              // Request a C contiguous view as that is what Tensor accepts now
              // (no strides or non row-major layout).
              int flags = PyBUF_C_CONTIGUOUS | PyBUF_FORMAT;
              std::unique_ptr<Py_buffer> view(new Py_buffer());
              if (PyObject_GetBuffer(buffer.ptr(), view.get(), flags) != 0) {
                throw py::error_already_set();
              }
              py::buffer_info info(view.release());
              auto elementType =
                  mapBufferFormatToElementType(info.format, info.itemsize);
              SmallVector<int32_t, 4> extents(info.shape.begin(),
                                              info.shape.end());

              npcompRefJitValueAddTensorCopy(ioList.instance, elementType,
                                             extents.data(), extents.size(),
                                             info.ptr);
            }

            // Invoke.
            char *errorMessageCstr;
            if (!npcompRefJitModuleInvoke(
                    self.instance,
                    MlirStringRef{functionName.data(), functionName.size()},
                    ioList.instance, &errorMessageCstr)) {
              std::string errorMessage(errorMessageCstr);
              std::free(errorMessageCstr);
              throw py::raisePyError(PyExc_RuntimeError, errorMessage.c_str());
            }

            // Prepare outputs.
            std::vector<py::object> outputs;
            for (intptr_t i = 0; i < npcompRefJitValueListSize(ioList.instance);
                 ++i) {
              if (npcompRefJitValueIsaTensor(ioList.instance, i)) {
                NpcompRefJitElementType elementType;
                intptr_t rank;
                const int32_t *extents;
                void *data = npcompRefJitValueGetTensor(
                    ioList.instance, i, &elementType, &rank, &extents);

                const char *format;
                switch (elementType) {
                case NPCOMP_REFJIT_F32:
                  format = "f";
                  break;
                default:
                  throw py::raiseValueError("unsupported tensor element type");
                }

                outputs.push_back(
                    py::array(py::dtype(format),
                              llvm::ArrayRef<std::int32_t>(extents, rank), data,
                              /*base=*/ioListObject));
              } else {
                throw py::raisePyError(PyExc_ValueError,
                                       "unsupported npcomp refjit return type");
              }
            }
            return outputs;
          },
          py::arg("function_name"), py::arg("inputs"));
}
