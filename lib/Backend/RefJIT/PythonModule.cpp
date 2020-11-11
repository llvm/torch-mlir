//===- PythonModule.cpp - RefJIT python bindings --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Backend/RefJIT/PythonModule.h"

#include "pybind11/numpy.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Pass.h"
#include "npcomp/Python/MlirIr.h"
#include "npcomp/Python/MlirPass.h"
#include "npcomp/RefBackend/JITHelpers/JITModule.h"

using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

// Make namespaces consistent.
using mlir::PyModuleOp;
using mlir::PyPassManager;
using refback::JITModule;
using refbackrt::Ref;
using refbackrt::Tensor;

template <typename T>
static T checkError(llvm::Expected<T> &&expected, Twine banner = {}) {
  if (LLVM_LIKELY(expected))
    return std::move(*expected);

  std::string errorMessage;
  llvm::raw_string_ostream os(errorMessage);
  llvm::logAllUnhandledErrors(expected.takeError(), os, banner);
  os.flush();
  throw py::raisePyError(PyExc_RuntimeError, errorMessage.c_str());
}

static refbackrt::ElementType
mapBufferFormatToElementType(const std::string &format, py::ssize_t itemSize) {
  if (format == "f")
    return refbackrt::ElementType::F32;

  std::string message("unsupported buffer format: ");
  message.append(format);
  throw py::raiseValueError(message);
}

static Ref<Tensor> copyBufferToTensor(py::buffer buffer) {
  // Request a C contiguous view as that is what Tensor accepts now (no strides
  // or non row-major layout).
  int flags = PyBUF_C_CONTIGUOUS | PyBUF_FORMAT;
  std::unique_ptr<Py_buffer> view(new Py_buffer());
  if (PyObject_GetBuffer(buffer.ptr(), view.get(), flags) != 0) {
    throw py::error_already_set();
  }
  py::buffer_info info(view.release());
  auto elementType = mapBufferFormatToElementType(info.format, info.itemsize);

  // TODO: Switch Tensor extents to ssize_t for efficiency.
  SmallVector<std::int32_t, 4> extents(info.shape.begin(), info.shape.end());
  return Tensor::create(
      refbackrt::ArrayRef<std::int32_t>(extents.data(), extents.size()),
      elementType, info.ptr);
}

py::array wrapTensorAsArray(Ref<Tensor> tensor) {
  auto pyTensor = py::cast(tensor);
  auto extents = tensor->getExtents();
  // TODO: Switch Tensor extents to ssize_t for efficiency.
  std::vector<ssize_t> shape(extents.data(), extents.data() + extents.size());

  const char *format;
  switch (tensor->getElementType()) {
  case refbackrt::ElementType::F32:
    format = "f";
    break;
  default:
    throw py::raiseValueError("unsupported tensor element type");
  }

  return py::array(py::dtype(format), shape, tensor->getData(),
                   /*base=*/std::move(pyTensor));
}

void npcomp::python::defineBackendRefJitModule(py::module &m) {
  m.def("build_backend_compilation_pipeline", [](MlirPassManager capiPm) {
    mlir::PassManager *pm = unwrap(capiPm);
    JITModule::buildBackendCompilationPipeline(*pm);
  });
  py::class_<JITModule>(m, "JITModule")
      .def_static(
          "from_compiled_module",
          [](MlirModule capiModule, std::vector<std::string> pySharedLibs)
              -> std::unique_ptr<JITModule> {
            SmallVector<StringRef, 4> sharedLibs(pySharedLibs.begin(),
                                                 pySharedLibs.end());
            auto module = unwrap(capiModule);
            auto jitModule =
                checkError(JITModule::fromCompiledModule(module, sharedLibs),
                           "error creating JITModule: ");
            return jitModule;
          },
          py::arg("module"), py::arg("shared_libs"))
      .def(
          "invoke",
          [](JITModule &self, std::string functionName,
             std::vector<py::buffer> inputs) {
            // Prepare inputs.
            llvm::SmallVector<Ref<Tensor>, 4> inputTensors;
            inputTensors.reserve(inputs.size());
            for (py::buffer &inputBuffer : inputs) {
              inputTensors.push_back(copyBufferToTensor(inputBuffer));
            }

            auto outputs = checkError(self.invoke(functionName, inputTensors),
                                      "error invoking JIT function: ");
            std::vector<py::array> outputArrays;
            outputArrays.reserve(outputs.size());
            for (Ref<Tensor> &outputTensor : outputs) {
              outputArrays.push_back(wrapTensorAsArray(outputTensor));
            }
            return outputArrays;
          },
          py::arg("function_name"), py::arg("inputs"));

  // A Ref<Tensor> needs to be bound because we use it as a base for the
  // ndarray (the array retains a reference to it). Users should not encounter
  // this unless if they go mucking through the array internals.
  py::class_<Ref<Tensor>>(m, "TensorRef");
}
