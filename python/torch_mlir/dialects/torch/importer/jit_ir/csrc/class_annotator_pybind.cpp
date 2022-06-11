//===- class_annotator_pybind.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "class_annotator_pybind.h"
#include "class_annotator.h"

#include <torch/csrc/Dtype.h>
#include <torch/csrc/utils/pybind.h>

using namespace torch_mlir;

static c10::ScalarType convertToC10ScalarType(py::object obj) {
  if (THPDtype_Check(obj.ptr())) {
    // Need reinterpret_cast, since no C++-level inheritance is involved.
    THPDtype *dtype = reinterpret_cast<THPDtype *>(obj.ptr());
    return dtype->scalar_type;
  }
  std::stringstream ss;
  ss << "unsupported scalar type '" << obj << "'";
  throw std::invalid_argument(ss.str());
}

static std::vector<ArgAnnotation> getArgAnnotations(py::list pyArgAnnotations) {
  std::vector<ArgAnnotation> argAnnotations(pyArgAnnotations.size());
  for (int i = 0, e = argAnnotations.size(); i != e; i++) {
    if (pyArgAnnotations[i].is_none()) {
      continue;
    }
    auto tuple = py::cast<py::tuple>(pyArgAnnotations[i]);
    auto shape = tuple[0];
    auto dtype = tuple[1];
    auto hasValueSemantics = tuple[2];
    if (!shape.is_none()) {
      argAnnotations[i].shape = py::cast<std::vector<int64_t>>(shape);
    }
    if (!dtype.is_none()) {
      argAnnotations[i].dtype = convertToC10ScalarType(dtype);
    }
    argAnnotations[i].hasValueSemantics = py::cast<bool>(hasValueSemantics);
  };

  return argAnnotations;
}

void torch_mlir::initClassAnnotatorBindings(py::module &m) {
  py::class_<ClassAnnotator>(m, "ClassAnnotator")
      .def(py::init<>())
      .def("exportPath", &ClassAnnotator::exportPath)
      .def("exportNone", &ClassAnnotator::exportNone)
      .def("annotateArgs",
           [&](ClassAnnotator &cls_annotator, c10::ClassType &rootClassType,
               std::vector<std::string> path, py::list argAnnotations) {
             cls_annotator.annotateArgs(rootClassType, path,
                                        getArgAnnotations(argAnnotations));
           })
      .def("__repr__", &ClassAnnotator::toString);
}
