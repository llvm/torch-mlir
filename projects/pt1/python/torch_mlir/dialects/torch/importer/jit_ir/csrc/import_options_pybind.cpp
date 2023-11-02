//===- import_options_pybind.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "import_options_pybind.h"
#include "import_options.h"

namespace py = pybind11;

using namespace torch_mlir;

void torch_mlir::initImportOptionsBindings(py::module &m) {
  py::class_<ImportOptions>(m, "ImportOptions")
      .def(py::init<>())
      .def_readwrite("assumeTensorsHaveValueSemantics",
                     &ImportOptions::assumeTensorsHaveValueSemantics)
      .def_readwrite("ignoreExistingTensorShapesAndDtypes",
                     &ImportOptions::ignoreExistingTensorShapesAndDtypes);
}
