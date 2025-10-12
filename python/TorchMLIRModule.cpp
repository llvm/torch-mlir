//===-- TorchBind.td - Torch dialect bind ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "torch-mlir-c/Dialects.h"
#include "torch-mlir-c/Registration.h"

namespace py = pybind11;

PYBIND11_MODULE(_torchMlir, m) {
  torchMlirRegisterAllPasses();

  m.doc() = "torch-mlir main python extension";

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__torch__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context"), py::arg("load") = true);
}
