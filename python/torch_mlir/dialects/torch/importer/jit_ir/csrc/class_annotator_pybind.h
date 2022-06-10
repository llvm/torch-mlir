//===- class_annotator_pybind.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
// Includes Torch-specific pybind and associated helpers.
// Depend on this for access to all Torch types (versus depending on pybind11
// directly).
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIRJITIRIMPORTER_CSRC_CLASS_ANNOTATOR_PYBIND_H
#define TORCHMLIRJITIRIMPORTER_CSRC_CLASS_ANNOTATOR_PYBIND_H

#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;
namespace torch_mlir {
void initClassAnnotatorBindings(py::module &m);
} // namespace torch_mlir

#endif // TORCHMLIRJITIRIMPORTER_CSRC_CLASS_ANNOTATOR_PYBIND_H
