//===- example_mlir_backend_pybind.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch/csrc/jit/python/pybind.h"
#include "torch/csrc/lazy/backend/backend_interface.h"

#include <exception>
#include <iostream>
#include <string>

#include "backend/backend_impl.h"
#include "utils/sys_utils.h"

namespace py = pybind11;

namespace {
bool verbose = sys_util::GetEnv("VERBOSE", false);

struct NoGilSection {
  NoGilSection() : state(PyEval_SaveThread()) {}
  ~NoGilSection() { PyEval_RestoreThread(state); }
  PyThreadState *state = nullptr;
};

/**
 * @brief Install the plugin
 */
void Initialize() {
  // Initialize the Example MLIR LTC Backend
  torch::lazy::InitExampleMlirBackend();

  // sanity check
  const torch::lazy::BackendImplInterface *mlir_backend =
      torch::lazy::GetExampleMlirBackendImpl();
  const torch::lazy::BackendImplInterface *lazy_backend =
      torch::lazy::getBackend();
  if (lazy_backend != mlir_backend) {
    std::cout << "Failed to initialize MLIR Lazy Backend" << std::endl;
    throw std::runtime_error("Failed to initialize MLIR Lazy Backend");
  }

  if (verbose) {
    std::cout << "MLIR LTC PyTorch Plugin Initialized." << std::endl;
  }
}

/**
 * @brief Uninstall the plugin
 */
void Shutdown() {
  if (verbose) {
    std::cout << "MLIR LTC PyTorch Plugin Shut down." << std::endl;
  }
}
} // anonymous namespace

PYBIND11_MODULE(_EXAMPLE_MLIR_BACKEND, m) {
  m.doc() = ("pybind11 for example MLIR LTC backend.");
  m.def("_initialize", []() {
    NoGilSection gil;
    Initialize();
  });
  m.def("_shutdown", []() {
    NoGilSection gil;
    Shutdown();
  });
}
