//===- reference_lazy_backend_pybind.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch/csrc/jit/python/pybind.h"
#include "torch/csrc/lazy/backend/backend_interface.h"
#include "torch/csrc/lazy/core/config.h"

#include <base_lazy_backend/mlir_lowering_context.h>
#include <base_lazy_backend/utils/string_utils.h>
#include <base_lazy_backend/utils/sys_utils.h>
#include <base_lazy_backend/utils/tensor_utils.h>

#include <exception>
#include <iostream>
#include <string>

#include "backend_impl.h"

namespace py = pybind11;

namespace {
bool verbose = sys_util::GetEnv("VERBOSE", false);
bool ir_debug = sys_util::GetEnv("LTC_IR_DEBUG", false);

struct NoGilSection {
  NoGilSection() : state(PyEval_SaveThread()) {}
  ~NoGilSection() { PyEval_RestoreThread(state); }
  PyThreadState* state = nullptr;
};

/**
 * @brief Install the plugin
 */
void Initialize() {
  // Initialize the Reference Lazy Backend
  torch::lazy::InitReferenceLazyBackend();

  // sanity check
  const torch::lazy::BackendImplInterface* mlir_backend =
      torch::lazy::GetReferenceLazyBackendImpl();
  const torch::lazy::BackendImplInterface* lazy_backend =
      torch::lazy::getBackend();
  if (lazy_backend != mlir_backend) {
    std::cout << "Failed to initialize MLIR Lazy Backend" << std::endl;
    throw std::runtime_error("Failed to initialize MLIR Lazy Backend");
  }

  if (verbose) {
    std::cout << "MLIR LTC PyTorch Plugin Initialized." << std::endl;
  }

  if (ir_debug) {
    FLAGS_torch_lazy_ir_debug = true;
    std::cout << "Enabled lazy tensor IR debugging." << std::endl;
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

PYBIND11_MODULE(_REFERENCE_LAZY_BACKEND, m) {
  py::class_<torch::lazy::TorchMlirComputation>(m, "TorchMlirComputation")
      .def("to_string", &torch::lazy::TorchMlirComputation::to_string)
      .def("debug_string", &torch::lazy::TorchMlirComputation::debug_string);

  m.doc() = ("pybind11 for the Reference Lazy backend.");
  m.def("get_latest_computation", []() {
    auto computation = static_cast<torch::lazy::TorchMlirComputation*>(
        torch::lazy::GetLatestComputation().get());
    return py::cast(computation);
  });
  m.def(
      "set_parameter_name",
      [](const at::Tensor& tensor, const std::string& name) -> bool {
        torch::lazy::DeviceData* ir_node =
            torch::lazy::device_data_cast(tensor);
        if (ir_node) {
          ir_node->SetName(name);
          return true;
        }
        return false;
      });
  m.def("_initialize", []() {
    NoGilSection gil;
    Initialize();
  });
  m.def("_shutdown", []() {
    NoGilSection gil;
    Shutdown();
  });
}
