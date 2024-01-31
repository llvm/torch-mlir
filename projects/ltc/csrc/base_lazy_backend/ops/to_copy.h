//===- to_copy.h ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
// this file is adapted from pytorch/pytorch
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/lazy/ts_backend/ops/to_copy.h
//===----------------------------------------------------------------------===//

#pragma once

#include "../mlir_node.h"

namespace torch {
namespace lazy {

// This IR was copied from code-generated output, but the entire _to_copy
// operator cannot be trivially code genereated since it is only desirable to
// capture IR for certain permutaions of _to_copy (e.g. dtype), and for the
// others it is difficult to even invoke the aten/eager fallback necessitating
// directly implementing the right to(device) behavior
class ToCopy : public torch::lazy::TorchMlirNode {
public:
  ToCopy(const torch::lazy::Value &self,
         const c10::optional<at::ScalarType> &dtype,
         const c10::optional<at::Layout> &layout,
         const c10::optional<at::Device> &device,
         const c10::optional<bool> &pin_memory, const bool &non_blocking,
         const c10::optional<at::MemoryFormat> &memory_format,
         std::vector<torch::lazy::Shape> &&shapes)
      : torch::lazy::TorchMlirNode(
            torch::lazy::OpKind(at::aten::_to_copy), {self}, std::move(shapes),
            /* num_outputs */ 1,
            torch::lazy::MHash(dtype, layout, device, pin_memory, non_blocking,
                               memory_format)),

        dtype(dtype), layout(layout), device(device), pin_memory(pin_memory),
        non_blocking(non_blocking), memory_format(memory_format) {}

  std::string ToString() const override {
    std::stringstream ss;
    ss << torch::lazy::TorchMlirNode::ToString();
    if (dtype.has_value()) {
      ss << ", dtype=" << dtype.value();
    } else {
      ss << ", dtype=null";
    }
    if (layout.has_value()) {
      ss << ", layout=" << layout.value();
    } else {
      ss << ", layout=null";
    }
    if (device.has_value()) {
      ss << ", device=" << device.value();
    } else {
      ss << ", device=null";
    }
    if (pin_memory.has_value()) {
      ss << ", pin_memory=" << pin_memory.value();
    } else {
      ss << ", pin_memory=null";
    }
    ss << ", non_blocking=" << non_blocking;
    if (memory_format.has_value()) {
      ss << ", memory_format=" << memory_format.value();
    } else {
      ss << ", memory_format=null";
    }
    return ss.str();
  }

  torch::lazy::TorchMlirOpVector
  Lower(TorchMlirFunction function,
        torch::lazy::TorchMlirLoweringContext *loctx) const override {
    std::vector<torch::jit::NamedValue> arguments;
    std::vector<torch::jit::NamedValue> kwarguments;
    arguments.reserve(1);
    kwarguments.reserve(6);
    size_t i = 0;
    arguments.emplace_back(loctx->GetOutputOp(operand(i++)));
    kwarguments.emplace_back("dtype", dtype);
    kwarguments.emplace_back("layout", layout);
    kwarguments.emplace_back("device", device);
    kwarguments.emplace_back("pin_memory", pin_memory);
    kwarguments.emplace_back("non_blocking", non_blocking);
    kwarguments.emplace_back("memory_format", memory_format);
    torch::lazy::TorchMlirOpVector _to_copy_out =
        torch::lazy::LowerTorchMlirBuiltin(function, op().op, shapes(),
                                           arguments, kwarguments);
    TORCH_CHECK_EQ(_to_copy_out.size(), 1);

    return _to_copy_out;
  }

  c10::optional<at::ScalarType> dtype;
  c10::optional<at::Layout> layout;
  c10::optional<at::Device> device;
  c10::optional<bool> pin_memory;
  bool non_blocking;
  c10::optional<at::MemoryFormat> memory_format;
};
} // namespace lazy
} // namespace torch
