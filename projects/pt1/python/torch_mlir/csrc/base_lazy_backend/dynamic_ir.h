//===- dynamic_ir.h -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
// This file is adapted from pytorch/pytorch
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/lazy/ts_backend/dynamic_ir.h
//===----------------------------------------------------------------------===//

#pragma once

#include <ATen/core/symbol.h>

#include <functional>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "mlir_node.h"
#include <c10/core/ScalarType.h>
#include <c10/util/Flags.h>
#include <torch/csrc/lazy/core/hash.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/ir_metadata.h>

C10_DECLARE_bool(ltc_enable_dynamic_shapes);

namespace torch {
namespace lazy {

/**
 * The goal of "dynamic" Nodes is to patch a hole in our tracing.
 * Previously, if a user called `sizes` on a Tensor, it would leak out
 * of our tracing system, as `sizes` returns a torch.Size or an int. To
 * prevent this from happening, we introduce DimensionNode, a new type
 * of Node that abstracts the operation of getting the dimensions of a
 * Tensor.
 *
 * Consider the following example:
 * ```
 * numel = x.shape()[0] * x.shape()[1]
 * ```
 *
 * Here, `x.shape()[i]` will be a SizeNode (subclass of DimensionNode),
 * and the multiplication of the two SizeNodes will be represented by
 * a SizeMul (also a subclass of DimensionNode). Through this, we can
 * prevent `numel` from being represented as a Python int and thus
 * burned into the Graph.
 */

class TORCH_API DimensionNode : public lazy::TorchMlirNode {
public:
  DimensionNode(OpKind op, OpList operands, hash_t hash_seed = kHashSeed);
  bool isDynamic() { return false; }

  std::string ToString() const override;

  virtual int64_t getStaticValue() const = 0;
};

// Represents the result of calling `size` on a Tensor
class TORCH_API SizeNode : public DimensionNode {
public:
  SizeNode(Value input, size_t dim);
  int64_t getStaticValue() const override;
  std::string ToString() const override;
  size_t dim_ = 0;
};

class TORCH_API SizeAdd : public DimensionNode {
public:
  SizeAdd(Value a, Value b);
  int64_t getStaticValue() const override;
  std::string ToString() const override;
};

class TORCH_API SizeMul : public DimensionNode {
public:
  SizeMul(Value a, Value b);
  int64_t getStaticValue() const override;
  std::string ToString() const override;
};

class TORCH_API SizeDiv : public DimensionNode {
public:
  SizeDiv(Value a, Value b);
  int64_t getStaticValue() const override;
  std::string ToString() const override;
};

} // namespace lazy
} // namespace torch
