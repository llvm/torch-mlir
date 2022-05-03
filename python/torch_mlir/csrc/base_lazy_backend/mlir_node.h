//===- mlir_node.h --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
// This file is adapted from pytorch/pytorch
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/lazy/ts_backend/ts_node.h
//===----------------------------------------------------------------------===//

#pragma once

#include <ATen/core/interned_strings.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/lazy/backend/lowering_context.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/shape.h>

#include "../utils/debug.h"
#include "../utils/exception.h"
#include "mlir_lowering_context.h"

namespace torch {
namespace lazy {

class TORCH_API TorchMlirNode : public torch::lazy::Node {
public:
  TorchMlirNode(
      OpKind op, OpList operands, std::vector<Shape>&& shapes,
      size_t num_outputs, hash_t hash_seed = kHashSeed);

  TorchMlirNode(
      OpKind op, OpList operands, const std::function<Shape()>& shape_fn,
      size_t num_outputs, hash_t hash_seed = kHashSeed);

  TorchMlirNode(
      OpKind op, OpList operands, size_t num_outputs,
      hash_t hash_seed = kHashSeed);

  TorchMlirNode(
      OpKind op, Shape shape, size_t num_outputs, hash_t hash_seed = kHashSeed);

  hash_t hash() const override;

  hash_t shapeHash() const override;

  virtual TorchMlirOpVector
  Lower(TorchMlirFunction function, TorchMlirLoweringContext* loctx) const;

private:
  // The hash of the dag WITH size info. Used for shape caching
  hash_t shape_hash_;
  // The hash of the dag used to look up the compiled graph by a hash
  // in this case, we will use the dag hash WITHOUT size info if dynamic shape
  // is enabled and use the dag hash WITH size info otherwise.
  hash_t dag_hash_;
};

// Note: this OpKind is separate from ltc_ops.h since it would be a circular
// import otherwise
const OpKind tensor_list_opkind = OpKind::Get("lazy_tensors::tensor_list");

// TensorList represents an at::TensorList which is a vector[Tensor] but is also
// a first-class IValue and can be fed as a single input to a TS program.  It is
// much easier to handle TensorLists in Lazy Tensor code if they are represented
// as a single Node so there can be more than one TensorList and more than one
// Tensor side-by-side as operands to an op.
//
// Note: shape is undefined for TensorList.  We assert in some places that
// #shapes matches #outputs and this stems from
//       the fact that currently all IR nodes represent tensors (there is no
//       type system for this IR).  Becuase of this, TensorList is a bit of a
//       hack.
//
// TODO(whc) once Shape() API is moved to Node base, also make it virtual, and
// then implement it as NotImplemented for TensorList, also fixing the assertion
// that would fail.
struct TORCH_API TensorList : public TorchMlirNode {
  TensorList() = delete;
  TensorList(OpList values);

  torch::lazy::TorchMlirOpVector Lower(
      TorchMlirFunction function,
      TorchMlirLoweringContext* loctx) const override;
};

} // namespace lazy
} // namespace torch
