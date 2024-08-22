//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/Torch/Utils/SparsityUtils.h"
#include "mlir/Dialect/SparseTensor/IR/Enums.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>

using namespace mlir;
using namespace mlir::sparse_tensor;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

FailureOr<Attribute> Torch::getSparsityWithDenseLTAtDim(Attribute attr,
                                                        Value dim) {
  if (!attr)
    return Attribute();

  auto enc = cast<SparseTensorEncodingAttr>(attr);
  int64_t dimInt = 0;
  int64_t rank = enc.getDimRank() + 1;
  if (matchPattern(dim, m_TorchConstantInt(&dimInt))) {
    dimInt = toPositiveDim(dimInt, rank);
    if (!isValidDim(dimInt, rank)) {
      return failure();
    }
    if (!enc.isIdentity()) {
      // TODO: support block sparsity and permutation (CSC).
      return failure();
    }
    auto denseLT = *LevelType::buildLvlType(LevelFormat::Dense, true, true);
    SmallVector<LevelType> lvlTps = llvm::to_vector(enc.getLvlTypes());
    lvlTps.insert(lvlTps.begin() + dimInt, denseLT);
    auto dim2Lvl = AffineMap::getMultiDimIdentityMap(rank, attr.getContext());
    return SparseTensorEncodingAttr::get(
        enc.getContext(), lvlTps, dim2Lvl, AffineMap(), enc.getPosWidth(),
        enc.getCrdWidth(), enc.getExplicitVal(), enc.getImplicitVal());
  }
  // Do not know how to handle dynamic dimension.
  return failure();
}
