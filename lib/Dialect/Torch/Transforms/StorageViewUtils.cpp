//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "StorageViewUtils.h"
#include "CheckedInt.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include <algorithm>
#include <limits>
#include <optional>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
constexpr int64_t kUnknownStride = std::numeric_limits<int64_t>::min();

/// Shape, stride, and storage offset for a chain of view ops.
///
/// Unknown sizes and strides are allowed only while they are not needed to
/// compute a static storage offset or prove a static index bound.
struct ViewLayout {
  SmallVector<int64_t> sizes;
  SmallVector<int64_t> strides;
  int64_t offset = 0;

  /// Return the current logical rank.
  int64_t rank() const { return sizes.size(); }

  /// Replace shape and strides together.
  LogicalResult assign(SmallVector<int64_t> newSizes,
                       SmallVector<int64_t> newStrides) {
    if (newSizes.size() != newStrides.size())
      return failure();
    sizes = std::move(newSizes);
    strides = std::move(newStrides);
    return success();
  }

  /// Drop one logical dim from both shape and stride metadata.
  void eraseDim(int64_t dim) {
    sizes.erase(sizes.begin() + dim);
    strides.erase(strides.begin() + dim);
  }
};

/// Match a Torch constant int.
FailureOr<int64_t> constantInt(Value value) {
  int64_t result;
  if (!matchPattern(value, m_TorchConstantInt(&result)))
    return failure();
  return result;
}

/// Match a Torch list containing only constant ints.
FailureOr<SmallVector<int64_t>> constantIntList(Value list) {
  SmallVector<int64_t> values;
  if (!matchPattern(list, m_TorchListOfConstantInts(values)))
    return failure();
  return values;
}

/// Convert a Torch dim to `[0, rank)` form. The caller checks the range.
int64_t toPositiveStorageDim(int64_t dim, int64_t rank) {
  return dim >= 0 ? dim : dim + rank;
}

/// Return true for a dim already normalized into `[0, rank)`.
bool isValidStorageDim(int64_t dim, int64_t rank) {
  return dim >= 0 && dim < rank;
}

/// Match a constant dim operand and normalize it against `rank`.
FailureOr<int64_t> constantDim(Value value, int64_t rank) {
  FailureOr<int64_t> dim = constantInt(value);
  if (failed(dim))
    return failure();
  *dim = toPositiveStorageDim(*dim, rank);
  if (!isValidStorageDim(*dim, rank))
    return failure();
  return *dim;
}

/// Read the ranked tensor shape produced by a one-result op.
///
/// Dynamic sizes are preserved as `kUnknownSize`; each view op decides whether
/// the unknown dimensions matter for offset tracing.
FailureOr<SmallVector<int64_t>> resultSizes(Operation *op,
                                            int64_t expectedRank = -1) {
  if (op->getNumResults() != 1)
    return failure();
  auto type = dyn_cast<BaseTensorType>(op->getResult(0).getType());
  if (!type || !type.hasSizes() ||
      (expectedRank >= 0 && llvm::size(type.getSizes()) != expectedRank))
    return failure();
  return SmallVector<int64_t>(type.getSizes());
}

/// Compute dense row-major strides while allowing dynamic outer dimensions.
///
/// A stride is known when all inner dimensions are known. For example, `[?, 4]`
/// has strides `[4, 1]`, while `[2, ?, 4]` has `[?, 4, 1]`.
FailureOr<SmallVector<int64_t>>
computeDenseStridesWithUnknowns(ArrayRef<int64_t> sizes) {
  SmallVector<int64_t> strides(sizes.size(), kUnknownStride);
  std::optional<int64_t> running = int64_t{1};
  for (int64_t dim = static_cast<int64_t>(sizes.size()); dim-- > 0;) {
    strides[dim] = running.value_or(kUnknownStride);
    if (!running || sizes[dim] == Torch::kUnknownSize) {
      running = std::nullopt;
      continue;
    }
    // PyTorch's contiguous stride convention treats zero extents as one when
    // computing strides. Example: `torch.empty(2, 0, 3).stride()` is
    // `(3, 3, 1)`, not `(0, 3, 1)`.
    if (dim == 0)
      continue;
    FailureOr<int64_t> next =
        checkedMul(*running, std::max<int64_t>(sizes[dim], 1));
    if (failed(next))
      return failure();
    running = *next;
  }
  return strides;
}

/// Return true when every stride is known and dense row-major.
bool isKnownDenseLayout(const ViewLayout &view) {
  if (llvm::is_contained(view.strides, kUnknownStride))
    return false;
  FailureOr<SmallVector<int64_t>> denseStrides =
      computeDenseStridesWithUnknowns(view.sizes);
  return succeeded(denseStrides) &&
         !llvm::is_contained(*denseStrides, kUnknownStride) &&
         view.strides == *denseStrides;
}

/// Return a known dim size, or failure if the dim is dynamic.
FailureOr<int64_t> knownSize(const ViewLayout &view, int64_t dim) {
  if (view.sizes[dim] == Torch::kUnknownSize)
    return failure();
  return view.sizes[dim];
}

/// Return a known dim stride, or failure if the stride is dynamic.
FailureOr<int64_t> knownStride(const ViewLayout &view, int64_t dim) {
  if (view.strides[dim] == kUnknownStride)
    return failure();
  return view.strides[dim];
}

/// Multiply a known stride in place. Dynamic strides stay dynamic.
LogicalResult multiplyStride(int64_t &stride, int64_t factor) {
  if (stride == kUnknownStride)
    return success();
  FailureOr<int64_t> product = checkedMul(stride, factor);
  if (failed(product))
    return failure();
  stride = *product;
  return success();
}

/// Add `index * stride[dim]` to the storage offset.
///
/// A zero index contributes no offset even if the stride is dynamic. Non-zero
/// indices need a known stride so the final storage offset remains static.
LogicalResult addOffsetForDim(ViewLayout &view, int64_t dim, int64_t index) {
  if (index == 0)
    return success();
  FailureOr<int64_t> stride = knownStride(view, dim);
  if (failed(stride))
    return failure();
  FailureOr<int64_t> offset = checkedMulAdd(index, *stride, view.offset);
  if (failed(offset))
    return failure();
  view.offset = *offset;
  return success();
}

/// Replace the traced layout with explicit PyTorch size/stride metadata.
LogicalResult assignExplicit(ViewLayout &view, Value sizeList, Value strideList,
                             bool nonNegative) {
  FailureOr<SmallVector<int64_t>> sizes = constantIntList(sizeList);
  FailureOr<SmallVector<int64_t>> strides = constantIntList(strideList);
  if (failed(sizes) || failed(strides) || sizes->size() != strides->size())
    return failure();
  auto isNegative = [](int64_t value) { return value < 0; };
  if (nonNegative &&
      (llvm::any_of(*sizes, isNegative) || llvm::any_of(*strides, isNegative)))
    return failure();
  // The tracer uses sentinel values for dynamic metadata. Do not accept an
  // explicit layout that would be indistinguishable from an analysis unknown.
  if (llvm::is_contained(*sizes, Torch::kUnknownSize) ||
      llvm::is_contained(*strides, kUnknownStride))
    return failure();
  return view.assign(std::move(*sizes), std::move(*strides));
}

/// Permute sizes/strides without changing the underlying storage offset.
LogicalResult applyPermutation(ViewLayout &view, ArrayRef<int64_t> rawDims) {
  SmallVector<int64_t> dims = llvm::map_to_vector(rawDims, [&](int64_t rawDim) {
    return toPositiveStorageDim(rawDim, view.rank());
  });
  if (llvm::size(rawDims) != view.rank() || !isPermutationVector(dims))
    return failure();
  return view.assign(mlir::applyPermutation(view.sizes, dims),
                     mlir::applyPermutation(view.strides, dims));
}

/// Match and normalize a static index into one dimension.
///
/// A static extent is required because zero is not a valid select/narrow start
/// for every runtime size; empty tensors still reject select index zero.
FailureOr<int64_t> constantIndex(Value value, int64_t extent, bool allowEnd) {
  FailureOr<int64_t> index = constantInt(value);
  if (failed(index))
    return failure();
  if (extent == Torch::kUnknownSize)
    return failure();
  if (*index < 0) {
    *index += extent;
  }
  if (*index < 0 || *index > extent || (!allowEnd && *index == extent))
    return failure();
  return *index;
}

/// Split one logical dim and derive the strides for the new axes.
LogicalResult splitDim(ViewLayout &view, FailureOr<SmallVector<int64_t>> shape,
                       int64_t dim, int64_t splitRank) {
  if (failed(shape) || splitRank <= 0 || dim + splitRank > llvm::size(*shape))
    return failure();
  FailureOr<int64_t> originalSize = knownSize(view, dim);
  if (failed(originalSize))
    return failure();

  ArrayRef<int64_t> splitSizes =
      ArrayRef<int64_t>(*shape).slice(dim, splitRank);
  if (llvm::is_contained(splitSizes, Torch::kUnknownSize))
    return failure();
  FailureOr<int64_t> splitProduct = checkedProduct(splitSizes);
  if (failed(splitProduct))
    return failure();
  if (*splitProduct != *originalSize)
    return failure();

  SmallVector<int64_t> newStrides = view.strides;
  int64_t running = view.strides[dim];
  newStrides.erase(newStrides.begin() + dim);
  for (int64_t i = splitRank; i-- > 0;) {
    newStrides.insert(newStrides.begin() + dim, running);
    if (i == 0)
      continue;
    if (running != kUnknownStride) {
      FailureOr<int64_t> product =
          checkedMul(running, std::max<int64_t>((*shape)[dim + i], 1));
      if (failed(product))
        return failure();
      running = *product;
    }
  }
  return view.assign(*shape, std::move(newStrides));
}

/// Apply one view op to the traced shape/stride/offset state.
LogicalResult applyViewOp(Operation *op, ViewLayout &view) {
  auto getDim = [&](Value value) -> FailureOr<int64_t> {
    return constantDim(value, view.rank());
  };

  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case<AtenAsStridedOp>([&](AtenAsStridedOp op) {
        // Replace the current layout with as_strided's explicit size/stride.
        // `storage_offset=None` keeps the offset already accumulated from
        // earlier views.
        if (!isa<Torch::NoneType>(op.getStorageOffset().getType())) {
          FailureOr<int64_t> parsedOffset = constantInt(op.getStorageOffset());
          if (failed(parsedOffset) || *parsedOffset < 0)
            return failure();
          view.offset = *parsedOffset;
        }
        return assignExplicit(view, op.getSize(), op.getStride(),
                              /*nonNegative=*/true);
      })
      .Case<AtenAliasOp, AtenDetachOp, PrimsViewOfOp>([](auto) {
        // These ops only forward the same tensor storage.
        return success();
      })
      .Case<TensorStaticInfoCastOp>([&](TensorStaticInfoCastOp op) {
        // A static-info cast may erase information but must not invent a
        // narrower static size while walking backward.
        FailureOr<SmallVector<int64_t>> shape = resultSizes(op, view.rank());
        if (failed(shape))
          return failure();
        for (auto [currentSize, refinedSize] :
             llvm::zip_equal(view.sizes, *shape)) {
          if (currentSize == Torch::kUnknownSize &&
              refinedSize != Torch::kUnknownSize)
            return failure();
          if (currentSize != Torch::kUnknownSize &&
              refinedSize != Torch::kUnknownSize && currentSize != refinedSize)
            return failure();
        }
        return success();
      })
      .Case<AtenPermuteOp>([&](AtenPermuteOp op) {
        // Reorder logical axes; storage offset is unchanged.
        FailureOr<SmallVector<int64_t>> dims = constantIntList(op.getDims());
        return failed(dims) ? failure() : applyPermutation(view, *dims);
      })
      .Case<AtenTransposeIntOp>([&](AtenTransposeIntOp op) {
        // Swap two logical axes; storage offset is unchanged.
        FailureOr<int64_t> dim0 = getDim(op.getDim0());
        FailureOr<int64_t> dim1 = getDim(op.getDim1());
        if (failed(dim0) || failed(dim1))
          return failure();
        return applyPermutation(view, computePermutationVector(view.rank(),
                                                               {*dim0, *dim1},
                                                               {*dim1, *dim0}));
      })
      .Case<AtenTOp>([&](AtenTOp) {
        // t() is identity for rank <= 1 and a 2-D transpose for rank 2.
        if (view.rank() > 2)
          return failure();
        return view.rank() == 2 ? applyPermutation(view, {1, 0}) : success();
      })
      .Case<AtenMovedimIntOp>([&](AtenMovedimIntOp op) {
        // Move one axis to a new position; storage offset is unchanged.
        FailureOr<int64_t> src = getDim(op.getSource());
        FailureOr<int64_t> dst = getDim(op.getDestination());
        if (failed(src) || failed(dst))
          return failure();
        return applyPermutation(
            view, computePermutationVector(view.rank(), {*src}, {*dst}));
      })
      .Case<AtenNumpyTOp>([&](AtenNumpyTOp) {
        // numpy_T reverses all axes.
        return applyPermutation(view, llvm::to_vector(llvm::reverse(
                                          llvm::seq<int64_t>(0, view.rank()))));
      })
      .Case<AtenExpandOp, AtenExpandAsOp, AtenBroadcastToOp>([&](auto op) {
        // Match ATen expand geometry: right-aligned existing axes keep their
        // stride unless broadcasting from size 1, and inserted leading axes use
        // the next inner size/stride unless they are themselves broadcast.
        FailureOr<SmallVector<int64_t>> shape = resultSizes(op);
        if (failed(shape))
          return failure();
        int64_t resultRank = static_cast<int64_t>(shape->size());
        int64_t inputRank = view.rank();
        if (resultRank < inputRank)
          return failure();
        SmallVector<int64_t> newStrides(resultRank, kUnknownStride);
        for (int64_t resultDim = resultRank; resultDim-- > 0;) {
          int64_t inputDim = inputRank - (resultRank - resultDim);
          int64_t targetSize = (*shape)[resultDim];
          if (targetSize == Torch::kUnknownSize)
            return failure();

          int64_t inputSize = 1;
          int64_t inputStride = 0;
          if (inputDim >= 0) {
            inputSize = view.sizes[inputDim];
            inputStride = view.strides[inputDim];
            if (inputSize == Torch::kUnknownSize)
              return failure();
          } else if (resultDim + 1 < resultRank) {
            int64_t nextSize = (*shape)[resultDim + 1];
            int64_t nextStride = newStrides[resultDim + 1];
            if (nextSize == Torch::kUnknownSize ||
                nextStride == kUnknownStride) {
              inputStride = kUnknownStride;
            } else {
              FailureOr<int64_t> stride = checkedMul(nextSize, nextStride);
              if (failed(stride))
                return failure();
              inputStride = *stride;
            }
          }

          if (inputSize != targetSize) {
            if (inputSize != 1)
              return failure();
            inputStride = 0;
          }
          newStrides[resultDim] = inputStride;
        }
        return view.assign(*shape, std::move(newStrides));
      })
      .Case<AtenSliceTensorOp>([&](AtenSliceTensorOp op) {
        // Slice shifts the offset by start * stride and scales that axis stride
        // by step.
        FailureOr<int64_t> dim = getDim(op.getDim());
        FailureOr<int64_t> step = constantInt(op.getStep());
        FailureOr<SmallVector<int64_t>> shape = resultSizes(op);
        if (failed(dim) || failed(step) || failed(shape) || *step <= 0)
          return failure();

        FailureOr<int64_t> start = isa<Torch::NoneType>(op.getStart().getType())
                                       ? FailureOr<int64_t>(0)
                                       : constantInt(op.getStart());
        if (failed(start))
          return failure();
        if (*start < 0) {
          FailureOr<int64_t> dimSize = knownSize(view, *dim);
          if (failed(dimSize))
            return failure();
          *start += *dimSize;
        }
        if (view.sizes[*dim] != Torch::kUnknownSize)
          *start = std::clamp(*start, int64_t{0}, view.sizes[*dim]);
        else if (*start != 0)
          return failure();

        if (failed(addOffsetForDim(view, *dim, *start)) ||
            failed(multiplyStride(view.strides[*dim], *step)))
          return failure();
        view.sizes = *shape;
        return success();
      })
      .Case<AtenSelectIntOp>([&](AtenSelectIntOp op) {
        // Select fixes one index, adds its offset contribution, and drops the
        // selected axis.
        FailureOr<int64_t> dim = getDim(op.getDim());
        if (failed(dim))
          return failure();
        FailureOr<int64_t> index =
            constantIndex(op.getIndex(), view.sizes[*dim],
                          /*allowEnd=*/false);
        if (failed(index) || failed(addOffsetForDim(view, *dim, *index)))
          return failure();
        view.eraseDim(*dim);
        return success();
      })
      .Case<AtenNarrowOp>([&](AtenNarrowOp op) {
        // Narrow is slice with a start and length; only the start contributes
        // to storage offset.
        FailureOr<int64_t> dim = getDim(op.getDim());
        FailureOr<SmallVector<int64_t>> shape = resultSizes(op);
        if (failed(dim) || failed(shape))
          return failure();
        FailureOr<int64_t> start =
            constantIndex(op.getStart(), view.sizes[*dim],
                          /*allowEnd=*/true);
        if (failed(start) || failed(addOffsetForDim(view, *dim, *start)))
          return failure();
        view.sizes = *shape;
        return success();
      })
      .Case<AtenDiagonalOp>([&](AtenDiagonalOp op) {
        // Diagonal removes two axes and appends one axis with stride
        // stride(dim1) + stride(dim2).
        if (view.rank() < 2)
          return failure();
        FailureOr<int64_t> dim1 = getDim(op.getDim1());
        FailureOr<int64_t> dim2 = getDim(op.getDim2());
        FailureOr<SmallVector<int64_t>> shape =
            resultSizes(op, view.rank() - 1);
        FailureOr<int64_t> diagOffset = constantInt(op.getOffset());
        if (failed(dim1) || failed(dim2) || failed(shape) ||
            failed(diagOffset) || *dim1 == *dim2 || shape->empty())
          return failure();

        // An out-of-range diagonal is an empty view at the original storage
        // offset. Adding the offset would make `as_strided(empty_diag, ...)`
        // read from the wrong storage element.
        if (shape->back() == Torch::kUnknownSize && *diagOffset != 0)
          return failure();
        if (shape->back() != 0) {
          int64_t strideDim = *diagOffset >= 0 ? *dim2 : *dim1;
          FailureOr<int64_t> index = *diagOffset >= 0
                                         ? FailureOr<int64_t>(*diagOffset)
                                         : checkedSub(0, *diagOffset);
          if (failed(index))
            return failure();
          if (failed(addOffsetForDim(view, strideDim, *index)))
            return failure();
        }

        int64_t diagonalStride = kUnknownStride;
        if (view.strides[*dim1] != kUnknownStride &&
            view.strides[*dim2] != kUnknownStride) {
          FailureOr<int64_t> sum =
              checkedAdd(view.strides[*dim1], view.strides[*dim2]);
          if (failed(sum))
            return failure();
          diagonalStride = *sum;
        }
        view.strides.push_back(diagonalStride);
        view.strides.erase(view.strides.begin() + std::max(*dim1, *dim2));
        view.strides.erase(view.strides.begin() + std::min(*dim1, *dim2));
        view.sizes = *shape;
        return success();
      })
      .Case<AtenUnfoldOp>([&](AtenUnfoldOp op) {
        // Unfold keeps the source axis, scales it by step, and appends a window
        // axis using the original source stride.
        FailureOr<int64_t> dim = getDim(op.getDimension());
        FailureOr<SmallVector<int64_t>> shape =
            resultSizes(op, view.rank() + 1);
        FailureOr<int64_t> size = constantInt(op.getSize());
        FailureOr<int64_t> step = constantInt(op.getStep());
        if (failed(dim) || failed(shape) || failed(size) || failed(step) ||
            *size < 0 || shape->back() != *size || *step <= 0)
          return failure();
        int64_t windowStride = view.strides[*dim];
        if (failed(multiplyStride(view.strides[*dim], *step)))
          return failure();
        view.strides.push_back(windowStride);
        view.sizes = *shape;
        return success();
      })
      .Case<AtenSqueezeDimOp>([&](AtenSqueezeDimOp op) {
        // squeeze.dim removes the requested singleton axis or leaves layout
        // unchanged when PyTorch keeps a non-singleton axis.
        FailureOr<int64_t> dim = getDim(op.getDim());
        FailureOr<SmallVector<int64_t>> shape = resultSizes(op);
        if (failed(dim) || failed(shape))
          return failure();
        if (shape->size() == view.sizes.size() - 1) {
          view.eraseDim(*dim);
          return success();
        }
        return shape->size() == view.sizes.size() ? success() : failure();
      })
      .Case<AtenSqueezeOp>([&](AtenSqueezeOp) {
        // squeeze removes every statically-known singleton axis.
        for (int64_t dim = view.rank(); dim-- > 0;) {
          if (view.sizes[dim] == Torch::kUnknownSize)
            return failure();
          if (view.sizes[dim] == 1)
            view.eraseDim(dim);
        }
        return success();
      })
      .Case<PrimsSqueezeOp>([&](PrimsSqueezeOp op) {
        // prims.squeeze removes the explicit singleton axes.
        FailureOr<SmallVector<int64_t>> dims =
            constantIntList(op.getDimensions());
        if (failed(dims))
          return failure();
        for (int64_t &dim : *dims) {
          dim = toPositiveStorageDim(dim, view.rank());
          if (!isValidStorageDim(dim, view.rank()) ||
              view.sizes[dim] == Torch::kUnknownSize || view.sizes[dim] != 1)
            return failure();
        }
        llvm::sort(*dims);
        if (std::adjacent_find(dims->begin(), dims->end()) != dims->end())
          return failure();
        for (int64_t dim : llvm::reverse(*dims))
          view.eraseDim(dim);
        return success();
      })
      .Case<AtenUnsqueezeOp>([&](AtenUnsqueezeOp op) {
        // Unsqueeze inserts a singleton axis. Its stride follows PyTorch's
        // dense-view convention when the adjacent layout is known.
        FailureOr<int64_t> dim = constantInt(op.getDim());
        if (failed(dim))
          return failure();
        *dim = toPositiveStorageDim(*dim, view.rank() + 1);
        if (*dim < 0 || *dim > view.rank())
          return failure();
        int64_t stride = 1;
        if (*dim != view.rank()) {
          stride = kUnknownStride;
          if (view.sizes[*dim] != Torch::kUnknownSize &&
              view.strides[*dim] != kUnknownStride) {
            FailureOr<int64_t> product =
                checkedMul(view.sizes[*dim], view.strides[*dim]);
            if (failed(product))
              return failure();
            stride = *product;
          }
        }
        view.sizes.insert(view.sizes.begin() + *dim, 1);
        view.strides.insert(view.strides.begin() + *dim, stride);
        return success();
      })
      .Case<AtenUnflattenIntOp>([&](AtenUnflattenIntOp op) {
        // Unflatten splits one logical axis into the result shape axes.
        FailureOr<int64_t> dim = getDim(op.getDim());
        FailureOr<SmallVector<int64_t>> shape = resultSizes(op);
        if (failed(dim) || failed(shape))
          return failure();
        return splitDim(view, shape, *dim,
                        llvm::size(*shape) - view.rank() + 1);
      })
      .Case<PrimsSplitDimOp>([&](PrimsSplitDimOp op) {
        // prims.split_dim splits one logical axis into two axes.
        FailureOr<int64_t> dim = getDim(op.getDim());
        FailureOr<SmallVector<int64_t>> shape =
            resultSizes(op, view.rank() + 1);
        FailureOr<int64_t> outer = constantInt(op.getOuterLength());
        if (failed(dim) || failed(shape) || failed(outer) || *outer <= 0 ||
            (*shape)[*dim] != *outer)
          return failure();
        return splitDim(view, shape, *dim, /*splitRank=*/2);
      })
      .Case<Aten_ReshapeAliasOp>([&](Aten_ReshapeAliasOp op) {
        // _reshape_alias provides explicit size/stride metadata.
        return assignExplicit(view, op.getSize(), op.getStride(),
                              /*nonNegative=*/false);
      })
      .Case<AtenViewOp, AtenReshapeOp, Aten_UnsafeViewOp,
            AtenFlattenUsingIntsOp>([&](auto op) {
        // Dense reshapes preserve storage only when the input layout is already
        // known dense.
        FailureOr<SmallVector<int64_t>> shape = resultSizes(op);
        if (failed(shape) || !isKnownDenseLayout(view))
          return failure();
        FailureOr<SmallVector<int64_t>> strides =
            computeDenseStridesWithUnknowns(*shape);
        if (failed(strides))
          return failure();
        return view.assign(*shape, std::move(*strides));
      })
      .Default([](Operation *) { return failure(); });
}

/// Return true for ops handled by applyViewOp.
///
/// Every op in this list must use operand 0 as the tensor whose storage is
/// viewed. Ops with another storage operand order need a dedicated case in
/// traceViewLikeStorageBase.
bool isSupportedStorageViewOp(Operation *op) {
  return isa<AtenAsStridedOp, AtenAliasOp, AtenBroadcastToOp, AtenDetachOp,
             AtenDiagonalOp, AtenExpandAsOp, AtenExpandOp,
             AtenFlattenUsingIntsOp, AtenMovedimIntOp, AtenNarrowOp,
             AtenNumpyTOp, AtenPermuteOp, AtenReshapeOp, AtenSelectIntOp,
             AtenSliceTensorOp, AtenSqueezeDimOp, AtenSqueezeOp, AtenTOp,
             AtenTransposeIntOp, AtenUnflattenIntOp, AtenUnfoldOp,
             AtenUnsqueezeOp, AtenViewOp, Aten_ReshapeAliasOp,
             Aten_UnsafeViewOp, PrimsSplitDimOp, PrimsSqueezeOp, PrimsViewOfOp,
             TensorStaticInfoCastOp>(op);
}

/// Return true for ops that stop the backward walk because they allocate/copy.
bool isStorageBoundaryOp(Operation *op) { return isa<AtenCloneOp>(op); }

/// Return true for ops that cannot be treated as storage-preserving views.
///
/// These ops may allocate, copy, reinterpret element storage, or hide layout
/// information. Stop tracing instead of guessing that operand 0 is still the
/// same storage.
bool isUnsupportedStorageViewOp(Operation *op) {
  return isa<AtenChannelShuffleOp, AtenContiguousOp, AtenImagOp,
             AtenNarrowTensorOp, AtenPixelShuffleOp, AtenPixelUnshuffleOp,
             AtenRealOp, AtenToDeviceOp, AtenToDtypeLayoutOp, AtenToDtypeOp,
             AtenViewAsComplexOp, AtenViewAsRealOp>(op);
}

} // namespace

FailureOr<StorageViewBase> Torch::traceViewLikeStorageBase(Value input) {
  SmallVector<Operation *> viewOps;
  Value base = input;
  while (Operation *op = base.getDefiningOp()) {
    // isSupportedStorageViewOp guarantees operand 0 is the storage source.
    if (isSupportedStorageViewOp(op)) {
      viewOps.push_back(op);
      base = op->getOperand(0);
      continue;
    }
    if (isUnsupportedStorageViewOp(op))
      return failure();
    if (isStorageBoundaryOp(op))
      break;
    break;
  }
  if (viewOps.empty())
    return StorageViewBase{base, 0};

  auto type = dyn_cast<BaseTensorType>(base.getType());
  if (!type || !type.hasSizes())
    return failure();
  FailureOr<SmallVector<int64_t>> baseStrides =
      computeDenseStridesWithUnknowns(type.getSizes());
  if (failed(baseStrides))
    return failure();

  // Torch tensor types do not carry stride metadata. Treat the traced base as
  // dense row-major storage; non-dense layout must come from the view suffix.
  ViewLayout view{SmallVector<int64_t>(type.getSizes()),
                  std::move(*baseStrides)};
  for (Operation *op : llvm::reverse(viewOps))
    if (failed(applyViewOp(op, view)))
      return failure();
  return StorageViewBase{base, view.offset};
}
