//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToTosa/TosaLegalizeCommon.h"
#include "torch-mlir/Conversion/TorchToTosa/TosaLegalizeUtils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"

#include <climits>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>

#include "mlir/Dialect/Quant/QuantTypes.h" // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h" // from @llvm-project
#include "mlir/Dialect/Tosa/IR/TosaOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/IR/Matchers.h"              // from @llvm-project
#include "mlir/IR/PatternMatch.h"          // from @llvm-project
#include "llvm/Support/FormatVariadic.h"

namespace mlir {
namespace tosa {

using namespace mlir::torch::Torch;

std::optional<Value>
createOneDimTfIndices(PatternRewriter &rewriter, Operation *op,
                      SmallVector<int64_t> indicesOneDimShape, int32_t dim,
                      ArrayRef<int64_t> indexShape) {
  unsigned indexRank = indexShape.size();
  SmallVector<int32_t> indicesVec;         // input vec to create tosaConstant
  SmallVector<int32_t> indicesMetaElement; // torch.meshgrid inputs
  int indicesMetaElementRepeatTimes{1};    // For torch.stack(torch.meshgrid)

  // Create torch.meshgrid inputs
  // Example: indexShape=[1,4,2]
  // dim0: indicesMetaElement = torch.arange(0, 1) = [0]
  // dim1: indicesMetaElement = torch.arange(0, 4) = [0,1,2,3]
  // dim2: indicesMetaElement = torch.arange(0, 2) = [0,1]
  for (int i = 0; i < indexShape[dim]; i++) {
    indicesMetaElement.push_back(i);
  }

  // Compute total number of meta element repeat times:
  // = product(indexShape[0:dim]) x product(indexShape[dim+1:-1]), skip dim
  // dim0: indicesMetaElementRepeatTimes = 1      x 4*2 = 8
  // dim1: indicesMetaElementRepeatTimes = 1 *1   x   2 = 2
  // dim2: indicesMetaElementRepeatTimes = 1 *1*4       = 4
  for (int i = 0; i < static_cast<int>(indexRank); i++) {
    if (i == dim) {
      continue;
    } else {
      indicesMetaElementRepeatTimes *= indexShape[i];
    }
  }

  if (dim != static_cast<int>(indexShape.size()) - 1) {
    // Create one dim indices for index except for last dim
    // Create indices raw vector.
    // torch.stack(torch.meshgrid)
    // dim0: indicesVec = [0 0 0 0 0 0 0 0]
    // dim0: indicesVec = [0 0 1 1 2 2 3 3]
    for (size_t elementId = 0; elementId < indicesMetaElement.size();
         elementId++) {
      for (int i = 0; i < indicesMetaElementRepeatTimes; i++) {
        indicesVec.push_back(indicesMetaElement[elementId]);
      }
    }
  } else { // Create the one dim indices for last dim of index
    // Create indices raw vector
    // dim2: indicesVec= [0 1 0 1 0 1 0 1]
    // Caution: indicesVec != [0 0 0 0 1 1 1 1]
    for (int i = 0; i < indicesMetaElementRepeatTimes; i++) {
      for (size_t elementId = 0; elementId < indicesMetaElement.size();
           elementId++) {
        indicesVec.push_back(indicesMetaElement[elementId]);
      }
    }
  }

  // Create tosa::ConstOp Tensor for indicesVec with target shape.
  // torch.unsqueeze(torch.stack(torch.meshgrid)))
  // dim0:          tensor([[   [ [0], [0] ],
  //			            	[ [0], [0] ],
  //			            	[ [0], [0] ],
  //			              	[ [0], [0] ], ]]) 1*4*2*1
  // dim1:	        tensor([[   [ [0], [0] ],
  //			             	[ [1], [1] ],
  //			            	[ [2], [2] ],
  //			             	[ [3], [3] ], ]]) 1*4*2*1
  // dim2/last dim:	tensor([[   [ [0], [1] ],
  //		                   	[ [0], [1] ],
  //			            	[ [0], [1] ],
  //		    	        	[ [0], [1] ], ]]) 1*4*2*1
  auto indicesDim = getConstTensor<int32_t>(rewriter, op,
                                            /*vec=*/indicesVec,
                                            /*shape=*/indicesOneDimShape);
  return indicesDim;
}

std::optional<Value> convertTorchIndexToTfIndices(PatternRewriter &rewriter,
                                                   Operation *op,
                                                   Value paramsValue,
                                                   Value indexValue,
                                                   int32_t axis) {
  // For easy understanding of this algorithm, the following comments are with
  // an exact example: torch.aten.gather(!torch.vtensor<[1,4,3],f32>, axis=2,
  // !torch.vtensor<[1,4,2],si64>) -> !torch.vtensor<[1,4,2],f32>
  // https://gist.github.com/AmosLewis/2f18434397025211da4491735bcc6db6
  //
  // Convert Torch Index     to       TF Indices
  //    [[                         [[   d0 d1  d2  d0 d1 d2
  //        [0,0],                     [[0, 0, 0],[0, 0, 0]],
  //        [1,0],                     [[0, 1, 1],[0, 1, 0]],
  //        [2,1],                     [[0, 2, 2],[0, 2, 1]],
  //        [2,1]                      [[0, 3, 2],[0, 3, 1]]
  //    ]] 1*4*2                   ]] 1*4*2*3

  auto paramsType = paramsValue.getType().dyn_cast<RankedTensorType>();
  auto indexType = indexValue.getType().dyn_cast<RankedTensorType>();
  auto paramsShape = paramsType.getShape(); // [1 4 3]
  auto indexShape = indexType.getShape();   // [1 4 2]
  int paramsRank = paramsShape.size();      // 3
  int indexRank = indexShape.size();        // 3

  // Initialize the final tf indices shape, and the shape of each dim that can
  // concat to this tf indices
  SmallVector<int64_t> indicesShape;       // [1 4 2 3]
  SmallVector<int64_t> indicesOneDimShape; // [1 4 2 1]
  for (auto shape : indexShape) {
    indicesShape.push_back(shape);
    indicesOneDimShape.push_back(shape);
  }
  indicesShape.push_back(paramsRank);
  indicesOneDimShape.push_back(1);

  // Get the chosen axis index
  // indexValue reshape to indicesDim: shape append 1
  // [1 4 2] -> [1 4 2 1]
  // dim2:	tensor([[   [ [0], [0] ],
  //			    [ [1], [0] ],
  //			    [ [2], [1] ],
  //			    [ [2], [1] ], ]]) 1*4*2*1
  auto indicesChosenAxis = tosa::CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, op->getLoc(),
      GetTypeFromTensorShape(indicesOneDimShape, indexType.getElementType()),
      indexValue, rewriter.getDenseI64ArrayAttr(indicesOneDimShape));

  SmallVector<Value> concatInputs;
  for (auto dim = 0; dim < paramsRank; dim++) {
    if (dim != axis) {
      auto indices = createOneDimTfIndices(rewriter, op, indicesOneDimShape,
                                           dim, indexShape);
      concatInputs.push_back(indices.value());
    } else {
      // the chosen axis indices will be replaced by index[i][j][k]
      concatInputs.push_back(indicesChosenAxis.getResult());
    }
  }

  // detailed example explanation
  // https://gist.github.com/AmosLewis/932a8dee3ba7657dcc6d09a4da4775d4 Get TF
  // indices: 1*4*2*3
  // [[  d0 d1  d2  d0 d1 d2
  //    [[0, 0, 0],[0, 0, 0]],
  //    [[0, 1, 1],[0, 1, 0]],
  //    [[0, 2, 2],[0, 2, 1]],
  //    [[0, 3, 2],[0, 3, 1]]
  // ]]
  auto indicesTf = tosa::CreateOpAndInfer<tosa::ConcatOp>(
      rewriter, op->getLoc(),
      GetTypeFromTensorShape(indicesShape, rewriter.getIntegerType(32)),
      concatInputs, indexRank);

  return indicesTf.getResult();
}

// Lowers Gather operators to a sequence of TOSA ops.
// taken from
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/tosa/transforms/legalize_common.cc
std::optional<Value> convertGatherNdOp(PatternRewriter &rewriter,
                                        Operation *op, Type outType,
                                        Value paramsValue, Value indicesValue) {
  auto resultType = outType.dyn_cast<ShapedType>();
  auto paramsType = paramsValue.getType().dyn_cast<RankedTensorType>();
  auto indicesType = indicesValue.getType().dyn_cast<RankedTensorType>();

  if (!resultType || !paramsType || !indicesType)
    return std::nullopt;

  // N: number of batches
  // Always 1 for GatherND
  //
  // Because TOSA's GATHER operator already uses the symbol 'N' for
  // the number of batches, we will use the symbol 'ND' to specify the
  // number of dimensions that are sliced from params instead of'N' in
  // the TF MLIR documentation.
  //
  // ND: indices.shape[-1]
  //
  // W: number of indices in each batch
  // Computed as:
  // product(indices.shape[0:-1]) (all but the last dimension)
  //
  // K: range of each index
  // Computed as:
  // product(params.shape[0:ND-1])
  //
  // C: number of channels for each index
  // Computed as:
  // product(params.shape[ND:])
  //
  // The params tensor needs to be reshaped, but not transposed, to move the
  // dimensions into [N, K, C] order.
  //
  // The dimensions of the input params[] tensor are grouped in the following
  // order to begin with:
  //
  //  [ParamIndices, ParamChannels]
  //  |------------||-------------|
  //         K              C
  //
  // The reshape simply flattens the params tensor into a 2D [K, C] shape.
  //
  // Indices needs to be put in the form of [N, W], but a simple flattening
  // will not suffice, because the indices need to index into a [W]-shape
  // vector instead of the params.shape[0:ND-1] tensor that we had before.
  //
  // To flatten the coordinates, first reshape indices to a [W, ND] matrix,
  // where the matrix now represents W ND-dimensional coordinates into the
  // params tensor.
  //
  // From here, we take each of the ND dimensions and multiply it with
  // the size of the next params dimension (or 1 for the last
  // dimension), then sum all these together with a reduce_sum
  // operator.  This is exactly the same mathematics as one would use
  // flatten the indices of an N-dimensional row-major array into a
  // 1-D array in C.
  //
  // More precisely, do an element-wise multiply with [params.shape[1
  // .. ND], 1] in axis 1, then reduce_sum in axis 1 to flatten to a
  // [W]-shaped tensor, then trivially reshape to [N=1, W] to be
  // compatible with the GATHER operator's shape.
  //
  // Then perform the tosa.GATHER() operation.
  //
  // Now we have result = [N, K, C].
  //
  // Reshape with a single, simple reshape to the final output shape of:
  //  [Indices, ParamChannels]
  //
  // Where, Indices is indices.shape[0:ND-1]
  //
  // For easy understanding, all following comments take an exact value for each
  // argument Example: Take TF style indices as input
  //    func.func @torch.aten.gather(%arg0: !torch.vtensor<[1,4,3],f32>,
  //        %arg1: !torch.vtensor<[1,4,2,3],i32>) -> !torch.vtensor<[1,4,2],f32>
  // Detail algorithm visualization:
  // https://gist.github.com/AmosLewis/bb6e3a0ad9fd1705c9f9d42a2eefbb88

  int N = 1, W = 1, K = 1, C = 1, ND = 1;

  int paramsRank = paramsType.getShape().size();   // 3
  int indicesRank = indicesType.getShape().size(); // 4

  //  ND: indices.shape[-1]
  ND = indicesType.getShape()[indicesRank - 1]; // 3

  if (ND > paramsRank) {
    (void)rewriter.notifyMatchFailure(
        op, "size of last dimension of indices must be <= params rank");
    return std::nullopt;
  }

  // Calculate N, K, W, C.  (N is always 1)
  // number of indices in each batch. product(indices.shape[0:-1]) (all but the
  // last dimension) W = 1*4*2 = 8
  for (int i = 0; i < (indicesRank - 1); i++) {
    W *= indicesType.getShape()[i];
  }

  // K: range of each index, total number of inputs(chould be gather) after
  // flattened k = 1*1*4*3 = 12
  for (int i = 0; i < ND; i++) {
    K *= paramsType.getShape()[i];
  }

  // C: number of channels for each index : numbers of values inside each
  // input(chould be gather) C = product(params.shape[ND:] ND = 3, paramsRank,
  // C = 1
  for (int i = ND; i < paramsRank; i++) {
    C *= paramsType.getShape()[i];
  }

  // int N = 1, W = 8, K = 12, C = 1, ND = 3;
  SmallVector<int64_t, 3> tosaValuesShape({N, K, C});  // {1,12,1}
  SmallVector<int64_t, 2> tosaIndicesShape({N, W});    // {1,8}
  SmallVector<int64_t, 2> indicesMatrixShape({W, ND}); // {8,3}
  SmallVector<int64_t, 2> indicesMatrixReducesumShape(
      {W, 1}); // {8,1} This is different from tf tosa code
  SmallVector<int64_t, 3> tosaGatherResultShape({N, W, C}); // {1,8,1}

  // %2 = "tosa.reshape"(%0) {new_shape = [1, 12, 1]} : (tensor<1x4x3xf32>) ->
  // tensor<1x12x1xf32>
  auto tosaValuesReshapeOp = tosa::CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, op->getLoc(),
      GetTypeFromTensorShape(tosaValuesShape, paramsType.getElementType()),
      paramsValue, rewriter.getDenseI64ArrayAttr(tosaValuesShape));

  // %3 = "tosa.reshape"(%1) {new_shape = [8, 3]} : (tensor<1x4x2x3xi32>) ->
  // tensor<8x3xi32> Flatten the input indices tensor to an [W, ND] matrix.
  auto indicesMatrixReshapeOp = tosa::CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, op->getLoc(),
      GetTypeFromTensorShape(indicesMatrixShape, indicesType.getElementType()),
      indicesValue, rewriter.getDenseI64ArrayAttr(indicesMatrixShape));

  SmallVector<int32_t> flattenedCoeffVec; //  [12,3,1]
  // flattenedCoeffVec = [4,3,1]
  for (int i = 1; i < ND; i++) {
    flattenedCoeffVec.push_back(paramsType.getShape()[i]);
  }
  flattenedCoeffVec.push_back(1);

  // flattenedCoeffVec = [12,3,1]
  for (int i = ND - 1; i > 0; i--) {
    flattenedCoeffVec[i - 1] *= flattenedCoeffVec[i];
  }

  // Create the tosaConstTensor for the flattenedCoeffVec
  // %4 = "tosa.const"() {value = dense<[12, 3, 1]> : tensor<3xi32>} : () ->
  // tensor<3xi32>
  auto flattenedCoeffValue =
      getConstTensor<int32_t>(rewriter, op, flattenedCoeffVec,
                              {static_cast<int64_t>(flattenedCoeffVec.size())});

  if (!flattenedCoeffValue)
    return std::nullopt;

  // Multiply the coefficients by the coordinates
  // %5 = "tosa.mul"(%3, %4) {shift = 0 : i32} : (tensor<8x3xi32>,
  // tensor<3xi32>) -> tensor<8x3xi32>
  auto flattenedIndicesMulOp = tosa::CreateOpAndInfer<tosa::MulOp>(
      rewriter, op->getLoc(),
      GetTypeFromTensorShape(indicesMatrixShape, indicesType.getElementType()),
      indicesMatrixReshapeOp.getResult(), flattenedCoeffValue.value(), 0);

  // Sum up the products of the coefficients and coordinates
  // %6 = "tosa.reduce_sum"(%5) {axis = 1 : i64} : (tensor<8x3xi32>) ->
  // tensor<8x1xi32>
  auto flattenedIndicesReduceOp = tosa::CreateOpAndInfer<tosa::ReduceSumOp>(
      rewriter, op->getLoc(),
      GetTypeFromTensorShape(indicesMatrixReducesumShape,
                             indicesType.getElementType()),
      flattenedIndicesMulOp.getResult(), rewriter.getI64IntegerAttr(1));

  // And reshape to [N, W]
  // %7 = "tosa.reshape"(%6) {new_shape = [1, 8]} : (tensor<8x1xi32>) ->
  // tensor<1x8xi32>
  auto tosaIndicesReshapeOp = tosa::CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, op->getLoc(),
      GetTypeFromTensorShape(tosaIndicesShape, indicesType.getElementType()),
      flattenedIndicesReduceOp.getResult(),
      rewriter.getDenseI64ArrayAttr(tosaIndicesShape));

  // Now the gather op itself
  // %9 = "tosa.gather"(%2, %7) : (tensor<1x12x1xf32>, tensor<1x8xi32>) ->
  // tensor<1x8x1xf32>
  auto tosaGatherOp = tosa::CreateOpAndInfer<tosa::GatherOp>(
      rewriter, op->getLoc(),
      GetTypeFromTensorShape(tosaGatherResultShape,
                             resultType.getElementType()),
      tosaValuesReshapeOp.getResult(), tosaIndicesReshapeOp.getResult());

  // Finally, reshape back to the original output shape of [Indices,
  // ParamChannels]. %10 = "tosa.reshape"(%9) {new_shape = [1, 4, 2]} :
  // (tensor<1x8x1xf32>) -> tensor<1x4x2xf32> %11 = torch_c.from_builtin_tensor
  // %10 : tensor<1x4x2xf32> -> !torch.vtensor<[1,4,2],f32>
  return tosa::CreateOpAndInfer<tosa::ReshapeOp>(
             rewriter, op->getLoc(), resultType, tosaGatherOp.getResult(),
             rewriter.getDenseI64ArrayAttr(resultType.getShape()))
      .getResult();
}

// Common function for lowering reduce operations to TOSA ops.
template <typename T>
std::optional<Value> convertReduceOpCommon(
    PatternRewriter &rewriter, Operation *op, RankedTensorType output_type,
    Value input_value, ElementsAttr axes_elems, bool keep_dims,
    Type reduce_element_type, bool is_quantized, double input_scale,
    int64_t input_zp, double output_scale, int64_t output_zp) {
  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type)
    return std::nullopt;

  ArrayRef<int64_t> input_shape = input_type.getShape();
  ArrayRef<int64_t> output_shape = output_type.getShape();
  auto input_rank = input_shape.size();
  Value val = input_value;

  if (axes_elems.getNumElements() == 0) {
    // No axes means return the original tensor.
    auto identity_op = CreateOpAndInfer<tosa::IdentityOp>(
        rewriter, op->getLoc(), output_type, val);
    val = identity_op.getResult();
  } else {
    // Reduce along each axis
    SmallVector<int64_t> shape_vec(input_shape.begin(), input_shape.end());

    if (is_quantized) {
      val = buildRescaleToInt32(rewriter, op, val, input_scale, input_zp);
    }

    for (int i = 0; i < axes_elems.getNumElements(); i++) {
      int64_t axis_val = axes_elems.getValues<IntegerAttr>()[i].getInt();
      if (axis_val < 0)
        axis_val += input_rank;
      auto axis_attr = rewriter.getI64IntegerAttr(axis_val);

      shape_vec[axis_val] = 1;
      RankedTensorType reduce_type = RankedTensorType::get(
          shape_vec,
          reduce_element_type);

      auto reduce_op = CreateOpAndInfer<T>(rewriter, op->getLoc(), reduce_type,
                                           val, axis_attr);

      val = reduce_op.getResult();
    }

    if (is_quantized) {
      RankedTensorType output_rescale_type =
          RankedTensorType::get(shape_vec, output_type.getElementType());
      val = buildRescale(rewriter, op, output_rescale_type, val, output_scale,
                         0, output_zp, false, true);
    }

    // Optionally squeeze out the reduced axes.
    if (!keep_dims) {
      auto reshape_op = CreateOpAndInfer<tosa::ReshapeOp>(
          rewriter, op->getLoc(), output_type, val,
          rewriter.getDenseI64ArrayAttr(output_shape));
      val = reshape_op.getResult();
    }
  }

  return val;
}

// Lowers ReduceAll to a sequence of TOSA ops.
std::optional<Value>
convertReduceAllOp(PatternRewriter &rewriter, Operation *op,
                   RankedTensorType output_type, Value input_value,
                   ElementsAttr axes_elems, bool keep_dims) {
  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type)
    return std::nullopt;

  return convertReduceOpCommon<tosa::ReduceAllOp>(
      rewriter, op, output_type, input_value, axes_elems, keep_dims,
      output_type.getElementType(), false, 1.0f, 0, 1.0f, 0);
}

// Lowers ReduceAny to a sequence of TOSA ops.
std::optional<Value>
convertReduceAnyOp(PatternRewriter &rewriter, Operation *op,
                   RankedTensorType output_type, Value input_value,
                   ElementsAttr axes_elems, bool keep_dims) {
  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type)
    return std::nullopt;

  return convertReduceOpCommon<tosa::ReduceAnyOp>(
      rewriter, op, output_type, input_value, axes_elems, keep_dims,
      output_type.getElementType(), false, 1.0f, 0, 1.0f, 0);
}

// Lowers ReduceMin to a sequence of TOSA ops.
std::optional<Value>
convertReduceMinOp(PatternRewriter &rewriter, Operation *op,
                   RankedTensorType output_type, Value input_value,
                   ElementsAttr axes_elems, bool keep_dims) {
  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type)
    return std::nullopt;

  return convertReduceOpCommon<tosa::ReduceMinOp>(
      rewriter, op, output_type, input_value, axes_elems, keep_dims,
      output_type.getElementType(), false, 1.0f, 0, 1.0f, 0);
}

// Lowers ReduceMax to a sequence of TOSA ops.
std::optional<Value>
convertReduceMaxOp(PatternRewriter &rewriter, Operation *op,
                   RankedTensorType output_type, Value input_value,
                   ElementsAttr axes_elems, bool keep_dims) {
  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type)
    return std::nullopt;

  return convertReduceOpCommon<tosa::ReduceMaxOp>(
      rewriter, op, output_type, input_value, axes_elems, keep_dims,
      output_type.getElementType(), false, 1.0f, 0, 1.0f, 0);
}

// Lowers ReduceProd to a sequence of TOSA ops.
std::optional<Value>
convertReduceProdOp(PatternRewriter &rewriter, Operation *op,
                    RankedTensorType output_type, Value input_value,
                    ElementsAttr axes_elems, bool keep_dims) {
  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type)
    return std::nullopt;

  bool input_is_qtype =
      input_type.getElementType().isa<mlir::quant::UniformQuantizedType>();
  bool output_is_qtype =
      output_type.getElementType().isa<mlir::quant::UniformQuantizedType>();

  if (input_is_qtype || output_is_qtype) {
    op->emitOpError("ConvertReduceProdOp: input/output tensor should "
                    "be all floating-point.");
    return std::nullopt;
  }

  return convertReduceOpCommon<tosa::ReduceProdOp>(
      rewriter, op, output_type, input_value, axes_elems, keep_dims,
      output_type.getElementType(), false, 1.0f, 0, 1.0f, 0);
}

// Lowers ReduceSum to a sequence of TOSA ops.
std::optional<Value>
convertReduceSumOp(PatternRewriter &rewriter, Operation *op,
                   RankedTensorType output_type, Value input_value,
                   ElementsAttr axes_elems, bool keep_dims) {
  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type)
    return std::nullopt;

  bool input_is_qtype =
      input_type.getElementType().isa<mlir::quant::UniformQuantizedType>();
  bool output_is_qtype =
      output_type.getElementType().isa<mlir::quant::UniformQuantizedType>();

  if (input_is_qtype != output_is_qtype) {
    op->emitOpError("ConvertReduceSumOp: input/output tensor should "
                    "be all quantized or all floating-point.");
    return std::nullopt;
  }

  double input_scale = 1.0f;
  double output_scale = 1.0f;
  int64_t input_zp = 0;
  int64_t output_zp = 0;
  Type reduce_element_type = input_type.getElementType();

  if (input_is_qtype) {
    auto input_qtype =
        input_type.getElementType().cast<mlir::quant::UniformQuantizedType>();
    auto output_qtype =
        output_type.getElementType().cast<mlir::quant::UniformQuantizedType>();

    int32_t input_shift = 20;

    input_scale =
        static_cast<double>(1 << input_shift) * input_qtype.getScale();
    output_scale =
        1.0 / (output_qtype.getScale() * static_cast<double>(1 << input_shift));

    input_zp = input_qtype.getZeroPoint();
    output_zp = output_qtype.getZeroPoint();
    reduce_element_type = rewriter.getI32Type();
  }

  return convertReduceOpCommon<tosa::ReduceSumOp>(
      rewriter, op, output_type, input_value, axes_elems, keep_dims,
      reduce_element_type, input_is_qtype, input_scale, input_zp, output_scale,
      output_zp);
}

// Lowers ReduceMean to a sequence of TOSA ops.
std::optional<Value>
convertReduceMeanOp(PatternRewriter &rewriter, Operation *op,
                    RankedTensorType output_type, Value input_value,
                    ElementsAttr axes_elems, bool keep_dims) {
  // reduce_mean is lowered as followed:
  // op1 = reduce_sum(input)
  // op2 = mul(op1, 1.0 / num_elements_on_reduced_axis)

  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type)
    return std::nullopt;

  bool input_is_qtype =
      input_type.getElementType().isa<mlir::quant::UniformQuantizedType>();
  bool output_is_qtype =
      output_type.getElementType().isa<mlir::quant::UniformQuantizedType>();

  if (input_is_qtype != output_is_qtype) {
    op->emitOpError("ConvertReduceSumOp: input/output tensor should "
                    "be all quantized or all floating-point.");
    return std::nullopt;
  }

  // Only supports float type mean() if it's non-quantized
  if (!input_is_qtype && !output_type.getElementType().isa<mlir::FloatType>()) {
    op->emitWarning(
        "Failed convertReduceMean: input unquantized type but output element "
        "not FloatType!");
    return std::nullopt;
  }

  int64_t input_rank = input_type.getRank();
  ArrayRef<int64_t> inputShape = input_type.getShape();
  int64_t num_elems_on_reduced_axis = 1;
  for (int i = 0; i < axes_elems.getNumElements(); i++) {
    int64_t axis_val = axes_elems.getValues<IntegerAttr>()[i].getInt();
    if (axis_val < 0)
      axis_val += input_rank;
    if (inputShape[axis_val] < 0)
      op->emitOpError("Failed convertReduceMean: support for dynamic input "
                      "shape not implemented");
    num_elems_on_reduced_axis *= inputShape[axis_val];
  }
  double div_scale = 1.0 / static_cast<double>(num_elems_on_reduced_axis);

  double input_scale = 1.0f;
  double output_scale = 1.0f;
  int64_t input_zp = 0;
  int64_t output_zp = 0;
  Type reduce_element_type = input_type.getElementType();

  if (input_is_qtype) {
    auto input_qtype =
        input_type.getElementType().cast<mlir::quant::UniformQuantizedType>();
    auto output_qtype =
        output_type.getElementType().cast<mlir::quant::UniformQuantizedType>();

    // Combine 'div_scale' as part of output rescale
    output_scale = div_scale * input_qtype.getScale() / output_qtype.getScale();

    input_zp = input_qtype.getZeroPoint();
    output_zp = output_qtype.getZeroPoint();
    reduce_element_type = rewriter.getI32Type();
  }

  auto val = convertReduceOpCommon<tosa::ReduceSumOp>(
      rewriter, op, output_type, input_value, axes_elems, keep_dims,
      reduce_element_type, input_is_qtype, input_scale, input_zp, output_scale,
      output_zp);

  if (!val.has_value())
    return std::nullopt;

  if (!input_is_qtype) {
    Value div_const = getTosaConstTensorSingleF32(rewriter, op, div_scale);
    return CreateOpAndInfer<tosa::MulOp>(rewriter, op->getLoc(), output_type,
                                         val.value(), div_const, 0)
        .getResult();
  }

  return val;
}

} // namespace tosa
} // namespace mlir
