//===- ATenDialectOpStats.cpp -----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/ATen/IR/ATenDialect.h"
#include "npcomp/Dialect/ATen/IR/ATenOpStatisticsUtils.h"

#include "llvm/Support/Debug.h"

#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"

#include <iostream>

#define DEBUG_TYPE "aten-op-stats"

// This file contains the StatisticsOpInterface implementations
// for ATDialect operations

using namespace mlir;

namespace {

std::vector<uint64_t> unpackListConstant(Value op) {
  std::vector<uint64_t> v;
  auto co = cast<mlir::NPCOMP::aten::ConstantOp>(op.getDefiningOp());
  DenseElementsAttr a = co.template getAttrOfType<DenseElementsAttr>("value");
  for (auto i : a.getIntValues())
    v.push_back(i.getSExtValue());
  return v;
};

} // namespace

namespace mlir {
namespace NPCOMP {
namespace aten {

std::map<std::string, uint64_t> AdaptiveAvgPool2dOp::getStatistics() {
  std::map<std::string, uint64_t> toReturn;
  // FIXME: unimplemented
  toReturn["reads"] = -1;
  toReturn["writes"] = -1;
  return toReturn;
}
std::map<std::string, uint64_t> AdaptiveAvgPool2dBackwardOp::getStatistics() {
  std::map<std::string, uint64_t> toReturn;
  // FIXME: unimplemented
  toReturn["reads"] = -1;
  toReturn["writes"] = -1;
  return toReturn;
}

// add_
std::map<std::string, uint64_t> AddUnderOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = getResult().getType().cast<TensorType>();
  TensorType aType = getOperand(0).getType().cast<TensorType>();
  Type bType = getOperand(1).getType();

  uint64_t ofm_volume = getTensorVolume(resultTy);

  toReturn["ops:+"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = getTensorVolume(aType);
  uint64_t b_volume = getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

// addmm
std::map<std::string, uint64_t> AddmmOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  // For linear, we need the number of output neurons and the number of input
  // neurons Then the number of forward MACs is input * output And the number of
  // adds is output if there is bias

  TensorType resultTy = getResult().getType().cast<TensorType>();
  TensorType biasTy = getOperand(0).getType().cast<TensorType>();
  TensorType inputTy = getOperand(1).getType().cast<TensorType>();
  TensorType weightTy = getOperand(2).getType().cast<TensorType>();

  uint64_t num_output_neurons = resultTy.getShape()[1];
  uint64_t ofm_volume = getTensorVolume(resultTy);

  // Use the weight tensor to find the number of input neurons
  uint64_t num_input_neurons = weightTy.getShape()[0];
  uint64_t total_MACs = ofm_volume * num_input_neurons;
  uint64_t weight_volume = getTensorVolume(weightTy);

  uint64_t ifm_volume = getTensorVolume(inputTy);

  toReturn["ops:MAC"] = total_MACs;
  toReturn["ops:+"] =
      ofm_volume; // Should be gated on whether there is bias at all
  toReturn["operand:1:activation_in"] = ifm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;
  toReturn["operand:0:parameters_in:bias"] = getTensorVolume(biasTy);
  toReturn["operand:2:parameters_in:weight"] = weight_volume;

  toReturn["reads"] = ifm_volume + weight_volume + num_output_neurons;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

// as_strided can be zero overhead
std::map<std::string, uint64_t> AsStridedOp::getStatistics() {
  std::map<std::string, uint64_t> toReturn;
  toReturn["reads"] = 0;
  toReturn["writes"] = 0;
  toReturn["operand:0:activation_in"] = 0;
  toReturn["result:0:activation_out"] = 0;
  return toReturn;
}

// batch_norm
std::map<std::string, uint64_t> BatchNormOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = getResult(0).getType().cast<TensorType>();
  uint64_t op_volume = getTensorVolume(resultTy);
  uint64_t weight_volume = getTensorVolume(getOperand(1).getType());
  uint64_t bias_volume = getTensorVolume(getOperand(2).getType());
  toReturn["operand:0:activation_in"] = op_volume;
  toReturn["result:0:activation_out"] = op_volume;
  toReturn["operand:1:parameters_in:weight"] = weight_volume;
  toReturn["operand:2:parameters_in:bias"] = bias_volume;

  // Now for the arithmetic.  Assume variance is calculated as sum of squares
  uint64_t ifm_depth = resultTy.getShape()[1];

  toReturn["ops:+"] = op_volume;  // Add up for mean
  toReturn["ops:*"] = op_volume;  // Square for variance
  toReturn["ops:+"] += op_volume; // Add up squares for variance

  toReturn["ops:*"] += ifm_depth; // Calc channel means
  toReturn["ops:-"] += ifm_depth; // Calc channel vars
  toReturn["ops:*"] += ifm_depth; // Calc channel vars

  toReturn["ops:sqrt"] = ifm_depth; // Convert to SD
  toReturn["ops:/"] = ifm_depth;    // Get the reciprocal

  toReturn["ops:+"] += op_volume; // Subtract mean off each pixel
  toReturn["ops:*"] += op_volume; // Multiply by 1/SD for each pixel

  toReturn["ops:+"] += op_volume; // Bias
  toReturn["ops:*"] += op_volume; // Scale

  toReturn["reads"] = op_volume + weight_volume + bias_volume;
  toReturn["writes"] = op_volume;

  return toReturn;
}

// div_
std::map<std::string, uint64_t> DivUnderOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = getResult().getType().cast<TensorType>();
  TensorType aType = getOperand(0).getType().cast<TensorType>();
  Type bType = getOperand(1).getType();

  uint64_t ofm_volume = getTensorVolume(resultTy);
  toReturn["ops:/"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = getTensorVolume(aType);
  uint64_t b_volume = getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

// expand can be zero overhead
std::map<std::string, uint64_t> ExpandOp::getStatistics() {
  std::map<std::string, uint64_t> toReturn;
  toReturn["reads"] = toReturn["operand:0:activation_in"] = 0;
  toReturn["writes"] = toReturn["result:0:activation_out"] = 0;
  return toReturn;
}

// flatten can be zero overhead
std::map<std::string, uint64_t> FlattenOp::getStatistics() {
  std::map<std::string, uint64_t> toReturn;
  toReturn["reads"] = toReturn["operand:0:activation_in"] = 0;
  toReturn["writes"] = toReturn["result:0:activation_out"] = 0;
  return toReturn;
}

std::map<std::string, uint64_t> GatherOp::getStatistics() {
  std::map<std::string, uint64_t> toReturn;
  // FIXME: unimplemented
  toReturn["reads"] = -1;
  toReturn["writes"] = -1;
  return toReturn;
}

// hardtanh
std::map<std::string, uint64_t> HardtanhOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType inputTy = getOperand(0).getType().cast<TensorType>();
  TensorType resultTy = getResult().getType().cast<TensorType>();

  uint64_t in_volume = getTensorVolume(inputTy);
  uint64_t out_volume = getTensorVolume(resultTy);

  toReturn["operand:0:activation_in"] = in_volume;
  toReturn["result:0:activation_out"] = out_volume;
  toReturn["reads"] = in_volume;
  toReturn["writes"] = out_volume;
  toReturn["ops:>"] = out_volume;

  return toReturn;
}

// hardtanh_
std::map<std::string, uint64_t> HardtanhUnderOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType inputTy = getOperand(0).getType().cast<TensorType>();
  TensorType resultTy = getResult().getType().cast<TensorType>();

  uint64_t in_volume = getTensorVolume(inputTy);
  uint64_t out_volume = getTensorVolume(resultTy);

  toReturn["operand:0:activation_in"] = in_volume;
  toReturn["result:0:activation_out"] = out_volume;
  toReturn["reads"] = in_volume;
  toReturn["writes"] = out_volume;
  toReturn["ops:>"] = out_volume;

  return toReturn;
}

std::map<std::string, uint64_t> HardtanhBackwardOp::getStatistics() {
  std::map<std::string, uint64_t> toReturn;
  // FIXME: unimplemented
  return toReturn;
}

// max_pool2d
std::map<std::string, uint64_t> MaxPool2dOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = getResult().getType().cast<TensorType>();
  TensorType inputType = getOperand(0).getType().cast<TensorType>();

  uint64_t ofm_volume = getTensorVolume(resultTy);
  toReturn["result:0:activation_out"] = ofm_volume;

  uint64_t ifm_volume = getTensorVolume(inputType);
  toReturn["input:0:activation_in"] = ifm_volume;

  // To find the number of compares, we need the filter extent

  std::vector<uint64_t> kernel_size = unpackListConstant(getOperand(1));

  uint64_t aperture = kernel_size[0] * kernel_size[1];
  toReturn["ops:>"] = ofm_volume * (aperture - 1);

  toReturn["reads"] = ifm_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

// max_pool2d_with_indices
std::map<std::string, uint64_t> MaxPool2dWithIndicesOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  uint64_t ofm_volume =
      getTensorVolume(getResult(0).getType().cast<TensorType>());
  uint64_t indices_volume =
      getTensorVolume(getResult(1).getType().cast<TensorType>());

  toReturn["writes"] = ofm_volume + indices_volume;
  toReturn["result:0:activation_out"] = ofm_volume;
  toReturn["result:1:indices_out"] = indices_volume;

  uint64_t ifm_volume =
      getTensorVolume(getOperand(0).getType().cast<TensorType>());
  toReturn["reads"] = ifm_volume;
  toReturn["operand:0:activation_in"] = ifm_volume;

  // To find the number of compares, we need the filter extent

  std::vector<uint64_t> kernel_size = unpackListConstant(getOperand(1));

  uint64_t aperture = kernel_size[0] * kernel_size[1];
  toReturn["ops:>"] = ofm_volume * (aperture - 1);

  return toReturn;
}

// max_pool2d_with_indices_backward
std::map<std::string, uint64_t>
MaxPool2dWithIndicesBackwardOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  Type resultTy = getResult().getType();
  TensorType tensorResultTy = resultTy.cast<TensorType>();
  uint64_t loss_out_volume = getTensorVolume(tensorResultTy);
  toReturn["writes"] = loss_out_volume;

  uint64_t loss_in_volume =
      getTensorVolume(getOperand(0).getType().cast<TensorType>());
  uint64_t act_in_volume = getTensorVolume(
      getOperand(1).getType().cast<TensorType>()); // TODO: Why is this needed?
  uint64_t indices_volume =
      getTensorVolume(getOperand(7).getType().cast<TensorType>());
  toReturn["reads"] = loss_in_volume + act_in_volume + indices_volume;
  toReturn["operand:0:activation_in"] = loss_in_volume;
  toReturn["operand:1:activation_in"] = act_in_volume;
  toReturn["operand:3:activation_in"] = indices_volume;
  toReturn["result:0:grad:dx"] = loss_out_volume;

  return toReturn;
}

// mean
std::map<std::string, uint64_t> MeanOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = getResult().getType().cast<TensorType>();
  TensorType aType = getOperand().getType().cast<TensorType>();

  uint64_t ofm_volume = getTensorVolume(resultTy);
  toReturn["ops:+"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = getTensorVolume(aType);

  toReturn["operand:0:activation_in"] = a_volume;

  toReturn["reads"] = a_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

// mm
// std::map<std::string, uint64_t> MMOp::getStatistics() {
//   getMMOpStatistics(*this);
// }
std::map<std::string, uint64_t> MmOp::getStatistics() {
  return getMMOpStatistics(*this);
}

// mul_
std::map<std::string, uint64_t> MulUnderOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = getResult().getType().cast<TensorType>();
  TensorType aType = getOperand(0).getType().cast<TensorType>();
  Type bType = getOperand(1).getType();

  uint64_t ofm_volume = getTensorVolume(resultTy);
  toReturn["ops:*"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = getTensorVolume(aType);
  uint64_t b_volume = getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

// native_batch_norm
std::map<std::string, uint64_t> NativeBatchNormOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = getResult(0).getType().cast<TensorType>();
  uint64_t op_volume = getTensorVolume(resultTy);
  uint64_t weight_volume = getTensorVolume(getOperand(1).getType());
  uint64_t bias_volume = getTensorVolume(getOperand(2).getType());
  toReturn["operand:0:activation_in"] = op_volume;
  toReturn["result:0:activation_out"] = op_volume;
  toReturn["operand:1:parameters_in:weight"] = weight_volume;
  toReturn["operand:2:parameters_in:bias"] = bias_volume;

  // Now for the arithmetic.  Assume variance is calculated as sum of squares
  uint64_t ifm_depth = resultTy.getShape()[1];

  toReturn["ops:+"] = op_volume;  // Add up for mean
  toReturn["ops:*"] = op_volume;  // Square for variance
  toReturn["ops:+"] += op_volume; // Add up squares for variance

  toReturn["ops:*"] += ifm_depth; // Calc channel means
  toReturn["ops:-"] += ifm_depth; // Calc channel vars
  toReturn["ops:*"] += ifm_depth; // Calc channel vars

  toReturn["ops:sqrt"] = ifm_depth; // Convert to SD
  toReturn["ops:/"] = ifm_depth;    // Get the reciprocal

  toReturn["ops:+"] += op_volume; // Subtract mean off each pixel
  toReturn["ops:*"] += op_volume; // Multiply by 1/SD for each pixel

  toReturn["ops:+"] += op_volume; // Bias
  toReturn["ops:*"] += op_volume; // Scale

  toReturn["reads"] = op_volume + weight_volume + bias_volume;
  toReturn["writes"] = op_volume;

  return toReturn;
}

// batchnorm backward
std::map<std::string, uint64_t> NativeBatchNormBackwardOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  ShapedType inputTy = getOperand(0).getType().cast<ShapedType>();
  uint64_t input_volume = getTensorVolume(inputTy);
  uint64_t input_channels = inputTy.getShape()[1];

  // # 3 components make up the gradInput: 1 gradInput, 2 gradMean, 3 gradVar
  // # totalGradInput = gradInput + (dL / dMean * dMean / dInput) +
  // #                  (dL / dVar * dVar / dInput)

  // # gradInput
  // total_ops["backward"]["*"] = in_c * (in_h*in_w*batch_size) # scale
  // # Bootstrap from previous
  // #total_ops["backward"]["sqrt"] = in_c # Convert to std_dev
  // #total_ops["backward"]["/"] = in_c # Calculate inverse sqrt first
  toReturn["ops:*"] = input_volume; // scale

  // # dL / dGradVar
  // total_ops["backward"]["pow"] = in_c
  // total_ops["backward"]["*"] = total_ops["backward"]["*"] + in_c
  // #total_ops["backward"]["+"] = total_ops["backward"]["+"] + in_c *
  // in_h*in_w*batch_size # Subtract mean, bootstrap from previous calculation
  // total_ops["backward"]["*"] = total_ops["backward"]["*"] + in_c *
  // (in_h*in_w*batch_size)
  toReturn["ops:pow"] = input_channels;
  ;
  toReturn["ops:*"] += input_channels;
  toReturn["ops:*"] += input_volume;

  // # dL / dGradMean
  // #total_ops["backward"]["+"] = total_ops["backward"]["+"] + in_c *
  // (in_h*in_w*batch_size) # bootstrap from previous total_ops["backward"]["*"]
  // = total_ops["backward"]["*"] + in_c # scale gradMean
  // total_ops["backward"]["*"] = total_ops["backward"]["*"] + in_c # eltwise
  // with dL / dGradVar total_ops["backward"]["+"] = in_c *
  // (in_h*in_w*batch_size) # sum gradXhat total_ops["backward"]["*"] =
  // total_ops["backward"]["*"] + in_c # scale gradXhat
  toReturn["ops:*"] += input_channels; // scale gradMean
  toReturn["ops:*"] += input_channels; // eltwise with dL / dGradVar
  toReturn["ops:+"] = input_volume;    // sum gradXhat
  toReturn["ops:*"] += input_channels; // scale gradXhat

  // # totalGradInput
  // total_ops["backward"]["+"] = total_ops["backward"]["+"] + in_c *
  // (in_h*in_w*batch_size) # Subtract mean, can't bootstrap this one
  // total_ops["backward"]["*"] = total_ops["backward"]["*"] + in_c # scale dL /
  // dMean total_ops["backward"]["*"] = total_ops["backward"]["*"] + in_c #
  // scale dL / dVar total_ops["backward"]["*"] = total_ops["backward"]["*"] +
  // in_c * (in_h*in_w*batch_size) # Eltwise multiply by dL / dVar
  // total_ops["backward"]["+"] = total_ops["backward"]["+"] + 2 * in_c *
  // (in_h*in_w*batch_size) # Accumulate gradient terms
  toReturn["ops:+"] += input_volume; // Subtract mean, can't bootstrap this one
  toReturn["ops:*"] += input_channels;   // scale dL / dMean
  toReturn["ops:*"] += input_channels;   // scale dL / dVar
  toReturn["ops:*"] += input_volume;     // Eltwise multiply by dL / dVar
  toReturn["OPS:+"] += 2 * input_volume; // Accumulate gradient terms

  uint64_t reads = 0;
  for (int i = 0; i < 7; i++) {
    auto v = getTensorVolume(getOperand(i).getType());
    toReturn["operand:" + std::to_string(i) + ":activation_in"] = v;
    reads += v;
  }

  uint64_t writes = 0;
  for (int i = 0; i < 3; i++) {
    auto v = getTensorVolume(getResult(i).getType());
    toReturn["result:" + std::to_string(i) + ":grad"] = v;
    writes += v;
  }

  toReturn["reads"] = reads;
  toReturn["writes"] = writes;

  return toReturn;
}

// std::map<std::string, uint64_t> ReLUUnderOp::getStatistics() {
//   return getReLUOpStatistics(*this);
// }
std::map<std::string, uint64_t> ReluUnderOp::getStatistics() {
  return getReLUOpStatistics(*this);
}

// sub
std::map<std::string, uint64_t> SubOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = getResult().getType().cast<TensorType>();
  TensorType aType = getOperand(0).getType().cast<TensorType>();
  Type bType = getOperand(1).getType();

  uint64_t ofm_volume = getTensorVolume(resultTy);

  toReturn["ops:-"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = getTensorVolume(aType);
  uint64_t b_volume = getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

// sub_
std::map<std::string, uint64_t> SubUnderOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = getResult().getType().cast<TensorType>();
  TensorType aType = getOperand(0).getType().cast<TensorType>();
  Type bType = getOperand(1).getType();

  uint64_t ofm_volume = getTensorVolume(resultTy);

  toReturn["ops:-"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = getTensorVolume(aType);
  uint64_t b_volume = getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

// sum
std::map<std::string, uint64_t> SumOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;
  TensorType ty = getOperand(0).getType().cast<TensorType>();
  uint64_t volume = getTensorVolume(ty);

  toReturn["ops:+"] = volume;

  toReturn["operand:0:activation_in"] = volume;
  toReturn["result:0:activation_out"] = volume;

  toReturn["reads"] = volume;
  toReturn["writes"] = volume;

  return toReturn;
}

// size op can be zero overhead
std::map<std::string, uint64_t> SizeOp::getStatistics() {
  std::map<std::string, uint64_t> toReturn;
  toReturn["reads"] = toReturn["operand:0:activation_in"] = 0;
  toReturn["writes"] = toReturn["result:0:activation_out"] = 0;
  return toReturn;
}

// squeeze can be zero overhead
std::map<std::string, uint64_t> SqueezeOp::getStatistics() {
  std::map<std::string, uint64_t> toReturn;
  toReturn["reads"] = toReturn["operand:0:activation_in"] = 0;
  toReturn["writes"] = toReturn["result:0:activation_out"] = 0;
  return toReturn;
}

// transpose can be zero overhead
std::map<std::string, uint64_t> TOp::getStatistics() {
  std::map<std::string, uint64_t> toReturn;
  toReturn["reads"] = toReturn["operand:0:activation_in"] = 0;
  toReturn["writes"] = toReturn["result:0:activation_out"] = 0;
  return toReturn;
}

// threshold_backward
std::map<std::string, uint64_t> ThresholdBackwardOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;
  uint64_t loss_in_volume =
      getTensorVolume(getOperand(0).getType().cast<TensorType>());
  uint64_t act_in_volume =
      getTensorVolume(getOperand(1).getType().cast<TensorType>());
  uint64_t loss_out_volume =
      getTensorVolume(getResult().getType().cast<TensorType>());

  toReturn["reads"] = toReturn["operand:0:activation_in"] =
      loss_in_volume + act_in_volume;
  toReturn["writes"] = toReturn["result:0:grad:dx"] = loss_out_volume;

  return toReturn;
}

// unsqueeze can be zero overhead
std::map<std::string, uint64_t> UnsqueezeOp::getStatistics() {
  std::map<std::string, uint64_t> toReturn;
  toReturn["reads"] = toReturn["operand:0:activation_in"] = 0;
  toReturn["writes"] = toReturn["result:0:activation_out"] = 0;
  return toReturn;
}

// view can be zero overhead
std::map<std::string, uint64_t> ViewOp::getStatistics() {
  std::map<std::string, uint64_t> toReturn;
  toReturn["reads"] = toReturn["operand:0:activation_in"] = 0;
  toReturn["writes"] = toReturn["result:0:activation_out"] = 0;
  return toReturn;
}

} // namespace aten
} // namespace NPCOMP
} // namespace mlir
