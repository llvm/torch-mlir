// This file is (mostly) automatically generated.  Please do not edit.
//===- ATenOps.td ------------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_ATEN_IR_GENERATED_ATEN_OPS
#define NPCOMP_DIALECT_ATEN_IR_GENERATED_ATEN_OPS

def aten_AddOp: aten_Op<"add", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor)> {
  let arguments = (
  ins AnyTensor:$self,
      AnyTensor:$other,
      AnyScalar:$alpha
  );
  let summary = "aten add operator";
  let description = [{
    AddOp
    aten add operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_AddUnderOp: aten_Op<"add_", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor)> {
  let arguments = (
  ins AnyTensor:$self,
      AnyTensor:$other,
      AnyScalar:$alpha
  );
  let summary = "aten add_ operator";
  let description = [{
    AddUnderOp
    aten add_ operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_AsStridedOp: aten_Op<"as_strided", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor)> {
  let arguments = (
  ins AnyTensor:$self,
      AnyType:$size,
      AnyType:$stride
  );
  let summary = "aten as_strided operator";
  let description = [{
    AsStridedOp
    aten as_strided operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_ConvolutionOverrideableOp: aten_Op<"convolution_overrideable", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor)> {
  let arguments = (
  ins AnyTensor:$input,
      AnyTensor:$weight,
      AnyTensor:$bias,
      AnyType:$stride,
      AnyType:$padding,
      AnyType:$dilation,
      AnyScalar:$transposed,
      AnyType:$output_padding,
      AnyScalar:$groups
  );
  let summary = "aten convolution_overrideable operator";
  let description = [{
    ConvolutionOverrideableOp
    aten convolution_overrideable operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_ConvolutionBackwardOverrideableOp: aten_Op<"convolution_backward_overrideable", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor, AnyTensor, AnyTensor)> {
  let arguments = (
  ins AnyTensor:$grad_output,
      AnyTensor:$input,
      AnyTensor:$weight,
      AnyType:$stride,
      AnyType:$padding,
      AnyType:$dilation,
      AnyScalar:$transposed,
      AnyType:$output_padding,
      AnyScalar:$groups
  );
  let summary = "aten convolution_backward_overrideable operator";
  let description = [{
    ConvolutionBackwardOverrideableOp
    aten convolution_backward_overrideable operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_DivOp: aten_Op<"div", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor)> {
  let arguments = (
  ins AnyTensor:$self,
      AnyTensor:$other
  );
  let summary = "aten div operator";
  let description = [{
    DivOp
    aten div operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_DivUnderOp: aten_Op<"div_", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor)> {
  let arguments = (
  ins AnyTensor:$self,
      AnyTensor:$other
  );
  let summary = "aten div_ operator";
  let description = [{
    DivUnderOp
    aten div_ operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_ExpandOp: aten_Op<"expand", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor)> {
  let arguments = (
  ins AnyTensor:$self,
      AnyType:$size,
      AnyScalar:$implicit
  );
  let summary = "aten expand operator";
  let description = [{
    ExpandOp
    aten expand operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_LogSoftmaxOp: aten_Op<"_log_softmax", [NoSideEffect]>,
    Results<(outs AnyTensor)> {
  let arguments = (
  ins AnyTensor:$self,
      AnyScalar:$dim,
      AnyScalar:$half_to_float
  );
  let summary = "aten _log_softmax operator";
  let description = [{
    LogSoftmaxOp
    aten _log_softmax operator
  }];
}

def aten_LogSoftmaxBackwardDataOp: aten_Op<"_log_softmax_backward_data", [NoSideEffect]>,
    Results<(outs AnyTensor)> {
  let arguments = (
  ins AnyTensor:$grad_output,
      AnyTensor:$output,
      AnyScalar:$dim,
      AnyTensor:$self
  );
  let summary = "aten _log_softmax_backward_data operator";
  let description = [{
    LogSoftmaxBackwardDataOp
    aten _log_softmax_backward_data operator
  }];
}

def aten_MeanOp: aten_Op<"mean", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor)> {
  let arguments = (
  ins AnyTensor:$self
  );
  let summary = "aten mean operator";
  let description = [{
    MeanOp
    aten mean operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_MmOp: aten_Op<"mm", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor)> {
  let arguments = (
  ins AnyTensor:$self,
      AnyTensor:$mat2
  );
  let summary = "aten mm operator";
  let description = [{
    MmOp
    aten mm operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_MulOp: aten_Op<"mul", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor)> {
  let arguments = (
  ins AnyTensor:$self,
      AnyTensor:$other
  );
  let summary = "aten mul operator";
  let description = [{
    MulOp
    aten mul operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_MulUnderOp: aten_Op<"mul_", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor)> {
  let arguments = (
  ins AnyTensor:$self,
      AnyTensor:$other
  );
  let summary = "aten mul_ operator";
  let description = [{
    MulUnderOp
    aten mul_ operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_NativeBatchNormOp: aten_Op<"native_batch_norm", [NoSideEffect, StatisticsOpInterface]>,
	 // FIXME: not quite automatically generated names
    Results<(outs AnyTensor:$output, AnyTensor:$save_mean, AnyTensor:$save_invstd)> {
  let arguments = (
  ins AnyTensor:$input,
      AnyTensor:$weight,
      AnyTensor:$bias,
      AnyTensor:$running_mean,
      AnyTensor:$running_var,
      AnyScalar:$training,
      AnyScalar:$momentum,
      AnyScalar:$eps
  );
  let summary = "aten native_batch_norm operator";
  let description = [{
    NativeBatchNormOp
    aten native_batch_norm operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_NativeBatchNormBackwardOp: aten_Op<"native_batch_norm_backward", [NoSideEffect, StatisticsOpInterface]>,
	 // FIXME: not quite automatically generated
	 Results<(outs AnyTensor:$dx, AnyTensor:$dm, AnyTensor:$dv)> {
let arguments = (
  ins AnyTensor:$grad_out,
      AnyTensor:$input,
      AnyTensor:$weight,
      AnyTensor:$running_mean,
      AnyTensor:$running_var,
      AnyTensor:$save_mean,
      AnyTensor:$save_invstd,
      AnyScalar:$train,
      AnyScalar:$eps
  );
  let summary = "aten native_batch_norm_backward operator";
  let description = [{
    NativeBatchNormBackwardOp
    aten native_batch_norm_backward operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_NegOp: aten_Op<"neg", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor)> {
  let arguments = (
  ins AnyTensor:$self
  );
  let summary = "aten neg operator";
  let description = [{
    NegOp
    aten neg operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_ReluOp: aten_Op<"relu", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor)> {
  let arguments = (
  ins AnyTensor:$self
  );
  let summary = "aten relu operator";
  let description = [{
    ReluOp
    aten relu operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_ReluUnderOp: aten_Op<"relu_", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor)> {
  let arguments = (
  ins AnyTensor:$self
  );
  let summary = "aten relu_ operator";
  let description = [{
    ReluUnderOp
    aten relu_ operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_SizeOp: aten_Op<"size", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor)> {
  let arguments = (
  ins AnyTensor:$self,
      AnyScalar:$dim
  );
  let summary = "aten size operator";
  let description = [{
    SizeOp
    aten size operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_SqueezeOp: aten_Op<"squeeze", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor)> {
  let arguments = (
  ins AnyTensor:$self,
      AnyScalar:$dim
  );
  let summary = "aten squeeze operator";
  let description = [{
    SqueezeOp
    aten squeeze operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_SumOp: aten_Op<"sum", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor)> {
  let arguments = (
  ins AnyTensor:$self,
      AnyType:$dim,
      AnyScalar:$keepdim
  );
  let summary = "aten sum operator";
  let description = [{
    SumOp
    aten sum operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_TOp: aten_Op<"t", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor)> {
  let arguments = (
  ins AnyTensor:$self
  );
  let summary = "aten t operator";
  let description = [{
    TOp
    aten t operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_ThresholdBackwardOp: aten_Op<"threshold_backward", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor)> {
  let arguments = (
  ins AnyTensor:$grad_output,
      AnyTensor:$self,
      AnyScalar:$threshold
  );
  let summary = "aten threshold_backward operator";
  let description = [{
    ThresholdBackwardOp
    aten threshold_backward operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_UnsqueezeOp: aten_Op<"unsqueeze", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor)> {
  let arguments = (
  ins AnyTensor:$self,
      AnyScalar:$dim
  );
  let summary = "aten unsqueeze operator";
  let description = [{
    UnsqueezeOp
    aten unsqueeze operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_SubOp: aten_Op<"sub", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor)> {
  let arguments = (
  ins AnyTensor:$self,
      AnyTensor:$other,
      AnyScalar:$alpha
  );
  let summary = "aten sub operator";
  let description = [{
    SubOp
    aten sub operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_SubUnderOp: aten_Op<"sub_", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor)> {
  let arguments = (
  ins AnyTensor:$self,
      AnyTensor:$other,
      AnyScalar:$alpha
  );
  let summary = "aten sub_ operator";
  let description = [{
    SubUnderOp
    aten sub_ operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_AddmmOp: aten_Op<"addmm", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor)> {
  let arguments = (
  ins AnyTensor:$self,
      AnyTensor:$mat1,
      AnyTensor:$mat2,
      AnyScalar:$beta,
      AnyScalar:$alpha
  );
  let summary = "aten addmm operator";
  let description = [{
    AddmmOp
    aten addmm operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_ViewOp: aten_Op<"view", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor)> {
  let arguments = (
  ins AnyTensor:$self,
      AnyType:$size
  );
  let summary = "aten view operator";
  let description = [{
    ViewOp
    aten view operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_GatherOp: aten_Op<"gather", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor)> {
  let arguments = (
  ins AnyTensor:$self,
      AnyScalar:$dim,
      AnyTensor:$index,
      AnyScalar:$sparse_grad
  );
  let summary = "aten gather operator";
  let description = [{
    GatherOp
    aten gather operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_NllLossForwardOp: aten_Op<"nll_loss_forward", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor, AnyTensor)> {
  let arguments = (
  ins AnyTensor:$self,
      AnyTensor:$target,
      AnyTensor:$weight,
      AnyScalar:$reduction,
      AnyScalar:$ignore_index
  );
  let summary = "aten nll_loss_forward operator";
  let description = [{
    NllLossForwardOp
    aten nll_loss_forward operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_NllLossBackwardOp: aten_Op<"nll_loss_backward", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor)> {
  let arguments = (
  ins AnyTensor:$grad_output,
      AnyTensor:$self,
      AnyTensor:$target,
      AnyTensor:$weight,
      AnyScalar:$reduction,
      AnyScalar:$ignore_index,
      AnyTensor:$total_weight
  );
  let summary = "aten nll_loss_backward operator";
  let description = [{
    NllLossBackwardOp
    aten nll_loss_backward operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_NllLoss2dForwardOp: aten_Op<"nll_loss2d_forward", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor, AnyTensor)> {
  let arguments = (
  ins AnyTensor:$self,
      AnyTensor:$target,
      AnyTensor:$weight,
      AnyScalar:$reduction,
      AnyScalar:$ignore_index
  );
  let summary = "aten nll_loss2d_forward operator";
  let description = [{
    NllLoss2dForwardOp
    aten nll_loss2d_forward operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_NllLoss2dBackwardOp: aten_Op<"nll_loss2d_backward", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor)> {
  let arguments = (
  ins AnyTensor:$grad_output,
      AnyTensor:$self,
      AnyTensor:$target,
      AnyTensor:$weight,
      AnyScalar:$reduction,
      AnyScalar:$ignore_index,
      AnyTensor:$total_weight
  );
  let summary = "aten nll_loss2d_backward operator";
  let description = [{
    NllLoss2dBackwardOp
    aten nll_loss2d_backward operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_HardtanhOp: aten_Op<"hardtanh", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor)> {
  let arguments = (
  ins AnyTensor:$self,
      AnyScalar:$min_val,
      AnyScalar:$max_val
  );
  let summary = "aten hardtanh operator";
  let description = [{
    HardtanhOp
    aten hardtanh operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_HardtanhBackwardOp: aten_Op<"hardtanh_backward", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor)> {
  let arguments = (
  ins AnyTensor:$grad_output,
      AnyTensor:$self,
      AnyScalar:$min_val,
      AnyScalar:$max_val
  );
  let summary = "aten hardtanh_backward operator";
  let description = [{
    HardtanhBackwardOp
    aten hardtanh_backward operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_HardtanhUnderOp: aten_Op<"hardtanh_", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor)> {
  let arguments = (
  ins AnyTensor:$self,
      AnyScalar:$min_val,
      AnyScalar:$max_val
  );
  let summary = "aten hardtanh_ operator";
  let description = [{
    HardtanhUnderOp
    aten hardtanh_ operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_AdaptiveAvgPool2dOp: aten_Op<"_adaptive_avg_pool2d", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor)> {
  let arguments = (
  ins AnyTensor:$self,
      AnyType:$output_size
  );
  let summary = "aten _adaptive_avg_pool2d operator";
  let description = [{
    AdaptiveAvgPool2dOp
    aten _adaptive_avg_pool2d operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_AdaptiveAvgPool2dBackwardOp: aten_Op<"_adaptive_avg_pool2d_backward", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor)> {
  let arguments = (
  ins AnyTensor:$grad_output,
      AnyTensor:$self
  );
  let summary = "aten _adaptive_avg_pool2d_backward operator";
  let description = [{
    AdaptiveAvgPool2dBackwardOp
    aten _adaptive_avg_pool2d_backward operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_MaxPool2dWithIndicesOp: aten_Op<"max_pool2d_with_indices", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor, AnyTensor)> {
  let arguments = (
  ins AnyTensor:$self,
      AnyType:$kernel_size,
      AnyType:$stride,
      AnyType:$padding,
      AnyType:$dilation,
      AnyScalar:$ceil_mode
  );
  let summary = "aten max_pool2d_with_indices operator";
  let description = [{
    MaxPool2dWithIndicesOp
    aten max_pool2d_with_indices operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_MaxPool2dWithIndicesBackwardOp: aten_Op<"max_pool2d_with_indices_backward", [NoSideEffect, StatisticsOpInterface]>,
    Results<(outs AnyTensor)> {
  let arguments = (
  ins AnyTensor:$grad_output,
      AnyTensor:$self,
      AnyType:$kernel_size,
      AnyType:$stride,
      AnyType:$padding,
      AnyType:$dilation,
      AnyScalar:$ceil_mode,
      AnyTensor:$indices
  );
  let summary = "aten max_pool2d_with_indices_backward operator";
  let description = [{
    MaxPool2dWithIndicesBackwardOp
    aten max_pool2d_with_indices_backward operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

#endif // NPCOMP_DIALECT_ATEN_IR_GENERATED_ATEN_OPS
