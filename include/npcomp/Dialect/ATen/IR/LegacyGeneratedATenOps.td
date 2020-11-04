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
