# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# This file describes the sets of tests expected to fail for each config.
# This information is deliberately kept in a side table, rather than
# in-situ on the test, as a deliberate layering decision: tests should
# have unique keys to identify them and enable side tables of various kinds
# (this includes down into lower parts of the stack, where a side table
# might be used to keep more elaborate sets of testing configurations).

# Lists of tests that fail to even reach the backends.
# These represent further work needed in torch-mlir to lower them properly
# to the backend contract.
COMMON_TORCH_MLIR_LOWERING_XFAILS = {
    "QuantizedMLP_basic",
    "IouOfModule_basic",
}
REFBACKEND_XFAIL_SET = COMMON_TORCH_MLIR_LOWERING_XFAILS

# Write the TOSA set as a "passing" set as it is very early in development
# and very few tests work yet.
TOSA_PASS_SET = {
    "ElementwiseUnaryModule_basic",
    "ElementwiseBinaryModule_basic",
    "ElementwiseSigmoidModule_basic",
    "ElementwiseReluModule_basic",
    "ElementwiseFloorModule_basic",
    "ElementwiseLogModule_basic",
    "ElementwiseBinaryStaticShapeModule_basic",
    "TanhBackward_basic",
    "ElementwiseAddModule_basic",
    "ReturnThreeTensorFloat32_basic",
    "AddCMulModule_basic",
    "AddCDivModule_basic",
    "SqueezeModule_broadcast",
    "BoolTensorReturnFalseModule_basic",
    "BoolTensorReturnTrueModule_basic",
    "BoolTensorReturnMixedModule_basic",
    "ElementwiseRsqrtModule_basic",
    "SqueezeModule_static",
    "SqueezeModule_noUnitDim",
    "SqueezeModule_allUnitDim",
    "TModuleRank1_basic",
    "TModuleRank0_basic",
    "ElementwiseToDtypeIdentityModule_basic",
    "View1DFoldModule_basic",
    "SqueezeDimModule_static",
    "SqueezeDimModule_identity",
    "SqueezeDimModule_unitDim",
    "ReturnTwoTensorF32I64_basic",
    "ElementwisePowModule_basic",
    "BmmModule_basic",
    "Matmul_dot",
    "Matmul_3d",
    "RsubModule_basic",
    "RsubModule_noalpha_basic",
    "ElementwiseGtFloatScalarModule_basic",
    "ElementwiseGtIntScalarModule_basic",
    "ElementwiseGtMixed2ScalarModule_basic",
    "ElementwiseGtFloatTensorModule_basic",
    "ElementwiseGtIntTensorModule_basic",
    "ElementwiseLtFloatScalarModule_basic",
    "ElementwiseLtIntScalarModule_basic",
    "ElementwiseLtDiffWidthScalarModule_basic",
    "ElementwiseLtFloatTensorModule_basic",
    "ElementwiseLtIntTensorModule_basic",
    "ElementwiseEqFloatScalarModule_basic",
    "ElementwiseEqIntScalarModule_basic",
    "ElementwiseEqDiffWidthScalarModule_basic",
    "ElementwiseEqFloatTensorModule_basic",
    "ElementwiseEqIntTensorModule_basic",
    "ElementwiseMulScalarModule_int",
    "ElementwiseMulScalarModule_float",
    "ElementwiseMulTensorIntModule_basic",
    "ElementwiseDivScalarModule_basic",
    "ElementwiseSubScalarFloatModule_basic",
    "ElementwiseAddScalarFloatModule_basic",
    "ElementwiseMulScalarModule_float",
    "ElementwiseCeilModule_basic",
    "ElementwiseReciprocalModule_basic",
    "TypePromotionAlphaWiderModule_basic",
    "Conv2dWithPaddingDilationStrideStaticModule_basic",
    "BatchNorm1DModule_basic",
    "BatchNorm2DModule_basic",
    "BatchNorm3DModule_basic",
}
