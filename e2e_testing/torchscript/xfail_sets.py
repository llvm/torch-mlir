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

from torch_mlir_e2e_test.test_suite import COMMON_TORCH_MLIR_LOWERING_XFAILS

REFBACKEND_XFAIL_SET = COMMON_TORCH_MLIR_LOWERING_XFAILS

EAGER_MODE_XFAIL_SET = {
    # RefBackend fails
    "TableBatchEmbeddingModule_basic",
    "QuantizedMLP_basic"
}

# Write the TOSA set as a "passing" set as it is very early in development
# and very few tests work yet.
TOSA_PASS_SET = {
    "ElementwiseUnaryModule_basic",
    "ElementwiseBinaryModule_basic",
    "ElementwiseSigmoidModule_basic",
    "ElementwiseExpModule_basic",
    "ElementwiseReluModule_basic",
    "ElementwiseFloorModule_basic",
    "ElementwiseLogModule_basic",
    "ElementwiseBinaryStaticShapeModule_basic",
    "ElementwiseMinimumModule_basic",
    "ElementwiseMinimumIntModule_basic",
    "ElementwiseMaximumModule_basic",
    "ElementwiseMaximumIntModule_basic",
    "TanhBackward_basic",
    "ElementwiseAddModule_basic",
    "ReturnThreeTensorFloat32_basic",
    "AddCMulModule_basic",
    "AddCDivModule_basic",
    "SqueezeModule_broadcast",
    "BoolTensorReturnFalseModule_basic",
    "BoolTensorReturnTrueModule_basic",
    "BoolTensorReturnMixedModule_basic",
    "BoolTensorHandleSignless_basic",
    "ElementwiseRsqrtModule_basic",
    "SqueezeModule_static",
    "SqueezeModule_noUnitDim",
    "SqueezeModule_allUnitDim",
    "TModuleRank1_basic",
    "TModuleRank0_basic",
    "ElementwiseToDtypeIdentityModule_basic",
    "View1DFoldModule_basic",
    "UnsafeView1DFoldModule_basic",
    "SqueezeDimModule_static",
    "SqueezeDimModule_identity",
    "SqueezeDimModule_unitDim",
    "ReturnTwoTensorF32I64_basic",
    "ElementwisePowModule_basic",
    "BmmModule_basic",
    "MmDagModule_basic",
    "Matmul4dStatic_basic",
    "Matmul_dot",
    "Matmul_3d",
    "RsubFloatModule_basic",
    "RsubFloatModule_noalpha_basic",
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
    "FlattenStaticModule_basic",
    "FlattenRank0Module_basic",
    "ElementwiseFlattenBroadcastModule_basic",
    "SquareModule_basic",
    "MaxPool2dStaticModule_basic",
    "ResNet18StaticModule_basic",
    "NativeLayerNormModule4D_basic",
    "LayerNormNormalizeOverAllDimsModule_basic",
    "PermuteModule_basic",
    "PermuteNegativeIndexModule_basic",
    "ElementwiseLog2Module_basic",
    "Threshold1dIntI32Module_basic",
    "Threshold1dFloatModule_basic",
    "Threshold2dFloatModule_basic",
    "Threshold3dFloatModule_basic",
    "ElementwiseSubScalarIntModule_basic",
    "ElementwiseAddScalarIntModule_basic",
    "ElementwiseMulScalarModule_basic",
    "ZerosModuleDefaultDtype_basic",
    "ZerosModuleInt2D_basic",
    "ZerosModuleInt3D_basic",
    "ZerosModuleFloat2D_basic",
    "ZerosModuleFloat3D_basic",
    "ZerosModuleFalsePinMemory_basic",
    "OnesModuleDefaultDtype_basic",
    "OnesModuleInt_basic",
    "OnesModuleFloat_basic",
    "OnesModuleFalsePinMemory_basic",
    "NewZerosModuleDefaultDtype_basic",
    "NewZerosModuleInt2D_basic",
    "NewZerosModuleInt3D_basic",
    "NewZerosModuleFloat2D_basic",
    "NewZerosModuleFloat3D_basic",
    "NewZerosModuleFalsePinMemory_basic",
    "NewOnesModuleDefaultDtype_basic",
    "NewOnesModuleInt2D_basic",
    "NewOnesModuleInt3D_basic",
    "NewOnesModuleFloat2D_basic",
    "NewOnesModuleFloat3D_basic",
    "NewOnesModuleFalsePinMemory_basic",
    "SiluModule_basic",
    "DropoutEvalIntModule_basic",
    "DropoutEvalFloatModule_basic",
    "ContiguousModule_basic",
    "DropoutModule_basic",
    "ViewExpandModule_basic",
    "ViewExpandOnesModule_basic",
    "ViewCollapseInferredDimModule_basic",
    "ViewExpandInferredDimModule_basic",
    "ViewNoChangeStaticModule_basic",
    "UnsafeViewExpandModule_basic",
    "ReshapeCollapseModule_basic",
    "ElementwiseGeluModule_basic",
    "GeluBackwardModule_basic",
    "ElementwiseNeIntScalarModule_basic",
    "ElementwiseNeFloatTensorModule_basic",
    "ConvolutionModule2DStatic_basic",
    "ElementwiseNegModule_basic",
    "TestMultipleTensorReturn_basic",
    "AdaptiveAvgPool2dUnitOutputSizeStaticModule_basic",
    "BaddbmmDynamicModule_basic",
    "BaddbmmStaticModule_basic",
    "BaddbmmWithAlphaBetaModule_basic",
    "BaddbmmWithAlphaModule_basic",
    "BaddbmmWithBetaModule_basic",
    "BaddbmmBroadcast1DInputModule_basic",
    "BaddbmmBroadcast2DInputModule_basic",
    "NumpyTRank1Module_basic",
    "NumpyTRank2Module_basic",
    "NumpyTRankNStaticModule_basic",
    "NumpyTRankNDynamicModule_basic",
    "EmbeddingModuleI32Static_basic",
}
