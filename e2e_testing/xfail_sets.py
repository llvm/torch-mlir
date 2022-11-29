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

EAGER_MODE_XFAIL_SET = COMMON_TORCH_MLIR_LOWERING_XFAILS | {
    # RefBackend fails for some reason.
    # These tests pass in the regular RefBackend flow, so it's unclear
    # why they fail here.
    "Matmul_vecmat",
    "UpSampleNearest2dDynamicFactor_basic",
}

TORCHDYNAMO_XFAIL_SET = {
    #### General TorchDynamo/PyTorch errors

    # https://github.com/pytorch/torchdynamo/issues/1891
    # AssertionError: torch.* op returned non-Tensor bool call_function aten.Bool
    "AllBoolFalseModule_basic",
    "AllBoolTrueModule_basic",
    "AnyBoolFalseModule_basic",
    "AnyBoolTrueModule_basic",
    "BoolFloatFalseModule_basic",
    "BoolFloatTrueModule_basic",
    "BoolFloatConstantModule_basic",
    "BoolIntFalseModule_basic",
    "BoolIntTrueModule_basic",
    "BoolIntConstantModule_basic",
    "CeilFloatModule_basic",
    "ContainsIntList_False",
    "ContainsIntList_True",
    "GeIntModule_basic",
    "LenStrModule_basic",
    "SqrtIntConstantModule_basic",
    "SqrtIntModule_basic",
    "NumelModule_basic",
    "NumelZeroRankModule_basic",

    # RecursionError: maximum recursion depth exceeded
    # RuntimeError: Failed running call_function aten.lift_fresh_copy(...
    # https://github.com/pytorch/pytorch/issues/89627
    "LiftFreshCopyModule_basic",
    # TypeError: new_empty(): argument 'size' (position 1) must be tuple of ints, but found element of type NoneType at pos 0
    # RuntimeError: Failed running call_function aten.convolution_backward(...
    # https://github.com/pytorch/pytorch/issues/89629
    "ConvolutionBackwardModule2DPadded_basic",
    "ConvolutionBackwardModule2D_basic",
    # RuntimeError: Index tensor must have the same number of dimensions as self tensor
    # RuntimeError: Failed running call_function aten.nll_loss_backward(...
    # https://github.com/pytorch/pytorch/issues/89630
    "NllLossModuleBackward1DMeanWeight_basic",
    "NllLossModuleBackward1DMean_basic",
    "NllLossModuleBackward1DSumWeight_basic",
    "NllLossModuleBackward1DSum_basic",
    "NllLossModuleBackward1DWeight_basic",
    "NllLossModuleBackward1D_basic",
    # Decomposition assertion:
    # assert device is not None or dtype is not None or memory_format is not None
    # https://github.com/pytorch/pytorch/issues/89633
    "ToCopyModule_basic",
    # TypeError: expected np.ndarray (got float)
    # TODO: This is due to returning a scalar float as output from the test.
    # We should probably just standardize all tests to return tensors.
    "DivIntModule_basic",

    #### Torch-MLIR internal compiler errors

    # These are probably due to slightly different ops being recorded by
    # torchdynamo vs. torchscript.

    # error: unsupported by backend contract: tensor with unknown rank
    "UpSampleNearest2dDynamicFactor_basic",
    "AdaptiveAvgPool2dNonUnitOutputSizeDynamicModule_basic",
    "AdaptiveAvgPool2dNonUnitOutputSizeStaticModule_basic",
    "AtenEmbeddingBagSumExample_basic",
    "BernoulliModule_basic",
    "DropoutTrainModule_basic",
    "ElementwiseWhereScalarModule_basic",
    "ElementwiseWhereScalarOtherModule_basic",
    "ElementwiseWhereScalarSelfModule_basic",
    "IndexPutImpl1DFloatAccumulateModule_basic",
    "IndexPutImpl1DFloatNonAccumulateModule_basic",
    "IndexPutImpl1DIntAccumulateModule_basic",
    "IndexPutImpl1DIntNonAccumulateModule_basic",
    "IndexPutImpl2DFloatAccumulateModule_basic",
    "IndexPutImpl2DFloatNonAccumulateModule_basic",
    "IndexPutImpl3DFloatAccumulateModule_basic",
    "IndexPutImpl3DFloatNonAccumulateModule_basic",
    "Matmul_dot",
    "Matmul_vecmat",
    "StdBiasedModule_basic",
    "StdDimBiasedModule_basic",
    "StdDimEmptyDimModule_basic",
    "StdDimKeepDimFalseModule_basic",
    "StdDimKeepDimTrueModule_basic",
    "StdDimNoneDimModule_basic",
    "StdUnbiasedModule_basic",
    "UniformModule_basic",
    "UniformStaticModule_basic",
    # https://github.com/llvm/torch-mlir/issues/1611
    # error: 'tensor.cast' op operand type 'tensor<0xi64>' and result type 'tensor<18xi64>' are cast incompatible
    "Aten_EmbeddingBagExample_basic",
    # error: failed to legalize operation 'torch.valsem.aten.bernoulli.float' that was explicitly marked illegal
    "BernoulliFloatModule_basic",
    # error: failed to legalize operation 'torch.aten.bernoulli.Tensor' that was explicitly marked illegal
    "BernoulliTensorModule_basic",
    # error: failed to legalize operation 'torch.aten.view' that was explicitly marked illegal
    "ElementwiseFlattenBroadcastModule_basic",
    "FlattenRank0Module_basic",
    # error: failed to materialize conversion for result #0 of operation 'torch.aten.t' that remained live after conversion
    "TModuleRank1_basic",
}

MHLO_PASS_SET = {
    "ArangeDtypeFloatModule_basic",
    "ArangeDtypeIntModule_basic",
    "ArangeFalsePinMemoryModule_basic",
    "ArangeFloatModule_basic",
    "ArangeIntModule_basic",
    "ArangeNegativeStartFloatModule_basic",
    "ArangeNegativeStartIntModule_basic",
    "ArangeStartFloatModule_basic",
    "ArangeStartIntModule_basic",
    "ArangeStartNegativeStepFloatModule_basic",
    "ArangeStartNegativeStepIntModule_basic",
    "ArangeStartStepFloatModule_basic",
    "ArangeStartStepIntModule_basic",
    "ArangeZeroElementOutputModule_basic",
    "BmmModule_basic",
    "BroadcastToModule_basic",
    "BroadcastToSameRankStaticModule_basic",
    "BroadcastZeroRankInputStaticModule_basic",
    "ElementwiseAtenWhereSelfModule_basic",
    "ElementwiseClampModule_basic",
    "ElementwiseClampMinModule_basic",
    "ElementwiseClampMaxModule_basic",
    "ElementwiseExpModule_basic",
    "ElementwiseLogModule_basic",
    "ElementwiseNegModule_basic",
    "ElementwiseSqrtModule_basic",
    "ElementwiseUnaryModule_basic",
    "ElementwiseUnsqueezeNegDimsModule_basic",
    "ElementwiseToDtypeF32ToI64Module_basic",
    "ElementwiseAddModule_basic",
    "ElementwiseAddScalarFloatModule_basic",
    "ElementwiseAddScalarInt64Module_basic",
    "ElementwiseAddScalarIntModule_basic",
    "ElementwiseDivScalarModule_basic",
    "ElementwiseEqDiffWidthScalarModule_basic",
    "ElementwiseEqFloatScalarModule_basic",
    "ElementwiseEqIntScalarModule_basic",
    "ElementwiseErfModule_basic",
    "ElementwiseGeluModule_basic",
    "ElementwiseGtFloatScalarModule_basic",
    "ElementwiseGtIntScalarModule_basic",
    "ElementwiseGtMixed2ScalarModule_basic",
    "ElementwiseGeFloatIntScalarModule_basic",
    "ElementwiseGeFloatScalarModule_basic",
    "ElementwiseGeIntScalarModule_basic",
    "ElementwiseGeMixedIntScalarModule_basic",
    "ElementwiseLeFloatIntScalarModule_basic",
    "ElementwiseLeFloatScalarModule_basic",
    "ElementwiseLeIntScalarModule_basic",
    "ElementwiseLeMixedIntScalarModule_basic",
    "ElementwiseLtDiffWidthScalarModule_basic",
    "ElementwiseLtFloatScalarModule_basic",
    "ElementwiseLtIntScalarModule_basic",
    "ElementwiseMulScalarModule_basic",
    "ElementwiseMulScalarModule_float",
    "ElementwiseMulScalarModule_int",
    "ElementwiseNeFloatTensorModule_basic",
    "ElementwiseNeIntScalarModule_basic",
    "ElementwiseReciprocalModule_basic",
    "ElementwiseRelu6Module_basic",
    "ElementwiseReluModule_basic",
    "ElementwiseRemainderScalarModule_Bool_basic",
    "ElementwiseRemainderScalarModule_Float_basic",
    "ElementwiseRemainderScalarModule_Int_Float_basic",
    "ElementwiseRemainderScalarModule_Int_basic",
    "ElementwiseSubScalarFloatModule_basic",
    "ElementwiseSubScalarIntModule_basic",
    "ElementwiseWhereScalarModule_basic",
    "EmbeddingModule1DIndices_basic",
    "EmbeddingModuleI32Static_basic",
    "EmbeddingModuleI32_basic",
    "EmbeddingModuleI64_basic",
    "ExpandAsIntModule_basic",
    "ExpandModule_basic",
    "FullLikeModuleDefaultDtype_basic",
    "FullLikeModuleFalsePinMemory_basic",
    "FullLikeModuleFloat2D_basic",
    "FullLikeModuleFloat3DStatic_basic",
    "FullLikeModuleFloat3D_basic",
    "FullLikeModuleInt2DStatic_basic",
    "FullLikeModuleInt2D_basic",
    "FullLikeModuleInt3D_basic",
    "FullModuleDefaultDtype_basic",
    "FullModuleFalsePinMemory_basic",
    "FullModuleFloat2D_basic",
    "FullModuleFloat3D_basic",
    "FullModuleInt2D_basic",
    "FullModuleInt3D_basic",
    "GatherStaticModule_basic",
    "GatherModule_basic",
    "Gather2DInputModdule_basic",
    "GatherRandomIndexModule_basic",
    "HardTanhIntModule_basic",
    "HardTanhModule_basic",
    "HardsigmoidModule_basic",
    "HardsigmoidRandomModule_basic",
    "IndexSelectDynamicIndexSizeModule_basic",
    "IndexSelectSingleIdxModule_basic",
    "IndexSelectTwoIdxModule_basic",
    "IndexSelectWholeDimensionModule_basic",
    "IndexSelectWholeTensorModule_basic",
    "MatmulBroadcastBatchDim_basic",
    "MatmulSingleDynamicBatchDim_basic",
    "Matmul_3d",
    "Matmul_4d",
    "MeanDimEmptyDimModule_basic",
    "MeanDtypeModule_basic",
    "MeanDynamicSizesModule_basic",
    "MeanLargeInputModule_basic",
    "MeanModule_basic",
    "MmTanhModule_basic",
    "Mv_basic",
    "PrimsConvertElementTypeModule_basic",
    "ReduceFrobeniusNormKeepDimModule_basic",
    "ReduceSumDimIntListElementTypeBoolModule_basic",
    "ReduceSumElementTypeBoolModule_basic",
    "ReduceSumDimIntListEmptyDimModule_basic",
    "ReduceSumDimIntListDtypeFloatModule_basic",
    "ReduceSumDimIntListDtypeIntModule_basic",
    "ReduceSumDimIntListKeepDimFloatModule_basic",
    "ReduceSumDimIntListKeepDimIntModule_basic",
    "ReduceSumDtypeFloatModule_basic",
    "ReduceSumDtypeIntModule_basic",
    "SelectIntModule_basic",
    "SliceSingleIdxModule_basic",
    "SqueezeDimModule_dynamic",
    "SqueezeDimModule_negDim",
    "ReduceFrobeniusNormModule_basic",
    "FlattenStaticModule_basic",
    "FlattenRank0Module_basic",
    "TensorsConcatNegativeDimModule_basic",
    "LiftFreshCopyModule_basic",
    "Mlp2LayerModuleNoBias_basic",
    "NumelModule_basic",
    "SqueezeModule_allUnitDim",
    "SqueezeDimModule_unitDim",
    "ViewCollapseOnesMiddleModule_basic",
    "ViewDoubleMergeStaticModule_basic",
    "ViewExpandDynamicDimModule_basic",
    "ViewFlattenAndExpandModule_basic",
    "ViewFiveTestStaticModule_basic",
    "ViewOffsetTestStaticModule_basic",
    "ViewTwoFiveThreeStaticModule_basic",
    "ViewTwoToThreeStaticModule_basic",
    "ViewExpandOnesMiddleOppModule_basic",
    "ViewOffsetBackwardTestStaticModule_basic",
    "NumToTensorFloatModule_basic",
    "AtenToDeviceModule_basic",
    "AvgPool2dStaticModule_basic",
    "Conv2dWithPaddingDilationStrideStaticModule_basic",
    "Convolution2DStaticModule_basic",
    "ConvolutionModule2DTransposeStridedStatic_basic",
    "ElementwiseCloneContiguousModule_basic",
    "ElementwiseCloneModule_basic",
    "ElementwiseBinaryStaticShapeModule_basic",
    "ReturnThreeTensorFloat32_basic",
    "BoolTensorReturnFalseModule_basic",
    "BoolTensorReturnTrueModule_basic",
    "BoolTensorReturnMixedModule_basic",
    "SqueezeModule_static",
    "TModuleRank1_basic",
    "TModuleRank0_basic",
    "ElementwiseToDtypeIdentityModule_basic",
    "View1DFoldModule_basic",
    "UnsafeView1DFoldModule_basic",
    "RsubFloatModule_basic",
    "RsubFloatModule_noalpha_basic",
    "RsubIntModule_basic",
    "RsubIntModule_noalpha_basic",
    "SliceStaticModule_basic",
    "SliceModule_basic",
    "SliceNegIdxModule_basic",
    "SliceOutOfLowerBoundStartIndexModule_basic",
    "SliceOutOfUpperBoundIndexModule_basic",
    "SliceStartEqEndModule_basic",
    "SliceSizeTwoStepModule_basic",
    "SliceWholeTensorModule_basic",
    "SqueezeDimModule_static",
    "SqueezeDimModule_identity",
    "SqueezeModule_broadcast",
    "ReturnTwoTensorF32I64_basic",
    "Matmul4dStatic_basic",
    "Matmul_dot",
    "Matmul_2d",
    "Matmul_matvec",
    "Matmul_vecmat",
    "MaxPool2dWithIndicesStaticModule_basic",
    "MmDagModule_basic",
    "MmModule_basic",
    "MmModule_chained",
    "MaxPool2dStaticModule_basic",
    "PermuteModule_basic",
    "PermuteNegativeIndexModule_basic",
    "ReduceSumDimIntListKeepDimNegativeDimStaticModule_basic",
    "ZerosLikeModule_defaultDtype",
    "ZerosLikeModule_falsePinMemory",
    "ZerosLikeModule_float",
    "ZerosLikeModule_int",
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
    "OnesLikeModule_defaultDtype",
    "OnesLikeModule_falsePinMemory",
    "OnesLikeModule_float",
    "OnesLikeModule_int",
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
    "DropoutEvalIntModule_basic",
    "DropoutEvalFloatModule_basic",
    "ContiguousModule_basic",
    "DropoutModule_basic",
    "ViewCollapseModule_basic",
    "ViewCollapseInferredDimModule_basic",
    "ViewDynamicExpandCollapseModule_basic",
    "ViewDynamicExpandModule_basic",
    "ViewExpandModule_basic",
    "ViewExpandOnesModule_basic",
    "ViewExpandOnesBeforeAndAfterModule_basic",
    "ViewExpandOnesMiddleModule_basic",
    "ViewExpandCollapseModule_basic",
    "ViewExpandCollapseWithOnesModule_basic",
    "ViewExpandInferredDimModule_basic",
    "ViewNoChangeStaticModule_basic",
    "ViewNoChange1dModule_basic",
    "ViewNoChange2dModule_basic",
    "ViewNoChange3dModule_basic",
    "UnsafeViewExpandModule_basic",
    "ReduceMaxAllDims_basic",
    "ReduceMaxFloatModule_basic",
    "ReduceMaxSignedIntModule_basic",
    "ReduceMaxUnsignedIntModule_basic",
    "ReduceSumDimIntListFloatModule_basic",
    "ReduceSumDimIntListIntModule_basic",
    "ReduceSumFloatModule_basic",
    "ReduceSumSignedIntModule_basic",
    "ReduceSumUnsignedIntModule_basic",
    "RepeatModule_basic",
    "ReshapeAliasCollapseModule_basic",
    "ReshapeAliasExpandModule_basic",
    "ReshapeExpandModule_basic",
    "RollModule_basic",
    "TestMultipleTensorReturn_basic",
    "AdaptiveAvgPool2dUnitOutputSizeStaticModule_basic",
    "BaddbmmStaticModule_basic",
    "BaddbmmBroadcast1DInputModule_basic",
    "BaddbmmBroadcast2DInputModule_basic",
    "NarrowHorizontalTest2_basic",
    "NarrowHorizontalTest_basic",
    "NarrowVerticalTest2_basic",
    "NarrowVerticalTest_basic",
    "NumToTensorIntModule_basic",
    "NumpyTRank0Module_basic",
    "NumpyTRank1Module_basic",
    "NumpyTRank2Module_basic",
    "NumpyTRankNStaticModule_basic",
    "NumpyTRankNDynamicModule_basic",
    "TModuleRank2_basic",
    "TensorLiteralModule_basic",
    "TensorsConcatModule_basic",
    "TensorOpaqueLiteralModule_basic",
    "TransposeIntModule_basic",
    "TransposeIntNegDimsModule_basic",
    "ToDtypeBoolLayoutNoneModule_basic",
    "ToDtypeBoolLayoutNoneStaticModule_basic",
    "ToDtypeLayoutNoneModule_basic",
    "ToDtypeLayoutStridedModule_basic",
    "TypeAsSameModule_basic",
    "TypeConversionF32ToF64Module_basic",
    "TypeConversionF64ToF32Module_basic",
    "TypeConversionI1ToF32Module_basic",
    "TypeConversionI1ToF64Module_basic",
    "TypeConversionI1ToI32Module_basic",
    "TypeConversionI1ToI64Module_basic",
    "TypeConversionI32ToI64Module_basic",
    "TypeConversionI64ToI32Module_basic",
    "TypePromotionAlphaWiderModule_basic",
    "TypePromotionSameCategoryZeroRankWider_basic",
    "TypePromotionZeroRankHigherCategoryModule_basic",
    "OnesModuleCPUDevice_basic",
    "Permute0RankModule_basic",
    "UnsafeViewCollapseModule_basic",
    "UnsafeViewDynamicExpandModule_basic",
    "AtenRoundIntModule_basic",
}

# Write the TOSA set as a "passing" set as it is very early in development
# and very few tests work yet.
TOSA_PASS_SET = {
    "ElementwiseCloneContiguousModule_basic",
    "ElementwiseCloneModule_basic",
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
    "ViewDoubleMergeStaticModule_basic",
    "ViewCollapseOnesMiddleModule_basic",
    "ViewFiveTestStaticModule_basic",
    "ViewOffsetTestStaticModule_basic",
    "ViewTwoFiveThreeStaticModule_basic",
    "ViewTwoToThreeStaticModule_basic",
    "ViewExpandOnesMiddleOppModule_basic",
    "ViewOffsetBackwardTestStaticModule_basic",
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
    "AtenToDeviceModule_basic",
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
    "ElementwiseNotIntegerModule_basic",
    "ElementwiseNotInt32Module_basic",
    "TypePromotionAlphaWiderModule_basic",
    "Conv2dWithPaddingDilationStrideStaticModule_basic",
    "BatchNorm1DModule_basic",
    "BatchNorm1DWith2DInputModule_basic",
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
    "OnesModuleCPUDevice_basic",
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
    "ViewExpandOnesBeforeAndAfterModule_basic",
    "ViewExpandOnesMiddleModule_basic",
    "ViewExpandCollapseModule_basic",
    "ViewExpandCollapseWithOnesModule_basic",
    "ViewCollapseInferredDimModule_basic",
    "ViewExpandInferredDimModule_basic",
    "ViewNoChangeStaticModule_basic",
    "UnsafeViewExpandModule_basic",
    "ReshapeCollapseModule_basic",
    "ElementwiseGeluModule_basic",
    "GeluBackwardModule_basic",
    "ElementwiseNeIntScalarModule_basic",
    "ElementwiseNeFloatTensorModule_basic",
    "Convolution2DStaticModule_basic",
    "ElementwiseNegModule_basic",
    "TestMultipleTensorReturn_basic",
    "TypeAsSameModule_basic",
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
    "TModuleRank2_basic",
    "TransposeIntModule_basic",
    "TransposeIntNegDimsModule_basic",
    "ArgmaxModule_keepDim",
    "ArgmaxModule_with_dim",
    "_LogSoftmaxModuleStable_basic",
    "ElementwiseAtenWhereSelfModule_basic",
    "ElementwiseUnsqueezeBroadcastModule_basic",
    "LiftFreshCopyModule_basic",
    "ReduceSumDimIntListKeepDimNegativeDimStaticModule_basic",
    "ReduceSumDimIntListFloatModule_basic",
    "ReduceSumDimIntListIntModule_basic",
    "ReduceSumDimIntListKeepDimFloatModule_basic",
    "ReduceSumDimIntListKeepDimIntModule_basic",
    "ReduceSumFloatModule_basic",
    "ReduceSumSignedIntModule_basic",
    "ReduceSumUnsignedIntModule_basic",
    "BroadcastToSameRankStaticModule_basic",
    "BroadcastZeroRankInputStaticModule_basic",
    "SliceStaticModule_basic",
    "ArangeStartStepIntModule_basic",
    "ArangeDtypeFloatModule_basic",
    "ArangeIntModule_basic",
    "ArangeNegativeStartIntModule_basic",
    "ArangeStartIntModule_basic",
    "ArangeStartNegativeStepIntModule_basic",
    "ArangeZeroElementOutputModule_basic",
    "NumToTensorIntModule_basic",
    "ToDtypeBoolLayoutNoneStaticModule_basic",
    "ToCopyBoolDTypeStaticModule_basic",
    "HardTanhIntModule_basic",
    "AtenRoundIntModule_basic",
    "MseLossNoReductionModule_basic",
    "AddCMul_Module_basic",
    "AddCDiv_Module_basic",
}

LTC_XFAIL_SET = {
    "_Convolution2DAllFalseModule_basic",
    "_Convolution2DBenchmarkModule_basic",
    "_Convolution2DCudnnModule_basic",
    "_Convolution2DDeterministicModule_basic",
    "_Convolution2DTF32Module_basic",
    "_ConvolutionDeprecated2DAllFalseModule_basic",
    "_ConvolutionDeprecated2DBenchmarkModule_basic",
    "_ConvolutionDeprecated2DCudnnModule_basic",
    "_ConvolutionDeprecated2DDeterministicModule_basic",
    "AdaptiveAvgPool2dNonUnitOutputSizeDynamicModule_basic",
    "AdaptiveAvgPool2dNonUnitOutputSizeStaticModule_basic",
    "AddIntModule_basic",
    "BernoulliFloatModule_basic",
    "BernoulliModule_basic",
    "BernoulliTensorModule_basic",
    "BincountMinlengthModule_basic",
    "BincountModule_basic",
    "BincountStaticSizeModule_basic",
    "BoolFloatFalseModule_basic",
    "BoolFloatTrueModule_basic",
    "BoolIntFalseModule_basic",
    "BoolIntTrueModule_basic",
    "CeilFloatModule_basic",
    "DivFloatModule_basic",
    "DropoutTrainModule_basic",
    "ElementwiseAtenFloorDivideBroadcastModule_basic",
    "ElementwiseAtenFloorDivideModule_basic",
    "EqIntModule_basic",
    "GeFloatIntModule_basic",
    "GeFloatModule_basic",
    "GeIntModule_basic",
    "GtFloatIntModule_basic",
    "GtIntModule_basic",
    "HBC_basic",
    "IndexPut1DFloatAccumulateModule_basic",
    "IndexPut1DFloatNonAccumulateModule_basic",
    "IndexPut1DIntAccumulateModule_basic",
    "IndexPut1DIntNonAccumulateModule_basic",
    "IndexPut2DFloatAccumulateModule_basic",
    "IndexPut2DFloatNonAccumulateModule_basic",
    "IndexPut2DIntAccumulateModule_basic",
    "IndexPut2DIntNonAccumulateModule_basic",
    "IndexPut3DFloatAccumulateModule_basic",
    "IndexPut3DFloatNonAccumulateModule_basic",
    "IndexPut3DIntAccumulateModule_basic",
    "IndexPut3DIntNonAccumulateModule_basic",
    "IndexPutHackedTwin1DFloatAccumulateModule_basic",
    "IndexPutHackedTwin1DFloatNonAccumulateModule_basic",
    "IndexPutHackedTwin1DIntAccumulateModule_basic",
    "IndexPutHackedTwin1DIntNonAccumulateModule_basic",
    "IndexPutHackedTwin2DFloatAccumulateModule_basic",
    "IndexPutHackedTwin2DFloatNonAccumulateModule_basic",
    "IndexPutHackedTwin2DIntAccumulateModule_basic",
    "IndexPutHackedTwin2DIntNonAccumulateModule_basic",
    "IndexPutHackedTwin3DFloatAccumulateModule_basic",
    "IndexPutHackedTwin3DFloatNonAccumulateModule_basic",
    "IndexPutHackedTwin3DIntAccumulateModule_basic",
    "IndexPutHackedTwin3DIntNonAccumulateModule_basic",
    "IndexPutImpl1DFloatAccumulateModule_basic",
    "IndexPutImpl1DFloatNonAccumulateModule_basic",
    "IndexPutImpl1DIntAccumulateModule_basic",
    "IndexPutImpl1DIntNonAccumulateModule_basic",
    "IndexPutImpl2DFloatAccumulateModule_basic",
    "IndexPutImpl2DFloatNonAccumulateModule_basic",
    "IndexPutImpl3DFloatAccumulateModule_basic",
    "IndexPutImpl3DFloatNonAccumulateModule_basic",
    "IndexTensorModule3dInput_basic",
    "IndexTensorModule_basic",
    "IndexTensorMultiInputContiguousCenter_basic",
    "IndexTensorMultiInputNonContiguous_basic",
    "IndexTensorMultiInputOneDim_basic",
    "IndexTensorMultiInputThreeIndexers_basic",
    "IndexTensorMultiInput_basic",
    "IndexTensorSelectDimModule_basic",
    "IndexTensorMultiInputContiguousOneDimDynamic_basic",
    "IndexTensorMultiInputNonContiguousOneDimDynamic_basic",
    "IndexTensorMultiInputNonContiguousDynamic_basic",
    "IndexTensorMultiInputNonContiguousMultipleStaticDims_basic",
    "IndexTensorHackedTwinModule_basic",
    "IndexTensorHackedTwinModule3dInput_basic",
    "IndexTensorHackedTwinMultiInputNonContiguousMultipleStaticDims_basic",
    "LiftFreshCopyModule_basic",
    "Matmul_dot",
    "MulIntModule_basic",
    "DivIntModule_basic",
    "NeFloatIntModule_basic",
    "NeIntModule_basic",
    "QuantizedMLP_basic",
    "RandLikeDtypeModule_basic",
    "RandLikeModule_basic",
    "RollModule_basic",
    "ScalarImplicitFloatModule_basic",
    "ScalarImplicitIntModule_basic",
    "SliceEndSleStartModule_basic",
    "SliceOutOfUpperBoundIndexModule_basic",
    "SliceStartEqEndModule_basic",
    "SqrtIntModule_basic",
    "StdBiasedModule_basic",
    "StdDimBiasedModule_basic",
    "StdDimKeepDimFalseModule_basic",
    "StdDimKeepDimTrueModule_basic",
    "StdDimEmptyDimModule_basic",
    "StdDimNoneDimModule_basic",
    "StdUnbiasedModule_basic",
    "SubFloatModule_basic",
    "SubIntModule_basic",
    "TensorsConcatNegativeDimModule_basic",
    "TensorToBoolZeroRank_basic",
    "TensorToBool_basic",
    "TensorToFloatZeroRank_basic",
    "TensorToFloat_basic",
    "TensorToIntZeroRank_basic",
    "TensorToInt_basic",
    "TensorsConcatModule_basic",
    "UniformModule_basic",
    "UniformStaticModule_basic",
    "UnsafeViewCollapseDynamicWithAtenSizeIntModule_basic",
    "ViewCollapseDynamicWithAtenSizeIntModule_basic",
    "AtenEmbeddingBagSumExample_basic",
    "Aten_EmbeddingBagExample_basic",
    "ElementwiseRemainderScalarModule_Int_Float_basic",
    "ElementwiseRemainderScalarModule_Float_basic",
    "ElementwiseRemainderScalarModule_Int_basic",
    "ElementwiseRemainderScalarModule_Bool_basic",
    "AtenIntTensorByteDtypeModule_basic",
    "AtenIntTensorCharDtypeModule_basic",
    "Fill_TensorFloat32WithFloat32_basic",
    "Fill_TensorFloat32WithFloat64_basic",
    "Fill_TensorFloat32WithInt64_basic",
    "UpSampleNearest2dBackwardVec_basic",
    "UpSampleNearest2dBackwardOutputSizeNone_basic",
    "ConvolutionBackwardModule2D_basic",
    "ConvolutionBackwardModule2DPadded_basic",
    "VarMeanCorrectionModule_basic",
    "VarMeanCorrectionNoneModule_basic",
    "PrimsConvertElementTypeModule_basic",
    "CopyModule_basic",
    "CopyWithDifferentDTypesAndSizesModule_basic",
    "CopyWithDifferentDTypesModule_basic",
    "CopyWithDifferentSizesModule_basic",
}
