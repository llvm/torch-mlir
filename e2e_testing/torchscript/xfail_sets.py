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
}
