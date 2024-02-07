# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# Lists of tests that fail to even reach the backends.
# These represent further work needed in torch-mlir to lower them properly
# to the backend contract.

from torch_mlir._version import torch_version_for_comparison, version

COMMON_TORCH_MLIR_LOWERING_XFAILS = {
    "NativeGroupNormBackwardModule_basic",
    "QuantizedMLP_basic",
    "ReduceMaxAlongDimUnsignedInt_basic",
    "ReduceMinAlongDimUnsignedInt_basic",
    "ElementwiseToDtypeI64ToUI8Module_basic",
}

# TODO: Delete once torch 2.1.0 is released
if torch_version_for_comparison() < version.parse("2.1.0.dev"):
    COMMON_TORCH_MLIR_LOWERING_XFAILS.update({
        "ScaledDotProductAttentionDifferentModule_basic",
        "ScaledDotProductAttentionSameModule_basic"
    })


def register_all_tests():
    """Registers all the built-in E2E tests that Torch-MLIR provides."""
    # Side-effecting import statements.
    from . import basic
    from . import vision_models
    from . import mlp
    from . import conv
    from . import norm_like
    from . import quantized_models
    from . import elementwise
    from . import type_promotion
    from . import type_conversion
    from . import backprop
    from . import reduction
    from . import matmul
    from . import reshape_like
    from . import scalar
    from . import scalar_comparison
    from . import elementwise_comparison
    from . import squeeze
    from . import slice_like
    from . import nll_loss
    from . import index_select
    from . import arange
    from . import constant_alloc
    from . import threshold
    from . import histogram_binning_calibration
    from . import rng
    from . import cast
    from . import scatter
    from . import pooling
    from . import return_types
    from . import control_flow
    from . import stats
    from . import padding
    from . import diagonal
