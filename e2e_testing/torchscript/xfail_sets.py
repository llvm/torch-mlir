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

XFAIL_SETS = {}

# Lists of tests that fail to even reach the backends.
# These represent further work needed in torch-mlir to lower them properly
# to the backend contract.
_common_torch_mlir_lowering_xfails = {
    'QuantizedMLP_basic',
}

XFAIL_SETS['refbackend'] = _common_torch_mlir_lowering_xfails

XFAIL_SETS['torchscript'] = {}

XFAIL_SETS['native_torch'] = {}
