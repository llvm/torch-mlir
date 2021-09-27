# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

# TODO: Fix ODR violation on non-static cl::opt in LLVM
# `cl::opt<FunctionSummary::ForceSummaryHotnessType, true>`.
# This causes double free on global dtors on exiting the program.
# The FileCheck still passes though.
# RUN: (%PYTHON %s || true) | FileCheck %s

from torch_mlir._mlir_libs._jit_ir_importer import get_registered_ops

# This check is just for a built-in op that is unlikely to change (and is
# otherwise insignificant).
# CHECK: {'name': ('aten::mul', 'Tensor'), 'is_c10_op': True, 'is_vararg': False, 'is_varret': False, 'is_mutable': False, 'arguments': [{'name': 'self', 'type': 'Tensor', 'pytype': 'Tensor'}, {'name': 'other', 'type': 'Tensor', 'pytype': 'Tensor'}], 'returns': [{'name': '', 'type': 'Tensor', 'pytype': 'Tensor'}]}
print('\n\n'.join([repr(r) for r in get_registered_ops()]))
