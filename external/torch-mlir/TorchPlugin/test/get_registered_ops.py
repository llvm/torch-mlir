# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE for license information.
# RUN: %PYTHON %s | FileCheck %s

import _torch_mlir

# This check is just for a built-in op that is unlikely to change (and is
# otherwise insignificant).
# CHECK: {'name': ('aten::mul', 'Tensor'), 'is_c10_op': True, 'is_vararg': False, 'is_varret': False, 'is_mutable': False, 'arguments': [{'name': 'self', 'type': 'Tensor', 'pytype': 'Tensor'}, {'name': 'other', 'type': 'Tensor', 'pytype': 'Tensor'}], 'returns': [{'name': '', 'type': 'Tensor', 'pytype': 'Tensor'}]}
print('\n\n'.join([repr(r) for r in _torch_mlir.get_registered_ops()]))
