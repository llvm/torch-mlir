# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

from typing import List

import torch
import torch._dynamo as dynamo
from torch_mlir.dynamo import make_simple_dynamo_backend
from torch_mlir_e2e_test.debug.lockstep import make_lockstep_debug_backend


@make_simple_dynamo_backend
@make_lockstep_debug_backend()
def miscompile_div_as_mul_backend(
    gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
):
    # Copy `gm` and rewrite `div` to `mul`.
    new_g = torch.fx.Graph()
    new_g.output(new_g.graph_copy(gm.graph, {}))
    for node in new_g.nodes:
        if node.op == "call_function":
            if node.target == torch.ops.aten.div:
                node.target = torch.ops.aten.mul
    new_gm = torch.fx.GraphModule(torch.nn.Module(), new_g)
    return new_gm


# TODO: As we get smarter about making this output more readable, we should
# have more focused tests rather than this "check the exact output" test.
# CHECK:      User result tensor([ 4., 10., 18.]) is not close to golden result tensor([0.2500, 0.4000, 0.5000]) for node div at
# CHECK-SAME:   File "{{.*}}python/test/debug/lockstep_basic.py", line {{.*}}, in f
# CHECK-NEXT:     c = x / y
@dynamo.optimize(miscompile_div_as_mul_backend)
def f(x, y):
    a = x * y
    b = x + y
    c = x / y
    return a, b, c


args = (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0]))
try:
    print(f(*args))
except AssertionError as e:
    print(e)
