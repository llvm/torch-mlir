# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

import typing

import torch
from torch_mlir.jit_ir_importer import ModuleBuilder

# UNSUPPORTED: system-darwin
# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s

mb = ModuleBuilder()


# CHECK-LABEL: torch.class_type @__torch__.TestModule {
class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.quantized.Linear(5, 2, dtype=torch.qint8)
        self.linear_no_bias = torch.nn.quantized.Linear(
            6, 2, bias_=False, dtype=torch.qint8
        )

    # CHECK: %[[SCALE:.*]] = torch.constant.float
    # CHECK: %[[ZERO_POINT:.*]] = torch.constant.int 0
    # CHECK: %[[INT_REPR:.*]] = torch.tensor.literal({{.*}}) : !torch.tensor<[2,5],si8>
    # CHECK: %[[WEIGHTS:.*]] = torch.per_tensor_affine.create %[[INT_REPR]], %[[SCALE]], %[[ZERO_POINT]] : !torch.tensor<[2,5],si8>, !torch.float, !torch.int -> !torch.tensor<[2,5],!torch.qint8>
    # CHECK: %[[BIAS:.*]] = torch.tensor.literal({{.*}}) : !torch.tensor<[2],f32>
    # CHECK: %[[LINEAR_PARAMS:.*]] = torch.linear_params.create %[[WEIGHTS]], %[[BIAS]] : !torch.tensor<[2,5],!torch.qint8>, !torch.tensor<[2],f32>
    @torch.jit.export
    def test_linear(self, t):
        return self.linear(t)

    # CHECK: %[[LINEAR_PARAMS_NO_BIAS:.*]] = torch.linear_params.create %{{[^,]*}} : !torch.tensor<[2,6],!torch.qint8>
    @torch.jit.export
    def test_linear_no_bias(self, t):
        return self.linear_no_bias(t)


test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c)
mb.module.operation.print()
