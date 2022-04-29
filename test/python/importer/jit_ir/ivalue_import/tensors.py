# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

import typing

import torch
from torch_mlir.dialects.torch.importer.jit_ir import ModuleBuilder

# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s

mb = ModuleBuilder()

class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Test (and make work) tensors that alias each other.
        self.ones = torch.ones(1)
        self.ones_i32 = torch.ones(1, dtype=torch.int32)
        self.ones_i64 = torch.ones(1, dtype=torch.int64)
        self.ones_f32 = torch.ones(1, dtype=torch.float32)
        self.ones_f64 = torch.ones(1, dtype=torch.float64)
        self.ones_bool = torch.ones(1, dtype=torch.bool)
        self.ones_bf16 = torch.ones(1, dtype=torch.bfloat16)
        self.arange = torch.nn.Parameter(torch.arange(3.0))

# CHECK: %[[ARANGE:.*]] = torch.tensor.literal(dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xf32>) : !torch.tensor<[3],f32>
# CHECK: %[[ONES:.*]] = torch.tensor.literal(dense<1.000000e+00> : tensor<1xf32>) : !torch.tensor<[1],f32>
# CHECK: %[[ONES_I32:.*]] = torch.tensor.literal(dense<1> : tensor<1xsi32>) : !torch.tensor<[1],si32>
# CHECK: %[[ONES_I64:.*]] = torch.tensor.literal(dense<1> : tensor<1xsi64>) : !torch.tensor<[1],si64>
# CHECK: %[[ONES_F32:.*]] = torch.tensor.literal(dense<1.000000e+00> : tensor<1xf32>) : !torch.tensor<[1],f32>
# CHECK: %[[ONES_F64:.*]] = torch.tensor.literal(dense<1.000000e+00> : tensor<1xf64>) : !torch.tensor<[1],f64>
# CHECK: %[[ONES_BOOL:.*]] = torch.tensor.literal(dense<true> : tensor<1xi1>) : !torch.tensor<[1],i1>
# CHECK: %[[ONES_BF16:.*]] = torch.tensor.literal(dense<1.000000e+00> : tensor<1xbf16>) : !torch.tensor<[1],bf16>
# CHECK: %[[ROOT:.*]] = torch.nn_module  {
# CHECK:   torch.slot "arange", %[[ARANGE]] : !torch.tensor<[3],f32>
# CHECK:   torch.slot "ones", %[[ONES]] : !torch.tensor<[1],f32>
# CHECK:   torch.slot "ones_i32", %[[ONES_I32]] : !torch.tensor<[1],si32>
# CHECK:   torch.slot "ones_i64", %[[ONES_I64]] : !torch.tensor<[1],si64>
# CHECK:   torch.slot "ones_f32", %[[ONES_F32]] : !torch.tensor<[1],f32>
# CHECK:   torch.slot "ones_f64", %[[ONES_F64]] : !torch.tensor<[1],f64>
# CHECK:   torch.slot "ones_bool", %[[ONES_BOOL]] : !torch.tensor<[1],i1>
# CHECK:   torch.slot "ones_bf16", %[[ONES_BF16]] : !torch.tensor<[1],bf16>
# CHECK: }


test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c)
mb.module.operation.print()
