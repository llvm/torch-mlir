# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

import typing

import torch
from torch_mlir.jit_ir_importer import ModuleBuilder

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
        # Because bools turn anything that is non-zero into `True`, it is
        # important to check a series of `True`s and `False`s to make sure the
        # actual values are being imported rather than just garbage.
        self.bool_ = torch.tensor(
            [True, False, True, False, True, False], dtype=torch.bool
        )
        self.ones_bf16 = torch.ones(1, dtype=torch.bfloat16)
        self.ones_f16 = torch.ones(1, dtype=torch.half)
        self.ones_ui8 = torch.ones(1, dtype=torch.uint8)
        self.ones_i8 = torch.ones(1, dtype=torch.int8)
        self.ones_qint8 = torch.quantize_per_tensor(torch.ones(1), 1.0, 0, torch.qint8)
        self.ones_quint8 = torch.quantize_per_tensor(
            torch.ones(1), 1.0, 0, torch.quint8
        )
        self.arange = torch.nn.Parameter(torch.arange(3.0))


# CHECK: %[[ARANGE:.*]] = torch.tensor.literal(dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xf32>) : !torch.tensor<[3],f32>
# CHECK: %[[ONES:.*]] = torch.tensor.literal(dense<1.000000e+00> : tensor<1xf32>) : !torch.tensor<[1],f32>
# CHECK: %[[ONES_I32:.*]] = torch.tensor.literal(dense<1> : tensor<1xsi32>) : !torch.tensor<[1],si32>
# CHECK: %[[ONES_I64:.*]] = torch.tensor.literal(dense<1> : tensor<1xsi64>) : !torch.tensor<[1],si64>
# CHECK: %[[ONES_F32:.*]] = torch.tensor.literal(dense<1.000000e+00> : tensor<1xf32>) : !torch.tensor<[1],f32>
# CHECK: %[[ONES_F64:.*]] = torch.tensor.literal(dense<1.000000e+00> : tensor<1xf64>) : !torch.tensor<[1],f64>
# CHECK: %[[BOOL_:.*]] = torch.tensor.literal(dense<[true, false, true, false, true, false]> : tensor<6xi1>) : !torch.tensor<[6],i1>
# CHECK: %[[ONES_BF16:.*]] = torch.tensor.literal(dense<1.000000e+00> : tensor<1xbf16>) : !torch.tensor<[1],bf16>
# CHECK: %[[ONES_F16:.*]] = torch.tensor.literal(dense<1.000000e+00> : tensor<1xf16>) : !torch.tensor<[1],f16>
# CHECK: %[[ONES_UI8:.*]] = torch.tensor.literal(dense<1> : tensor<1xui8>) : !torch.tensor<[1],ui8>
# CHECK: %[[ONES_I8:.*]] = torch.tensor.literal(dense<1> : tensor<1xsi8>) : !torch.tensor<[1],si8>
# CHECK: %[[ONES_QINT8_DATA:.*]] = torch.tensor.literal(dense<1> : tensor<1xsi8>) : !torch.tensor<[1],si8>
# CHECK: %[[SCALE:.*]] = torch.constant.float 1.000000e+00
# CHECK: %[[ZERO_POINT:.*]] = torch.constant.int 0
# CHECK: %[[ONES_QINT8:.*]] = torch.per_tensor_affine.create %[[ONES_QINT8_DATA]], %[[SCALE]], %[[ZERO_POINT]] : !torch.tensor<[1],si8>, !torch.float, !torch.int -> !torch.tensor<[1],!torch.qint8>
# CHECK: %[[ONES_QUINT8_DATA:.*]] = torch.tensor.literal(dense<1> : tensor<1xui8>) : !torch.tensor<[1],ui8>
# CHECK: %[[ONES_QUINT8:.*]] = torch.per_tensor_affine.create %[[ONES_QUINT8_DATA]], %[[SCALE]], %[[ZERO_POINT]] : !torch.tensor<[1],ui8>, !torch.float, !torch.int -> !torch.tensor<[1],!torch.quint8>
# CHECK: %[[ROOT:.*]] = torch.nn_module  {
# CHECK:   torch.slot "arange", %[[ARANGE]] : !torch.tensor<[3],f32>
# CHECK:   torch.slot "ones", %[[ONES]] : !torch.tensor<[1],f32>
# CHECK:   torch.slot "ones_i32", %[[ONES_I32]] : !torch.tensor<[1],si32>
# CHECK:   torch.slot "ones_i64", %[[ONES_I64]] : !torch.tensor<[1],si64>
# CHECK:   torch.slot "ones_f32", %[[ONES_F32]] : !torch.tensor<[1],f32>
# CHECK:   torch.slot "ones_f64", %[[ONES_F64]] : !torch.tensor<[1],f64>
# CHECK:   torch.slot "bool_", %[[BOOL_]] : !torch.tensor<[6],i1>
# CHECK:   torch.slot "ones_bf16", %[[ONES_BF16]] : !torch.tensor<[1],bf16>
# CHECK:   torch.slot "ones_f16", %[[ONES_F16]] : !torch.tensor<[1],f16>
# CHECK:   torch.slot "ones_ui8", %[[ONES_UI8]] : !torch.tensor<[1],ui8>
# CHECK:   torch.slot "ones_i8", %[[ONES_I8]] : !torch.tensor<[1],si8>
# CHECK:   torch.slot "ones_qint8", %[[ONES_QINT8]] : !torch.tensor<[1],!torch.qint8>
# CHECK:   torch.slot "ones_quint8", %[[ONES_QUINT8]] : !torch.tensor<[1],!torch.quint8>
# CHECK: }


test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c)
mb.module.operation.print()
