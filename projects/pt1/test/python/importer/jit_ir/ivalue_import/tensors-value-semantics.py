# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

import typing

import torch
from torch_mlir.jit_ir_importer import ClassAnnotator, ImportOptions, ModuleBuilder

# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s

mb = ModuleBuilder()


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ones_i32 = torch.ones(1, dtype=torch.int32)
        self.ones_qint8 = torch.quantize_per_tensor(torch.ones(1), 1.0, 0, torch.qint8)
        self.arange = torch.nn.Parameter(torch.arange(3.0))


# CHECK: %[[ARANGE:.*]] = torch.vtensor.literal(dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xf32>) : !torch.vtensor<[3],f32>
# CHECK: %[[ONES_I32:.*]] = torch.vtensor.literal(dense<1> : tensor<1xsi32>) : !torch.vtensor<[1],si32>
# CHECK: %[[ONES_QINT8_DATA:.*]] = torch.vtensor.literal(dense<1> : tensor<1xsi8>) : !torch.vtensor<[1],si8>
# CHECK: %[[SCALE:.*]] = torch.constant.float 1.000000e+00
# CHECK: %[[ZERO_POINT:.*]] = torch.constant.int 0
# CHECK: %[[ONES_QINT8:.*]] = torch.per_tensor_affine.create %[[ONES_QINT8_DATA]], %[[SCALE]], %[[ZERO_POINT]] : !torch.vtensor<[1],si8>, !torch.float, !torch.int -> !torch.vtensor<[1],!torch.qint8>
# CHECK: %[[ROOT:.*]] = torch.nn_module  {
# CHECK:   torch.slot "arange", %[[ARANGE]] : !torch.vtensor<[3],f32>
# CHECK:   torch.slot "ones_i32", %[[ONES_I32]] : !torch.vtensor<[1],si32>
# CHECK:   torch.slot "ones_qint8", %[[ONES_QINT8]] : !torch.vtensor<[1],!torch.qint8>
# CHECK: }
test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)

import_options = ImportOptions()
import_options.assumeTensorsHaveValueSemantics = True

class_annotator = ClassAnnotator()

# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c, class_annotator, import_options)
mb.module.operation.print()
