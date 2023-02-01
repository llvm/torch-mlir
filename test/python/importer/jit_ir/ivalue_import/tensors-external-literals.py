# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

import typing

import torch
from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ImportOptions, ModuleBuilder

# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.big = torch.ones(512, 1024)


test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)

import_options = ImportOptions()
import_options.assumeTensorsHaveValueSemantics = True
import_options.useExternalReferencesIfNumelExceeds = 16
class_annotator = ClassAnnotator()
mb = ModuleBuilder()
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c, class_annotator, import_options)
# CHECK:         %[[VTENSOR:.*]] = torch.vtensor.external.literal(@big) : !torch.vtensor<[512,1024],f32>
# CHECK-LABEL:   %{{.*}} = torch.nn_module {
# CHECK:           torch.slot "big", %[[VTENSOR]] : !torch.vtensor<[512,1024],f32>
mb.module.operation.print()


import_options = ImportOptions()
import_options.assumeTensorsHaveValueSemantics = False
import_options.useExternalReferencesIfNumelExceeds = 16
class_annotator = ClassAnnotator()
mb = ModuleBuilder()
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c, class_annotator, import_options)
# CHECK:         %[[VTENSOR:.*]] = torch.tensor.external.literal(@big) : !torch.tensor<[512,1024],f32>
# CHECK-LABEL:   %{{.*}} = torch.nn_module {
# CHECK:           torch.slot "big", %[[VTENSOR]] : !torch.tensor<[512,1024],f32>
mb.module.operation.print()

# Test that the number of elements check works as intended.
import_options = ImportOptions()
import_options.assumeTensorsHaveValueSemantics = False
import_options.useExternalReferencesIfNumelExceeds = 2**20
class_annotator = ClassAnnotator()
mb = ModuleBuilder()
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c, class_annotator, import_options)
# CHECK:         %[[VTENSOR:.*]] = torch.tensor.literal(dense<1.000000e+00> : tensor<512x1024xf32>) : !torch.tensor<[512,1024],f32>
# CHECK-LABEL:   %{{.*}} = torch.nn_module {
# CHECK:           torch.slot "big", %[[VTENSOR]] : !torch.tensor<[512,1024],f32>
mb.module.operation.print()
