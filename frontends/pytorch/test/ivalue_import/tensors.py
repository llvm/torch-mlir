# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import typing

import torch
import torch_mlir

# RUN: %PYTHON %s | npcomp-opt | FileCheck %s

mb = torch_mlir.ModuleBuilder()

class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Test (and make work) tensors that alias each other.
        self.t = torch.ones(1)
        self.p = torch.nn.Parameter(torch.arange(3.0))

# CHECK:         %[[CP:.*]] = constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xf32>
# CHECK:         %[[P:.*]] = numpy.create_array_from_tensor %[[CP]] : (tensor<3xf32>) -> !numpy.ndarray<*:!numpy.any_dtype>
# CHECK:         %[[CT:.*]] = constant dense<1.000000e+00> : tensor<1xf32>
# CHECK:         %[[T:.*]] = numpy.create_array_from_tensor %[[CT]] : (tensor<1xf32>) -> !numpy.ndarray<*:!numpy.any_dtype>
# CHECK:         %[[ROOT:.*]] = torch.nn_module  {
# CHECK:           torch.slot "p", %[[P]] : !numpy.ndarray<*:!numpy.any_dtype>
# CHECK:           torch.slot "t", %[[T]] : !numpy.ndarray<*:!numpy.any_dtype>
# CHECK:         }


test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c)
mb.module.operation.print()
