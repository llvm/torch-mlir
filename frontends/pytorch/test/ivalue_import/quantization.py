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
        self.linear = torch.nn.quantized.Linear(5, 2, dtype=torch.qint8)
        self.linear_no_bias = torch.nn.quantized.Linear(6,
                                                        2,
                                                        bias_=False,
                                                        dtype=torch.qint8)
    # CHECK-DAG:  %[[SCALE:.*]] = basicpy.numeric_constant {{.*}} : f64
    # CHECK-DAG:  %[[ZERO_POINT:.*]] = basicpy.numeric_constant 0 : i64
    # CHECK-DAG:  %[[INT_REPR:.*]] = constant dense<{{.*}}> : tensor<2x5xi8>
    # CHECK-DAG:  %[[WEIGHTS:.*]] = torch.per_tensor_affine.create %[[INT_REPR]], %[[SCALE]], %[[ZERO_POINT]] : tensor<2x5xi8>, f64, i64 -> tensor<2x5x!torch.qint8>
    # CHECK-DAG:  %[[WEIGHTS_ARRAY:.*]] = numpy.create_array_from_tensor %[[WEIGHTS]] : (tensor<2x5x!torch.qint8>) -> !numpy.ndarray<*:!numpy.any_dtype>
    # CHECK-DAG:  %[[BIAS:.*]] = constant dense<{{.*}}> : tensor<2xf32>
    # CHECK-DAG:  %[[BIAS_ARRAY:.*]] = numpy.create_array_from_tensor %[[BIAS]] : (tensor<2xf32>) -> !numpy.ndarray<*:!numpy.any_dtype>
    # CHECK-DAG:  %[[LINEAR_PARAMS:.*]] = torch.linear_params.create %[[WEIGHTS_ARRAY]], %[[BIAS_ARRAY]] : !numpy.ndarray<*:!numpy.any_dtype>, !numpy.ndarray<*:!numpy.any_dtype>
    @torch.jit.export
    def test_linear(self, t):
        return self.linear(t)

    # CHECK: %[[LINEAR_PARAMS_NO_BIAS:.*]] = torch.linear_params.create %{{.*}} : !numpy.ndarray<*:!numpy.any_dtype>{{$}}
    @torch.jit.export
    def test_linear_no_bias(self, t):
        return self.linear_no_bias(t)


test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c)
mb.module.operation.print()
