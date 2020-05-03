#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..types import *
from ..exporter import *
from .mlir_trace import *
from ..utils import test_utils

test_utils.start_filecheck_test()

def simple_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  return a * b + a + b

# TODO: Implement subclassing and deriving constraints by run
exp = Exporter()
exp.simple_mul = simple_mul
exp.simple_mul.sig.args["a"] += Shape(1, 4)
exp.simple_mul.sig.args["a"] += DynamicDim(0)
exp.simple_mul.sig.args["a"] += DType(np.float32)
exp.simple_mul.sig.args["b"] += Shape(1)
exp.simple_mul.sig.args["b"] += DType(np.float32)
exp.simple_mul.sig.result += Shape(1, 4)
exp.simple_mul.sig.result += DynamicDim(0)
exp.simple_mul.sig.result += DType(np.float32)    

mb = ModuleBuilder()
mb.trace(exp.simple_mul)
# CHECK: func @simple_mul(%arg0: tensor<?x4xf32>, %arg1: tensor<1xf32>) -> tensor<?x4xf32> {
# CHECK:  %0 = numpy.ufunc_call @numpy.multiply(%arg0, %arg1) : (tensor<?x4xf32>, tensor<1xf32>) -> tensor<*x!numpy.any_dtype>
# CHECK:  %1 = numpy.ufunc_call @numpy.add(%0, %arg0) : (tensor<*x!numpy.any_dtype>, tensor<?x4xf32>) -> tensor<*x!numpy.any_dtype>
# CHECK:  %2 = numpy.ufunc_call @numpy.add(%1, %arg1) : (tensor<*x!numpy.any_dtype>, tensor<1xf32>) -> tensor<*x!numpy.any_dtype>
# CHECK:  %3 = numpy.narrow %2 : (tensor<*x!numpy.any_dtype>) -> tensor<?x4xf32>
# CHECK:  return %3 : tensor<?x4xf32>
# CHECK: }
print(mb.module.to_asm())

test_utils.end_filecheck_test(__file__)
