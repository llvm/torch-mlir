# RUN: %PYTHON %s | FileCheck %s --dump-input=fail

#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import npcomp as npc
from npcomp.types import *

weights = np.random.uniform(size=(16, 4)).astype(np.float32)
bias = np.random.uniform(size=(4,)).astype(np.float32)


def constants(a: np.ndarray) -> np.ndarray:
  return np.dot(a, weights) + bias


# TODO: Implement subclassing and deriving constraints by run
exp = npc.Exporter()
exp.constants = constants

mb = npc.tracing.ModuleBuilder()
mb.trace(exp.constants)
# CHECK-LABEL:   func @constants(
# CHECK-SAME:                    %[[VAL_0:.*]]: tensor<*x!numpy.any_dtype>) -> tensor<*x!numpy.any_dtype> {
# CHECK:           %[[VAL_1:.*]] = constant dense<{{.*}}> : tensor<16x4xf32>
# CHECK:           %[[VAL_2:.*]] = numpy.dot %[[VAL_0]], %[[VAL_1]] : (tensor<*x!numpy.any_dtype>, tensor<16x4xf32>) -> tensor<*x!basicpy.UnknownType>
# CHECK:           %[[VAL_3:.*]] = constant dense<{{.*}}> : tensor<4xf32>
# CHECK:           %[[VAL_4:.*]] = numpy.builtin_ufunc_call<"numpy.add"> (%[[VAL_2]], %[[VAL_3]]) : (tensor<*x!basicpy.UnknownType>, tensor<4xf32>) -> tensor<*x!basicpy.UnknownType>
# CHECK:           %[[VAL_5:.*]] = numpy.narrow %[[VAL_4]] : (tensor<*x!basicpy.UnknownType>) -> tensor<*x!numpy.any_dtype>
# CHECK:           return %[[VAL_5]] : tensor<*x!numpy.any_dtype>
# CHECK:         }
print(mb.module)
