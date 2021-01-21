# RUN: %PYTHON %s | FileCheck %s --dump-input=fail

#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import npcomp as npc
from npcomp.types import *


def dot2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  return np.dot(a, b)


# TODO: Implement subclassing and deriving constraints by run
exp = npc.Exporter()
exp.dot2d = dot2d
exp.dot2d.sig.args["a"] += Shape(4, 16)
exp.dot2d.sig.args["a"] += DynamicDim(0)
exp.dot2d.sig.args["a"] += DType(np.float32)
exp.dot2d.sig.args["b"] += Shape(16, 32)
exp.dot2d.sig.args["b"] += DType(np.float32)
exp.dot2d.sig.result += Shape(4, 32)
exp.dot2d.sig.result += DynamicDim(0)
exp.dot2d.sig.result += DType(np.float32)

mb = npc.tracing.ModuleBuilder()
mb.trace(exp.dot2d)

# CHECK-LABEL:   func @dot2d(
# CHECK-SAME:                %[[VAL_0:.*]]: tensor<?x16xf32>,
# CHECK-SAME:                %[[VAL_1:.*]]: tensor<16x32xf32>) -> tensor<?x32xf32> {
# CHECK:           %[[VAL_2:.*]] = numpy.dot %[[VAL_0]], %[[VAL_1]] : (tensor<?x16xf32>, tensor<16x32xf32>) -> tensor<*x!basicpy.UnknownType>
# CHECK:           %[[VAL_3:.*]] = numpy.narrow %[[VAL_2]] : (tensor<*x!basicpy.UnknownType>) -> tensor<?x32xf32>
# CHECK:           return %[[VAL_3]] : tensor<?x32xf32>
# CHECK:         }
print(mb.module)
