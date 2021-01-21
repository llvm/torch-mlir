# RUN: %PYTHON %s | FileCheck %s --dump-input=fail

#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import npcomp as npc
from npcomp.types import *


def transpose_attribute(a: np.ndarray) -> np.ndarray:
  return a.T


def transpose(a: np.ndarray) -> np.ndarray:
  return np.transpose(a)


# TODO: Implement subclassing and deriving constraints by run
exp = npc.Exporter()
exp.transpose_attribute = transpose_attribute
exp.transpose = transpose

mb = npc.tracing.ModuleBuilder()
mb.trace(exp.transpose_attribute, exp.transpose)

# TODO: Consolidate any_dtype -> UnknownType.
# CHECK-LABEL:   func @transpose_attribute(
# CHECK-SAME:                              %[[VAL_0:.*]]: tensor<*x!numpy.any_dtype>) -> tensor<*x!numpy.any_dtype> {
# CHECK:           %[[VAL_1:.*]] = numpy.transpose %[[VAL_0]] : (tensor<*x!numpy.any_dtype>) -> tensor<*x!basicpy.UnknownType>
# CHECK:           %[[VAL_2:.*]] = numpy.narrow %[[VAL_1]] : (tensor<*x!basicpy.UnknownType>) -> tensor<*x!numpy.any_dtype>
# CHECK:           return %[[VAL_2]] : tensor<*x!numpy.any_dtype>
# CHECK:         }

# CHECK-LABEL:   func @transpose(
# CHECK-SAME:                    %[[VAL_0:.*]]: tensor<*x!numpy.any_dtype>) -> tensor<*x!numpy.any_dtype> {
# CHECK:           %[[VAL_1:.*]] = numpy.transpose %[[VAL_0]] : (tensor<*x!numpy.any_dtype>) -> tensor<*x!basicpy.UnknownType>
# CHECK:           %[[VAL_2:.*]] = numpy.narrow %[[VAL_1]] : (tensor<*x!basicpy.UnknownType>) -> tensor<*x!numpy.any_dtype>
# CHECK:           return %[[VAL_2]] : tensor<*x!numpy.any_dtype>
# CHECK:         }
print(mb.module)
