# RUN: %PYTHON %s | FileCheck %s --dump-input=fail

#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import npcomp as npc
from npcomp.types import *


def slice_array1(a: np.ndarray) -> np.ndarray:
  return a[1, 2:10:2, 3:4, ..., :, 0]


# TODO: Implement subclassing and deriving constraints by run
exp = npc.Exporter()
exp.slice_array1 = slice_array1

mb = npc.tracing.ModuleBuilder()
mb.trace(exp.slice_array1)

# TODO: The numpy.get_slice op emission should be analyzed: it probably
# needs to both accept and produce either arrays or tensors and the following
# narrow should do likewise.
# CHECK-LABEL:   func @slice_array1(
# CHECK-SAME:                       %[[VAL_0:.*]]: tensor<*x!numpy.any_dtype>) -> tensor<*x!numpy.any_dtype> {
# CHECK:           %[[VAL_1:.*]] = constant 1 : index
# CHECK:           %[[VAL_2:.*]] = constant 2 : index
# CHECK:           %[[VAL_3:.*]] = constant 10 : index
# CHECK:           %[[VAL_4:.*]] = constant 2 : index
# CHECK:           %[[VAL_5:.*]] = basicpy.slot_object_make(%[[VAL_2]], %[[VAL_3]], %[[VAL_4]]) -> !basicpy.SlotObject<slice, index, index, index>
# CHECK:           %[[VAL_6:.*]] = constant 3 : index
# CHECK:           %[[VAL_7:.*]] = constant 4 : index
# CHECK:           %[[VAL_8:.*]] = basicpy.singleton : !basicpy.NoneType
# CHECK:           %[[VAL_9:.*]] = basicpy.slot_object_make(%[[VAL_6]], %[[VAL_7]], %[[VAL_8]]) -> !basicpy.SlotObject<slice, index, index, !basicpy.NoneType>
# CHECK:           %[[VAL_10:.*]] = basicpy.singleton : !basicpy.EllipsisType
# CHECK:           %[[VAL_11:.*]] = basicpy.singleton : !basicpy.NoneType
# CHECK:           %[[VAL_12:.*]] = basicpy.singleton : !basicpy.NoneType
# CHECK:           %[[VAL_13:.*]] = basicpy.singleton : !basicpy.NoneType
# CHECK:           %[[VAL_14:.*]] = basicpy.slot_object_make(%[[VAL_11]], %[[VAL_12]], %[[VAL_13]]) -> !basicpy.SlotObject<slice, !basicpy.NoneType, !basicpy.NoneType, !basicpy.NoneType>
# CHECK:           %[[VAL_15:.*]] = constant 0 : index
# CHECK:           %[[VAL_16:.*]] = numpy.get_slice %[[VAL_0]], %[[VAL_1]], %[[VAL_5]], %[[VAL_9]], %[[VAL_10]], %[[VAL_14]], %[[VAL_15]] : (tensor<*x!numpy.any_dtype>, index, !basicpy.SlotObject<slice, index, index, index>, !basicpy.SlotObject<slice, index, index, !basicpy.NoneType>, !basicpy.EllipsisType, !basicpy.SlotObject<slice, !basicpy.NoneType, !basicpy.NoneType, !basicpy.NoneType>, index) -> !numpy.ndarray<*:?>
# CHECK:           %[[VAL_17:.*]] = numpy.narrow %[[VAL_16]] : (!numpy.ndarray<*:?>) -> tensor<*x!numpy.any_dtype>
# CHECK:           return %[[VAL_17]] : tensor<*x!numpy.any_dtype>
# CHECK:         }

print(mb.module)
