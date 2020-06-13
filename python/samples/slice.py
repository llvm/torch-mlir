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
print(mb.module.to_asm())
