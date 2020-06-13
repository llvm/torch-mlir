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
print(mb.module.to_asm())
