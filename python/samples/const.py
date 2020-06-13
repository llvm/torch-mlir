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
print(mb.module.to_asm())
