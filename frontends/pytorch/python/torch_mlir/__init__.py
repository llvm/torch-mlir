#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This is a trampoline module which loads the _torch_mlir native module
# and binds names locally. It exists to allow for customization of behavior
# prior to loading shared objects.

import torch
import npcomp

# Our native extension is not self-contained. It references libraries which
# must come in via the above first.
from _torch_mlir import *


__all__ = [
  "debug_trace_to_stderr",
  "ModuleBuilder",
  "ClassAnnotator",
]
