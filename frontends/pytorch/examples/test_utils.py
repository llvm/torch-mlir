# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import sys
import textwrap

import numpy as np

INDENT = "  "


def _indent(value):
  return textwrap.indent(str(value), INDENT)


def compare_outputs(torch_func, jit_func, *args):
  print('—' * 80)

  print(f"Input args:\n{_indent(args)}", file=sys.stderr)
  result = torch_func(*args)
  jit_result = jit_func(*args)

  np.testing.assert_allclose(result.numpy(), jit_result)

  # Only print these if the test passes, as np.testing will print them if it
  # fails.
  print(f"PyTorch Result:\n{_indent(result.numpy())}", file=sys.stderr)
  print(f"JIT Result:\n{_indent(jit_result)}", file=sys.stderr)
