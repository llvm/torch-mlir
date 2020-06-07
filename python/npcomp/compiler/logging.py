#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import string
import sys

__all__ = ["debug"]

_ENABLED = "NPCOMP_DEBUG" in os.environ
_formatter = string.Formatter()


def debug(format_string, *args, **kwargs):
  if not _ENABLED:
    return
  formatted = _formatter.vformat(format_string, args, kwargs)
  print("DEBUG:", formatted, file=sys.stderr)
