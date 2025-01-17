# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

"""Torch-MLIR modular optimizer driver

Typically, when installed from a wheel, this can be invoked as:

  torch-mlir-opt [options] <input file>

To see available passes, dialects, and options, run:

  torch-mlir-opt --help
"""
import os
import platform
import subprocess
import sys

from typing import Optional


def _get_builtin_tool(exe_name: str) -> Optional[str]:
    if platform.system() == "Windows":
        exe_name = exe_name + ".exe"
    this_path = os.path.dirname(__file__)
    tool_path = os.path.join(this_path, "..", "..", "_mlir_libs", exe_name)
    return tool_path


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    exe = _get_builtin_tool("torch-mlir-opt")
    return subprocess.call(args=[exe] + args)


if __name__ == "__main__":
    sys.exit(main())
