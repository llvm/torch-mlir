# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE for license information.

# Some checks that we can import the various extensions and libraries and
# not have symbol collisions or other goings on.
# RUN: %PYTHON %s

import sys

print(f"PYTHONPATH={sys.path}")

import mlir.ir
import torch_mlir

print("Extensions all loaded")
