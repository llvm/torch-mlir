# Some checks that we can import the various extensions and libraries and
# not have symbol collisions or other goings on.
# RUN: %PYTHON %s

import sys

print(f"PYTHONPATH={sys.path}")

import mlir
import npcomp
import _npcomp

print("Extensions all loaded")
