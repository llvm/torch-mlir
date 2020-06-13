# RUN: %PYTHON %s | npcomp-opt -split-input-file | FileCheck %s --dump-input=fail

# Subset of constant tests which verify against a GenericTarget32.

from npcomp.compiler.frontend import *
from npcomp.compiler.target import *


def import_global(f):
  fe = ImportFrontend(target_factory=GenericTarget32)
  fe.import_global_function(f)
  print("// -----")
  print(fe.ir_module.to_asm())
  return f


# CHECK-LABEL: func @integer_constants
@import_global
def integer_constants():
  # CHECK: %[[A:.*]] = constant 100 : i32
  a = 100
  return a


# CHECK-LABEL: func @float_constants
@import_global
def float_constants():
  # CHECK: %[[A:.*]] = constant 2.200000e+00 : f32
  a = 2.2
  return a
