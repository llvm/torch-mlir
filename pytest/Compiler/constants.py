# RUN: %PYTHON %s | npcomp-opt -split-input-file | FileCheck %s --dump-input=fail

from npcomp.compiler.frontend import *


def import_global(f):
  fe = ImportFrontend()
  fe.import_global_function(f)
  print("// -----")
  print(fe.ir_module.to_asm())


# CHECK: func @integer_constants
@import_global
def integer_constants():
  # CHECK: %[[A:.*]] = constant 100 : i64
  a = 100
  # CHECK: %[[A_CAST:.*]] = basicpy.unknown_cast %[[A]] : i64 -> !basicpy.UnknownType
  # CHECK: return %[[A_CAST]]
  return a

# CHECK: func @float_constants
@import_global
def float_constants():
  # CHECK: %[[A:.*]] = constant 2.200000e+00 : f64
  a = 2.2
  # CHECK: %[[A_CAST:.*]] = basicpy.unknown_cast %[[A]] : f64 -> !basicpy.UnknownType
  # CHECK: return %[[A_CAST]]
  return a
