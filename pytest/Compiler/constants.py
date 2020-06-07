# RUN: %PYTHON %s | npcomp-opt -split-input-file | FileCheck %s --dump-input=fail

from npcomp.compiler.frontend import *


def import_global(f):
  fe = ImportFrontend()
  fe.import_global_function(f)
  print("// -----")
  print(fe.ir_module.to_asm())
  return f


# CHECK-LABEL: func @integer_constants
@import_global
def integer_constants():
  # CHECK: %[[A:.*]] = constant 100 : i64
  a = 100
  # CHECK: %[[A_CAST:.*]] = basicpy.unknown_cast %[[A]] : i64 -> !basicpy.UnknownType
  # CHECK: return %[[A_CAST]]
  return a

# CHECK-LABEL: func @float_constants
@import_global
def float_constants():
  # CHECK: %[[A:.*]] = constant 2.200000e+00 : f64
  a = 2.2
  # CHECK: %[[A_CAST:.*]] = basicpy.unknown_cast %[[A]] : f64 -> !basicpy.UnknownType
  # CHECK: return %[[A_CAST]]
  return a

# CHECK-LABEL: func @bool_true_constant
@import_global
def bool_true_constant():
  # CHECK: %[[A:.*]] = basicpy.bool_constant 1
  # CHECK: basicpy.unknown_cast %[[A]]
  a = True
  return a

# CHECK-LABEL: func @bool_false_constant
@import_global
def bool_false_constant():
  # CHECK: %[[A:.*]] = basicpy.bool_constant 0
  # CHECK: basicpy.unknown_cast %[[A]]
  a = False
  return a

# CHECK-LABEL: func @string_constant
@import_global
def string_constant():
  # CHECK: %[[A:.*]] = basicpy.str_constant "foobar"
  # CHECK: basicpy.unknown_cast %[[A]]
  a = "foobar"
  return a

# CHECK-LABEL: func @joined_string_constant
@import_global
def joined_string_constant():
  # CHECK: %[[A:.*]] = basicpy.str_constant "I am still here"
  # CHECK: basicpy.unknown_cast %[[A]]
  a = "I am" " still here"
  return a

# CHECK-LABEL: func @ellipsis
@import_global
def ellipsis():
  # CHECK: %[[A:.*]] = basicpy.singleton : !basicpy.EllipsisType
  # CHECK: basicpy.unknown_cast %[[A]]
  a = ...
  return a
