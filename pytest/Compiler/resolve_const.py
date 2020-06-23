# RUN: %PYTHON %s | npcomp-opt -split-input-file | FileCheck %s --dump-input=fail
"""Module docstring."""

from npcomp.compiler.frontend import *


def import_global(f):
  fe = ImportFrontend()
  fe.import_global_function(f)
  print("// -----")
  print(fe.ir_module.to_asm())
  return f


OUTER_ONE = 1
OUTER_STRING = "Hello"


# CHECK-LABEL: func @global_int
@import_global
def global_int():
  # CHECK: constant 1 : i64
  return OUTER_ONE


# CHECK-LABEL: func @module_docstring
@import_global
def module_docstring():
  # CHECK: basicpy.str_constant "Module docstring."
  return __doc__


# CHECK-LABEL: func @builtin_debug
@import_global
def builtin_debug():
  # CHECK: basicpy.bool_constant
  return __debug__
