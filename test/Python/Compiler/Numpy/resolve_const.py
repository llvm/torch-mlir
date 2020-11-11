# RUN: %PYTHON %s | npcomp-opt -split-input-file | FileCheck %s --dump-input=fail
"""Module docstring."""

from npcomp.compiler.numpy import test_config

import_global = test_config.create_import_dump_decorator()

OUTER_ONE = 1
OUTER_STRING = "Hello"


# CHECK-LABEL: func @global_int
@import_global
def global_int():
  # CHECK: constant 1 : i64
  return OUTER_ONE


# CHECK-LABEL: func @module_string
@import_global
def module_string():
  # CHECK: basicpy.str_constant "Hello"
  return OUTER_STRING


# CHECK-LABEL: func @builtin_debug
@import_global
def builtin_debug():
  # CHECK: basicpy.bool_constant
  return __debug__
