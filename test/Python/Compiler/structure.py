# RUN: %PYTHON %s | npcomp-opt -split-input-file | FileCheck %s --dump-input=fail

from npcomp.compiler import test_config

import_global = test_config.create_import_dump_decorator()


# CHECK-LABEL: func @positional_args
# CHECK-SAME: (%arg0: !basicpy.UnknownType, %arg1: !basicpy.UnknownType) -> !basicpy.UnknownType
@import_global
def positional_args(a, b):
  # CHECK: basicpy.binary_expr %arg0 "Add" %arg1
  return a + b


# CHECK-LABEL: func @pass_no_return
@import_global
def pass_no_return():
  # CHECK: %[[NONE:.*]] = basicpy.singleton : !basicpy.NoneType
  # CHECK: %[[NONE_CAST:.*]] =  basicpy.unknown_cast %[[NONE]] : !basicpy.NoneType -> !basicpy.UnknownType
  # CHECK: return %[[NONE_CAST]]
  # CHECK-NOT: return
  pass


# CHECK-LABEL: func @expr_statement
@import_global
def expr_statement():
  # CHECK: basicpy.exec {
  # CHECK:   %[[V:.*]] = basicpy.binary_expr
  # CHECK:   basicpy.exec_discard %[[V]]
  # CHECK: }
  # CHECK: return
  a = 1
  a + 2
