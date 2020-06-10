# RUN: %PYTHON %s | npcomp-opt -split-input-file | FileCheck %s --dump-input=fail

from npcomp.compiler.frontend import *


def import_global(f):
  fe = ImportFrontend()
  fe.import_global_function(f)
  print("// -----")
  print(fe.ir_module.to_asm())
  return f

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
