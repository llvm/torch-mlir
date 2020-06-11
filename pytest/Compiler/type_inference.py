# RUN: %PYTHON %s | npcomp-opt -split-input-file -basicpy-type-inference | FileCheck %s --dump-input=fail

from npcomp.compiler.frontend import *


def import_global(f):
  fe = ImportFrontend()
  fe.import_global_function(f)
  print(fe.ir_module.to_asm())
  return f

# CHECK-LABEL: func @arithmetic_expression
# CHECK-SAME: () -> i64
@import_global
def arithmetic_expression():
  # CHECK: basicpy.binary_expr{{.*}} : (i64, i64) -> i64
  # CHECK: basicpy.binary_expr{{.*}} : (i64, i64) -> i64
  # CHECK: basicpy.binary_expr{{.*}} : (i64, i64) -> i64
  # CHECK: basicpy.unknown_cast{{.*}} : i64 -> i64
  # CHECK: return{{.*}} : i64
  return 1 + 2 - 3 * 4
