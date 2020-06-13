# RUN: %PYTHON %s | npcomp-opt -split-input-file -basicpy-type-inference | FileCheck %s --dump-input=fail

from npcomp.compiler.frontend import *


def import_global(f):
  fe = ImportFrontend()
  fe.import_global_function(f)
  print(fe.ir_module.to_asm())
  print("// -----")
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


# CHECK-LABEL: func @arg_inference
# CHECK-SAME: (%arg0: i64, %arg1: i64) -> i64
@import_global
def arg_inference(a, b):
  # CHECK: basicpy.binary_expr{{.*}} : (i64, i64) -> i64
  # CHECK: basicpy.binary_expr{{.*}} : (i64, i64) -> i64
  # CHECK: basicpy.unknown_cast{{.*}} : i64 -> i64
  # CHECK: return{{.*}} : i64
  return a + 2 * b


# CHECK-LABEL: func @conditional_inference
# CHECK-SAME: (%arg0: i64, %arg1: !basicpy.BoolType, %arg2: i64) -> !basicpy.BoolType
@import_global
def conditional_inference(cond, a, b):
  # CHECK-NOT: UnknownType
  return a if cond + 1 else not (b * 4)
