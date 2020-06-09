# RUN: %PYTHON %s | npcomp-opt -split-input-file | FileCheck %s --dump-input=fail

from npcomp.compiler.frontend import *


def import_global(f):
  fe = ImportFrontend()
  fe.import_global_function(f)
  print("// -----")
  print(fe.ir_module.to_asm())
  return f


# Full checking for add. Others just check validity.
# CHECK-LABEL: func @add
@import_global
def add():
  # CHECK: %[[A:.*]] = constant 1 : i64
  # CHECK: %[[B:.*]] = constant 2 : i64
  a = 1
  b = 2
  # CHECK: {{.*}} = basicpy.binary_expr %[[A]] "Add" %[[B]] : (i64, i64) -> !basicpy.UnknownType
  return a + b

# CHECK-LABEL: func @sub
@import_global
def sub():
  # CHECK: basicpy.binary_expr {{.*}} "Sub"
  return 4 - 2

# CHECK-LABEL: func @mult
@import_global
def mult():
  # CHECK: basicpy.binary_expr {{.*}} "Mult"
  return 4 * 2

# CHECK-LABEL: func @div
@import_global
def div():
  # CHECK: basicpy.binary_expr {{.*}} "Div"
  return 4 / 2

# CHECK-LABEL: func @floor_div
@import_global
def floor_div():
  # CHECK: basicpy.binary_expr {{.*}} "FloorDiv"
  return 4 // 2

# CHECK-LABEL: func @matmul
@import_global
def matmul():
  # CHECK: basicpy.binary_expr {{.*}} "MatMult"
  return 4 @ 2

# CHECK-LABEL: func @modulo
@import_global
def modulo():
  # CHECK: basicpy.binary_expr {{.*}} "Mod"
  return 4 % 2

# CHECK-LABEL: func @left_shift
@import_global
def left_shift():
  # CHECK: basicpy.binary_expr {{.*}} "LShift"
  return 4 << 2

# CHECK-LABEL: func @right_shift
@import_global
def right_shift():
  # CHECK: basicpy.binary_expr {{.*}} "RShift"
  return 4 >> 2

# CHECK-LABEL: func @bit_and
@import_global
def bit_and():
  # CHECK: basicpy.binary_expr {{.*}} "BitAnd"
  return 4 & 2

# CHECK-LABEL: func @bit_xor
@import_global
def bit_xor():
  # CHECK: basicpy.binary_expr {{.*}} "BitXor"
  return 4 ^ 2

# CHECK-LABEL: func @bit_or
@import_global
def bit_or():
  # CHECK: basicpy.binary_expr {{.*}} "BitOr"
  return 4 | 2
