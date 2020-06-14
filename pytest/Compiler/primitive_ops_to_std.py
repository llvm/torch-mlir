# RUN: %PYTHON %s | npcomp-opt -split-input-file -basicpy-type-inference -convert-basicpy-to-std | FileCheck %s --dump-input=fail

from npcomp.compiler.frontend import *


def import_global(f):
  fe = ImportFrontend()
  fe.import_global_function(f)
  print("// -----")
  print(fe.ir_module.to_asm())
  return f


################################################################################
# Integer tests
################################################################################


# CHECK-LABEL: func @int_add
@import_global
def int_add(a: int, b: int):
  # CHECK: %0 = addi %arg0, %arg1 : i64
  return a + b


# CHECK-LABEL: func @int_sub
@import_global
def int_sub(a: int, b: int):
  # CHECK: %0 = subi %arg0, %arg1 : i64
  return a - b


# CHECK-LABEL: func @int_mult
@import_global
def int_mult(a: int, b: int):
  # CHECK: %0 = muli %arg0, %arg1 : i64
  return a * b


# CHECK-LABEL: func @int_floordiv
@import_global
def int_floordiv(a: int, b: int):
  # CHECK: %0 = divi_signed %arg0, %arg1 : i64
  return a // b


# CHECK-LABEL: func @int_modulo
@import_global
def int_modulo(a: int, b: int):
  # CHECK: %0 = remi_signed %arg0, %arg1 : i64
  return a % b


# CHECK-LABEL: func @int_left_shift
@import_global
def int_left_shift(a: int, b: int):
  # CHECK: %0 = shift_left %arg0, %arg1 : i64
  return a << b


# CHECK-LABEL: func @int_right_shift
@import_global
def int_right_shift(a: int, b: int):
  # CHECK: %0 = shift_right_signed %arg0, %arg1 : i64
  return a >> b


# CHECK-LABEL: func @int_and
@import_global
def int_and(a: int, b: int):
  # CHECK: %0 = and %arg0, %arg1 : i64
  return a & b


# CHECK-LABEL: func @int_xor
@import_global
def int_xor(a: int, b: int):
  # CHECK: %0 = xor %arg0, %arg1 : i64
  return a ^ b


# CHECK-LABEL: func @int_or
@import_global
def int_or(a: int, b: int):
  # CHECK: %0 = or %arg0, %arg1 : i64
  return a | b


################################################################################
# Floating point
################################################################################


# CHECK-LABEL: func @float_add
@import_global
def float_add(a: float, b: float):
  # CHECK: %0 = addf %arg0, %arg1 : f64
  return a + b


# CHECK-LABEL: func @float_sub
@import_global
def float_sub(a: float, b: float):
  # CHECK: %0 = subf %arg0, %arg1 : f64
  return a - b


# CHECK-LABEL: func @float_mult
@import_global
def float_mult(a: float, b: float):
  # CHECK: %0 = mulf %arg0, %arg1 : f64
  return a * b


# CHECK-LABEL: func @float_floordiv
@import_global
def float_floordiv(a: float, b: float):
  # CHECK: %0 = divf %arg0, %arg1 : f64
  return a / b


################################################################################
# Bool conversions
################################################################################


# CHECK-LABEL: func @to_boolean
@import_global
def to_boolean(x: int):
  # CHECK: %[[ZERO:.*]] = constant 0 : i64
  # CHECK: %[[BOOL:.*]] = cmpi "ne", %arg0, %[[ZERO]] : i64
  # CHECK: select %[[BOOL]]
  # Note that the not operator is just used to force a bool conversion.
  return not x


# CHECK-LABEL: func @to_boolean_float
@import_global
def to_boolean_float(x: float):
  # CHECK: %[[ZERO:.*]] = constant 0.000000e+00 : f64
  # CHECK: %[[BOOL:.*]] = cmpf "one", %arg0, %[[ZERO]] : f64
  # CHECK: select %[[BOOL]]
  # Note that the not operator is just used to force a bool conversion.
  return not x


################################################################################
# Integer comparisons
################################################################################


# CHECK-LABEL: func @int_lt_
@import_global
def int_lt_(x: int, y: int):
  # CHECK: %[[CMP:.*]] = cmpi "slt", %arg0, %arg1 : i64
  # CHECK: %{{.*}} = basicpy.bool_cast %[[CMP]] : i1 -> !basicpy.BoolType
  return x < y


# CHECK-LABEL: func @int_gt_
@import_global
def int_gt_(x: int, y: int):
  # CHECK: cmpi "sgt"
  return x > y


# CHECK-LABEL: func @int_lte_
@import_global
def int_lte_(x: int, y: int):
  # CHECK: cmpi "sle"
  return x <= y


# CHECK-LABEL: func @int_gte_
@import_global
def int_gte_(x: int, y: int):
  # CHECK: cmpi "sge"
  return x >= y


# CHECK-LABEL: func @int_eq_
@import_global
def int_eq_(x: int, y: int):
  # CHECK: cmpi "eq"
  return x == y


# CHECK-LABEL: func @int_neq_
@import_global
def int_neq_(x: int, y: int):
  # CHECK: cmpi "ne"
  return x != y


# CHECK-LABEL: func @int_is_
@import_global
def int_is_(x: int, y: int):
  # CHECK: cmpi "eq"
  return x is y


# CHECK-LABEL: func @int_is_not_
@import_global
def int_is_not_(x: int, y: int):
  # CHECK: cmpi "ne"
  return x is not y


################################################################################
# Float comparisons
################################################################################


# CHECK-LABEL: func @float_lt_
@import_global
def float_lt_(x: float, y: float):
  # CHECK: %[[CMP:.*]] = cmpf "olt", %arg0, %arg1 : f64
  # CHECK: %{{.*}} = basicpy.bool_cast %[[CMP]] : i1 -> !basicpy.BoolType
  return x < y


# CHECK-LABEL: func @float_gt_
@import_global
def float_gt_(x: float, y: float):
  # CHECK: cmpf "ogt"
  return x > y


# CHECK-LABEL: func @float_lte_
@import_global
def float_lte_(x: float, y: float):
  # CHECK: cmpf "ole"
  return x <= y


# CHECK-LABEL: func @float_gte_
@import_global
def float_gte_(x: float, y: float):
  # CHECK: cmpf "oge"
  return x >= y


# CHECK-LABEL: func @float_eq_
@import_global
def float_eq_(x: float, y: float):
  # CHECK: cmpf "oeq"
  return x == y


# CHECK-LABEL: func @float_neq_
@import_global
def float_neq_(x: float, y: float):
  # CHECK: cmpf "one"
  return x != y


# CHECK-LABEL: func @float_is_
@import_global
def float_is_(x: float, y: float):
  # CHECK: cmpf "oeq"
  return x is y


# CHECK-LABEL: func @float_is_not_
@import_global
def float_is_not_(x: float, y: float):
  # CHECK: cmpf "one"
  return x is not y
