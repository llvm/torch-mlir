# RUN: %PYTHON %s | npcomp-opt -split-input-file | FileCheck %s --dump-input=fail

from npcomp.compiler.frontend import *


def import_global(f):
  fe = ImportFrontend()
  fe.import_global_function(f)
  print("// -----")
  print(fe.ir_module.to_asm())
  return f


# CHECK-LABEL: func @binary_lt_
@import_global
def binary_lt_():
  # CHECK: %[[A:.*]] = constant 1 : i64
  # CHECK: %[[B:.*]] = constant 2 : i64
  x = 1
  y = 2
  # CHECK: {{.*}} = basicpy.binary_compare %[[A]] "Lt" %[[B]] : i64, i64
  return x < y


# CHECK-LABEL: func @binary_gt_
@import_global
def binary_gt_():
  x = 1
  y = 2
  # CHECK: {{.*}} = basicpy.binary_compare {{.*}} "Gt" {{.*}} : i64, i64
  return x > y


# CHECK-LABEL: func @binary_lte_
@import_global
def binary_lte_():
  x = 1
  y = 2
  # CHECK: {{.*}} = basicpy.binary_compare {{.*}} "LtE" {{.*}} : i64, i64
  return x <= y


# CHECK-LABEL: func @binary_gte_
@import_global
def binary_gte_():
  x = 1
  y = 2
  # CHECK: {{.*}} = basicpy.binary_compare {{.*}} "GtE" {{.*}} : i64, i64
  return x >= y


# CHECK-LABEL: func @binary_eq_
@import_global
def binary_eq_():
  x = 1
  y = 2
  # CHECK: {{.*}} = basicpy.binary_compare {{.*}} "Eq" {{.*}} : i64, i64
  return x == y


# CHECK-LABEL: func @binary_neq_
@import_global
def binary_neq_():
  x = 1
  y = 2
  # CHECK: {{.*}} = basicpy.binary_compare {{.*}} "NotEq" {{.*}} : i64, i64
  return x != y


# CHECK-LABEL: func @binary_is_
@import_global
def binary_is_():
  x = 1
  y = 2
  # CHECK: {{.*}} = basicpy.binary_compare {{.*}} "Is" {{.*}} : i64, i64
  return x is y


# CHECK-LABEL: func @binary_is_not_
@import_global
def binary_is_not_():
  x = 1
  y = 2
  # CHECK: {{.*}} = basicpy.binary_compare {{.*}} "IsNot" {{.*}} : i64, i64
  return x is not y


# CHECK-LABEL: func @binary_in_
@import_global
def binary_in_():
  x = 1
  y = 2
  # CHECK: {{.*}} = basicpy.binary_compare {{.*}} "In" {{.*}} : i64, i64
  return x in y


# CHECK-LABEL: func @binary_not_in_
@import_global
def binary_not_in_():
  x = 1
  y = 2
  # CHECK: {{.*}} = basicpy.binary_compare {{.*}} "NotIn" {{.*}} : i64, i64
  return x not in y


@import_global
def short_circuit():
  # CHECK: %[[X:.*]] = constant 1 : i64
  # CHECK: %[[Y:.*]] = constant 2 : i64
  # CHECK: %[[Z:.*]] = constant 3 : i64
  # CHECK: %[[OMEGA:.*]] = constant 5 : i64
  x = 1
  y = 2
  z = 3
  omega = 5
  # CHECK: %[[FALSE:.*]] = basicpy.bool_constant false
  # CHECK: %[[CMP0:.*]] = basicpy.binary_compare %[[X]] "Lt" %[[Y]]
  # CHECK: %[[CMP0_CAST:.*]] = basicpy.bool_cast %[[CMP0]] : !basicpy.BoolType -> i1
  # CHECK: %[[IF0:.*]] = scf.if %[[CMP0_CAST]] -> (!basicpy.BoolType) {
  # CHECK:   %[[CMP1:.*]] = basicpy.binary_compare %[[Y]] "Eq" %[[Z]]
  # CHECK:   %[[CMP1_CAST:.*]] = basicpy.bool_cast %[[CMP1]] : !basicpy.BoolType -> i1
  # CHECK:   %[[IF1:.*]] = scf.if %[[CMP1_CAST]] {{.*}} {
  # CHECK:     %[[CMP2:.*]] = basicpy.binary_compare %[[Z]] "GtE" %[[OMEGA]]
  # CHECK:     scf.yield %[[CMP2]]
  # CHECK:   } else {
  # CHECK:     scf.yield %[[FALSE]]
  # CHECK:   }
  # CHECK:   scf.yield %[[IF1]]
  # CHECK: } else {
  # CHECK:   scf.yield %[[FALSE]]
  # CHECK: }
  # CHECK: %[[RESULT:.*]] = basicpy.unknown_cast %[[IF0]]
  # CHECK: return %[[RESULT]]
  return x < y == z >= omega


# CHECK-LABEL: nested_short_circuit_expression
@import_global
def nested_short_circuit_expression():
  x = 1
  y = 2
  z = 3
  # Verify that the (z + 5) gets nested into the if.
  # CHECK: scf.if {{.*}} {
  # CHECK-NEXT: constant 6
  # CHECK-NEXT: binary_expr
  return x < y == (z + 6)
