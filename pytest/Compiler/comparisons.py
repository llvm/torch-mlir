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

# @import_global
# def short_circuit():
#   x = 1
#   y = 2
#   z = 3
#   return x < y < z

