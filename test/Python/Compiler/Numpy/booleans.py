# RUN: %PYTHON %s | npcomp-opt -split-input-file | FileCheck %s --dump-input=fail

from npcomp.compiler.numpy import test_config

import_global = test_config.create_import_dump_decorator()


# CHECK-LABEL: func @logical_and
@import_global
def logical_and():
  # CHECK: %[[X:.*]] = constant 1
  # CHECK: %[[Y:.*]] = constant 0
  # CHECK: %[[Z:.*]] = constant 2
  x = 1
  y = 0
  z = 2
  # CHECK: %[[XBOOL:.*]] = basicpy.as_predicate_value %[[X]]
  # CHECK: %[[IF0:.*]] = scf.if %[[XBOOL]] -> (!basicpy.UnknownType) {
  # CHECK:   %[[YBOOL:.*]] = basicpy.as_predicate_value %[[Y]]
  # CHECK:   %[[IF1:.*]] = scf.if %[[YBOOL]] -> (!basicpy.UnknownType) {
  # CHECK:     %[[ZCAST:.*]] = basicpy.unknown_cast %[[Z]]
  # CHECK:     scf.yield %[[ZCAST]]
  # CHECK:   } else {
  # CHECK:     %[[YCAST:.*]] = basicpy.unknown_cast %[[Y]]
  # CHECK:     scf.yield %[[YCAST]]
  # CHECK:   }
  # CHECK:   %[[IF1CAST:.*]] = basicpy.unknown_cast %[[IF1]]
  # CHECK:   scf.yield %[[IF1CAST]]
  # CHECK: } else {
  # CHECK:   %[[XCAST:.*]] = basicpy.unknown_cast %[[X]]
  # CHECK:   scf.yield %[[XCAST]]
  # CHECK: }
  return x and y and z


# CHECK-LABEL: func @logical_or
@import_global
def logical_or():
  # CHECK: %[[X:.*]] = constant 0
  # CHECK: %[[Y:.*]] = constant 1
  # CHECK: %[[Z:.*]] = constant 2
  # CHECK: %[[XBOOL:.*]] = basicpy.as_predicate_value %[[X]]
  # CHECK: %[[IF0:.*]] = scf.if %[[XBOOL]] -> (!basicpy.UnknownType) {
  # CHECK:   %[[XCAST:.*]] = basicpy.unknown_cast %[[X]]
  # CHECK:   scf.yield %[[XCAST]]
  # CHECK: } else {
  # CHECK:   %[[YBOOL:.*]] = basicpy.as_predicate_value %[[Y]]
  # CHECK:   %[[IF1:.*]] = scf.if %[[YBOOL]] -> (!basicpy.UnknownType) {
  # CHECK:     %[[YCAST:.*]] = basicpy.unknown_cast %[[Y]]
  # CHECK:     scf.yield %[[YCAST]]
  # CHECK:   } else {
  # CHECK:     %[[ZCAST:.*]] = basicpy.unknown_cast %[[Z]]
  # CHECK:     scf.yield %[[ZCAST]]
  # CHECK:   }
  # CHECK:   %[[IF1CAST:.*]] = basicpy.unknown_cast %[[IF1]]
  # CHECK:   scf.yield %[[IF1CAST]]
  # CHECK: }
  x = 0
  y = 1
  z = 2
  return x or y or z


# CHECK-LABEL: func @logical_not
@import_global
def logical_not():
  # CHECK: %[[X:.*]] = constant 1
  x = 1
  # CHECK-DAG: %[[TRUE:.*]] = basicpy.bool_constant true
  # CHECK-DAG: %[[FALSE:.*]] = basicpy.bool_constant false
  # CHECK-DAG: %[[CONDITION:.*]] = basicpy.as_predicate_value %[[X]]
  # CHECK-DAG: %{{.*}} = select %[[CONDITION]], %[[FALSE]], %[[TRUE]] : !basicpy.BoolType
  return not x


# CHECK-LABEL: func @conditional
@import_global
def conditional():
  # CHECK: %[[X:.*]] = constant 1
  x = 1
  # CHECK: %[[CONDITION:.*]] = basicpy.as_predicate_value %[[X]]
  # CHECK: %[[IF0:.*]] = scf.if %[[CONDITION]] -> (!basicpy.UnknownType) {
  # CHECK:   %[[TWO:.*]] = constant 2 : i64
  # CHECK:   %[[TWO_CAST:.*]] = basicpy.unknown_cast %[[TWO]]
  # CHECK:   scf.yield %[[TWO_CAST]]
  # CHECK: } else {
  # CHECK:   %[[THREE:.*]] = constant 3 : i64
  # CHECK:   %[[THREE_CAST:.*]] = basicpy.unknown_cast %[[THREE]]
  # CHECK:   scf.yield %[[THREE_CAST]]
  # CHECK: }
  return 2 if x else 3
