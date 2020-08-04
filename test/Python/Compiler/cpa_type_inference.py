# RUN: %PYTHON %s | npcomp-opt -split-input-file -npcomp-cpa-type-inference | FileCheck %s --dump-input=fail

from npcomp.compiler import test_config

import_global = test_config.create_import_dump_decorator()


# CHECK-LABEL: func @arithmetic_expression
@import_global
def arithmetic_expression():
  return 1 + 2 - 3 * 4


# CHECK-LABEL: func @arg_inference
@import_global
def arg_inference(a: int, b: int):
  return a + 2 * b


# CHECK-LABEL: func @conditional_inference
@import_global
def conditional_inference(cond: int, a: bool, b: int):
  return a if cond + 1 else not (b * 4)
