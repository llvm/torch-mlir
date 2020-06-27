# RUN: %PYTHON %s | npcomp-opt -split-input-file | FileCheck %s --dump-input=fail

import math
from npcomp.compiler.frontend import *


def import_global(f):
  fe = ImportFrontend()
  fe.import_global_function(f)
  print("// -----")
  print(fe.ir_module.to_asm())
  return f


# CHECK-LABEL: func @call_ceil_positional
@import_global
def call_ceil_positional(n):
  # CHECK: basicpy.func_template_call @__global$math.ceil(%arg0) kw [] : (!basicpy.UnknownType) -> !basicpy.UnknownType
  return math.ceil(n)


# CHECK-LABEL: func @call_isclose_kw
@import_global
def call_isclose_kw(n):
  # CHECK-DAG: %[[RTOL:.*]] = constant 2.000000e-06
  # CHECK-DAG: %[[ABSTOL:.*]] = constant 2.000000e-01
  # CHECK: basicpy.func_template_call @__global$math.isclose(%arg0, %[[RTOL]], %[[ABSTOL]]) kw ["rtol", "abs_tol"] : (!basicpy.UnknownType, f64, f64) -> !basicpy.UnknownType
  return math.isclose(n, rtol=2e-6, abs_tol=0.2)
