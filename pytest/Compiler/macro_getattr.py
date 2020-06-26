# RUN: %PYTHON %s | npcomp-opt -split-input-file | FileCheck %s --dump-input=fail

import collections
import math
from npcomp.compiler.frontend import *


def import_global(f):
  fe = ImportFrontend()
  fe.import_global_function(f)
  print("// -----")
  print(fe.ir_module.to_asm())
  return f


# CHECK-LABEL: func @module_constant
@import_global
def module_constant():
  # CHECK: constant 3.1415926535897931 : f64
  return math.pi


Sub = collections.namedtuple("Sub", "term")
Record = collections.namedtuple("Record", "fielda,fieldb,inner")
record = Record(5, 25, Sub(6))


# CHECK-LABEL: func @namedtuple_attributes
@import_global
def namedtuple_attributes():
  # CHECK: constant 6
  # CHECK: constant 25
  return record.inner.term - record.fieldb
