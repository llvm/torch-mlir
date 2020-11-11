# RUN: %PYTHON %s | npcomp-opt -split-input-file | FileCheck %s --dump-input=fail

import collections
import math
from npcomp.compiler.numpy import test_config

import_global = test_config.create_import_dump_decorator()


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
