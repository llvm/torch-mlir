# XFAIL: *
# RUN: %PYTHON %s | npcomp-opt -split-input-file | FileCheck %s --dump-input=fail

from npcomp.compiler.frontend import *


def import_global(f):
  fe = ImportFrontend()
  fe.import_global_function(f)
  print("// -----")
  print(fe.ir_module.to_asm())
  return f


OUTER_ONE = 1
OUTER_STRING = "Hello"


# CHECK-LABEL: func @outer_one
@import_global
def outer_one():
  return OUTER_ONE

