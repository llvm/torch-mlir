from npcomp.compiler.frontend import *


def import_global(f):
  fe = ImportFrontend()
  fe.import_global_function(f)
  print(fe.ir_module.to_asm())
  return f

@import_global
def arithmetic_expression():
  return 1 + 2 - 3 * 4
