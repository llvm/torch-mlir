# RUN: %PYTHON %s


def run_doctest(mod):
  print("TESTING:", mod)
  import doctest
  import sys
  import importlib
  m = importlib.import_module(mod)
  fc, _ = doctest.testmod(m)
  if fc:
    sys.exit(1)


run_doctest("npcomp.compiler.py_value_utils")
