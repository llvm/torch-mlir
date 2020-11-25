# RUN: %PYTHON %s

import os
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "1"

import traceback


def run_doctest(mod):
  print("\n\nTESTING:", mod)
  print("--------")
  import doctest
  import sys
  import importlib
  try:
    m = importlib.import_module(mod)
  except:
    print("ERROR IMPORTING MODULE:", mod)
    sys.exit(1)
  fc, _ = doctest.testmod(m)
  if fc:
    sys.exit(1)


TEST_MODULES = (
    "npcomp.compiler.numpy.py_value_utils",
    "npcomp.dialect.Basicpy",
    "npcomp.dialect.Numpy",
    "npcomp.meta.types",
    "npcomp.tracing.context",
    "npcomp.tracing.emitters",
    "npcomp.tracing.exporter",
    "npcomp.tracing.mlir_trace",
)

for mname in TEST_MODULES:
  run_doctest(mname)
