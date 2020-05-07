#!/usr/bin/env python3

import os
import subprocess
import sys


TEST_MODULES = (
  "npcomp.mlir_ir_test",
  "npcomp.dialect.Basicpy",
  "npcomp.dialect.Numpy",
  "npcomp.tracing.context",
  "npcomp.tracing.mlir_trace",
  "npcomp.types",
  "npcomp.exporter",
  "npcomp.tracing.mlir_trace_test",
)

# Compute PYTHONPATH for sub processes.
DIRSEP = os.path.pathsep
LOCAL_PYTHONPATH_COMPONENTS = [
  # This directory.
  os.path.abspath(os.path.dirname(__file__)),
  # The parallel python_native directory (assuming in the build tree).
  os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python_native"))
]
PYTHONPATH = DIRSEP.join(LOCAL_PYTHONPATH_COMPONENTS)
if "PYTHONPATH" in os.environ:
  PYTHONPATH = PYTHONPATH + DIRSEP + os.environ["PYTHONPATH"]
CHILD_ENVIRON = dict(os.environ)
CHILD_ENVIRON["PYTHONPATH"] = PYTHONPATH

# Configure filecheck.
FILECHECK_BINARY = os.path.abspath(
    os.path.join(
      os.path.dirname(__file__), 
      "..", "..", "..", "bin", "FileCheck"))
if os.path.exists(FILECHECK_BINARY):
  CHILD_ENVIRON["FILECHECK_BINARY"] = FILECHECK_BINARY
else:
  print("WARNING! Built FileCheck not found. Leaving to path resolution")

passed = []
failed = []

for test_module in TEST_MODULES:
  print("--------====== RUNNING %s ======--------" % test_module)
  try:
    subprocess.check_call([sys.executable, "-Wignore", "-m", test_module], 
                          env=CHILD_ENVIRON)
    print("--------====== DONE %s ======--------\n" % test_module)
    passed.append(test_module)
  except subprocess.CalledProcessError:
    print("!!!!!!!!====== ERROR %s ======!!!!!!!!\n" % test_module)
    failed.append(test_module)
    
print("Done: %d passed, %d failed" % (len(passed), len(failed)))
if failed:
  for test_module in failed:
    print("  %s: FAILED" % test_module)
  sys.exit(1)
