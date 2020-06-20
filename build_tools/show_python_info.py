#!/usr/bin/python3
# Simple script to run and verify python path and ability to find
# modules. Run/debug this from your terminal or IDE to make sure things
# are setup correctly.

import sys
print("PYTHONPATH =", sys.path)
print("SYS VERSION =", sys.version)
print("PYTHON EXE =", sys.executable)

try:
    import npcomp
    print("Loaded npcomp module")
except ImportError:
    print("ERROR: Could not load the npcomp module.",
          "Ensure that build/python is on your PYTHONPATH")

try:
    import _npcomp
    print("Loaded (native) _npcomp module")
except ImportError:
    print("ERROR: Could not load the _npcomp native module.",
          "Ensure that build/python_native is on your PYTHONPATH")
