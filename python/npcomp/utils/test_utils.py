#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import contextlib
import io
import os
import subprocess
import sys

def run_under_filecheck(main_file, callback, disable_filecheck=False):
  """Runs a callback under a FileCheck sub-process.

  This is typically called from a main context and will sys.exit on
  completion.
  
  Args:
    main_file: The file to process filecheck directives on. Typically
      __file__ from the caller's perspective.
    callback: The no-argument callback to invoke.
    disable_filecheck: Whether to disable filecheck.
  """
  disable_var = "NPCOMP_DISABLE_FILECHECK"
  filecheck_binary_var = "FILECHECK_BINARY"
  if "NPCOMP_DISABLE_FILECHECK" in os.environ:
    print("WARNING:FileCheck disabled due to", disable_var, 
        "in the environment", file=sys.stderr)
    disable_filecheck = True
  if disable_filecheck:
    callback()
    sys.exit(0)

  # Redirect through FileCheck
  filecheck_capture_io = io.StringIO()
  with contextlib.redirect_stdout(filecheck_capture_io):
    callback()
  filecheck_capture_io.flush()
  filecheck_input = filecheck_capture_io.getvalue()
  filecheck_binary = "FileCheck"
  if filecheck_binary_var in os.environ:
    filecheck_binary = os.environ[filecheck_binary_var]
  print("Using FileCheck binary", filecheck_binary, 
        "(customize by setting", filecheck_binary_var, ")", file=sys.stderr)
  filecheck_args = [filecheck_binary, main_file, "--dump-input=fail"]
  print("LAUNCHING FILECHECK:", filecheck_args, file=sys.stderr)
  p = subprocess.Popen(filecheck_args, stdin=subprocess.PIPE)
  p.communicate(filecheck_input.encode("UTF-8"))
  sys.exit(p.returncode)  
