#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import contextlib
import io
import os
import subprocess
import sys

_disable_var = "NPCOMP_DISABLE_FILECHECK"
_filecheck_binary_var = "FILECHECK_BINARY"
_redirect_io = None
_redirect_context = None

def is_filecheck_disabled():
  return _disable_var in os.environ


def start_filecheck_test():
  if is_filecheck_disabled():
    print("WARNING:FileCheck disabled due to", _disable_var, 
        "in the environment", file=sys.stderr)
    return
  global _redirect_io
  global _redirect_context
  _redirect_io = io.StringIO()
  _redirect_context = contextlib.redirect_stdout(_redirect_io)
  _redirect_context.__enter__()


def end_filecheck_test(main_file):
  if is_filecheck_disabled(): return
  global _redirect_io
  global _redirect_context
  _redirect_context.__exit__(None, None, None)
  _redirect_context = None
  _redirect_io.flush()
  filecheck_input = _redirect_io.getvalue()
  _redirect_io = None
  filecheck_binary = "FileCheck"
  if _filecheck_binary_var in os.environ:
    filecheck_binary = os.environ[_filecheck_binary_var]
  print("Using FileCheck binary", filecheck_binary, 
        "(customize by setting", _filecheck_binary_var, ")", file=sys.stderr)
  filecheck_args = [filecheck_binary, main_file, "--dump-input=fail"]
  print("LAUNCHING FILECHECK:", filecheck_args, file=sys.stderr)
  p = subprocess.Popen(filecheck_args, stdin=subprocess.PIPE)
  p.communicate(filecheck_input.encode("UTF-8"))
  sys.exit(p.returncode)  


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
  if disable_filecheck or is_filecheck_disabled():
    print("WARNING:FileCheck disabled due to", _disable_var, 
        "in the environment", file=sys.stderr)
    callback()
    sys.exit(0)

  try:
    start_filecheck_test()
    callback()
  finally:
    end_filecheck_test(main_file)
