#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exports configuration for the package and settings for building libraries."""

import os
import platform

from mlir._mlir_libs import get_include_dirs, get_lib_dirs

__all__ = [
  "get_include_dirs",
  "get_lib_dirs",
]


def get_capi_link_library_name() -> str:
  """Gets the library name of the CAPI shared library to link against."""
  return "NPCOMPPythonCAPI"


def get_capi_link_library_path() -> str:
  """Returns an absolute path to the CAPI shared library.

  This should be preferred when seeking to create a non relocatable package
  as it eliminates the possibility of interference of similar named libraries
  on the link path.

  Raises:
    ValueError: If the library cannot be found.
  """
  system = platform.system()
  lib_prefix = "lib"
  lib_suffix = ".so"
  if system == "Darwin":
    lib_suffix = ".dylib"
  elif system == "Windows":
    lib_prefix = ""
    lib_suffix = ".lib"
  lib_filename = f"{lib_prefix}{get_capi_link_library_name()}{lib_suffix}"

  for lib_dir in get_lib_dirs():
    full_path = os.path.join(lib_dir, lib_filename)
    if os.path.exists(full_path): return full_path

  raise ValueError(
    f"Unable to find npcomp development library {lib_filename} in "
    f"{get_lib_dirs()}")

