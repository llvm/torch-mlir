# RUN: %PYTHON %s 2>&1

#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from npcomp import build

assert build.get_include_dirs()
assert build.get_lib_dirs()
print("CAPI Path:", build.get_capi_link_library_path())
