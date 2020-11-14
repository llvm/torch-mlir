#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os


def get_translate_exe():
  search_names = ["iree-translate", "iree-translate.exe"]
  resources_dir = os.path.join(os.path.dirname(__file__))
  for search_name in search_names:
    exe = os.path.join(resources_dir, search_name)
    if os.path.exists(exe):
      return exe
  raise RuntimeError(f"Could not find iree-translate at path: {resources_dir} "
                     f"(is it installed?)")
