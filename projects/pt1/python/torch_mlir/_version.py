# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from packaging import version
import torch


def torch_version_for_comparison():
    # Ignore +cpu, +cu117m, etc. in comparisons
    return version.parse(torch.__version__.split("+", 1)[0])
