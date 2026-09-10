# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""Seed op test suite for the ONNX-native e2e framework.

Importing this package triggers the side-effecting registrations in each
sub-module so that ``GLOBAL_ONNX_TEST_REGISTRY`` is populated before the
runner queries it.
"""


def register_all_tests():
    """Register all built-in ONNX e2e tests."""
    # Imported only to run their @register_onnx_test side effects.
    from . import elementwise  # noqa: F401
    from . import linear  # noqa: F401
    from . import recurrent  # noqa: F401
    from . import signal  # noqa: F401
    from . import reduction  # noqa: F401
