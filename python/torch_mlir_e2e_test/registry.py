# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from typing import Callable

import torch

from .framework import Test

# The global registry of tests.
GLOBAL_TEST_REGISTRY = []
# Ensure that there are no duplicate names in the global test registry.
_SEEN_UNIQUE_NAMES = set()


def register_test_case(module_factory: Callable[[], torch.nn.Module]):
    """Convenient decorator-based test registration.

    Adds a `framework.Test` to the global test registry based on the decorated
    function. The test's `unique_name` is taken from the function name, the
    test's `program_factory` is taken from `module_factory`, and the
    `program_invoker` is the decorated function.
    """
    def decorator(f):
        # Ensure that there are no duplicate names in the global test registry.
        if f.__name__ in _SEEN_UNIQUE_NAMES:
            raise Exception(
                f"Duplicate test name: '{f.__name__}'. Please make sure that the function wrapped by `register_test_case` has a unique name.")
        _SEEN_UNIQUE_NAMES.add(f.__name__)

        # Store the test in the registry.
        GLOBAL_TEST_REGISTRY.append(
            Test(unique_name=f.__name__,
                 program_factory=module_factory,
                 program_invoker=f))
        return f

    return decorator
