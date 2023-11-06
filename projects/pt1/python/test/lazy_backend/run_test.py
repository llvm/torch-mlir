# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: true


def run_test(*args, XPASS=False, XFAIL=False):
    def _run_test(test):
        test_name = test.__name__
        try:
            test()
            print(("X" if XPASS else "") + f"PASS - {test_name}")
        except Exception as e:
            print(("X" if XFAIL else "") + f"FAIL - {test_name}")
            print("Errors: ", e)
        print(flush=True)

    if len(args):
        _run_test(args[0])
    else:
        return _run_test
