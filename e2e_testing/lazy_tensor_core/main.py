# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import ltc_backend.ltc_backend._EXAMPLE_MLIR_BACKEND as ltc_backend

import unittest

# Example models
import ltc_backend_bert
import ltc_backend_mnist
import os
import pathlib


class LTCTests(unittest.TestCase):
    def run_test(self, run_model, mlir_path):
        run_model()

        # Compare the generated MLIR with a known good output.
        with open(os.path.join(pathlib.Path(__file__).parent.resolve(), mlir_path), 'r') as file:
            self.assertEqual(ltc_backend.get_latest_computation().to_string(), file.read())

    def test_bert(self):
        self.run_test(ltc_backend_bert.main, 'bert.mlir')

    def test_mnist(self):
        self.run_test(ltc_backend_mnist.main, 'mnist.mlir')


if __name__ == '__main__':
    ltc_backend._initialize()

    unittest.main()
