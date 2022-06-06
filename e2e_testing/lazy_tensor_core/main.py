# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import os
import pathlib
import unittest

import ltc_backend.ltc_backend._EXAMPLE_MLIR_BACKEND as ltc_backend
from numpy.testing import assert_almost_equal

# Example models
import ltc_backend_bert
import ltc_backend_mnist


class LTCNumericTests(unittest.TestCase):
    """
    This test suite validates numerics by comparing the output of the models when
    executed using the MLIR LTC backend and ensuring they match the results on CPU.
    """

    def assert_tensors_list_almost_equal(self, tensors_a, tensors_b):
        self.assertEqual(len(tensors_a), len(tensors_b))

        for idx in range(len(tensors_a)):
            a = tensors_a[idx].cpu().detach().numpy()
            b = tensors_b[idx].cpu().detach().numpy()

            assert_almost_equal(a, b)

    def run_test(self, run_model):
        model_torch_mlir, loss_torch_mlir = run_model('lazy')
        model_cpu, loss_cpu = run_model('cpu')

        # Ensure that model states and losses are almost equal between LTC and CPU.
        self.assert_tensors_list_almost_equal(loss_torch_mlir, loss_cpu)
        self.assert_tensors_list_almost_equal(list(model_torch_mlir.parameters()), list(model_cpu.parameters()))

    def test_bert(self):
        self.run_test(ltc_backend_bert.main)

    def test_mnist(self):
        self.run_test(ltc_backend_mnist.main)


class LTCMlirTests(unittest.TestCase):
    """
    This test suite validates that the emitted MLIR matches a known good output.
    """

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
