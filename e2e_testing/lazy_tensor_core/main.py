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

    def assert_tensors_almost_equal(self, tensor_a, tensor_b, message):
        a, b = tensor_a.cpu().detach().numpy(), tensor_b.cpu().detach().numpy()

        # Ensure tensors match up to 7 decimals of precision.
        assert_almost_equal(a, b, 7, message)

    def run_test(self, run_model):
        model_torch_mlir, loss_torch_mlir = run_model('lazy')
        model_cpu, loss_cpu = run_model('cpu')

        # Check losses match.
        self.assertEqual(len(loss_torch_mlir), len(loss_cpu))
        for idx in range(len(loss_torch_mlir)):
            self.assert_tensors_almost_equal(loss_torch_mlir[idx], loss_cpu[idx],
                                             f'Losses at index {idx} do not match!')

        # Check that number of parameters match.
        torch_mlir_params, cpu_params = [list(model.named_parameters()) for model in (model_torch_mlir, model_cpu)]
        self.assertEqual(len(torch_mlir_params), len(cpu_params))

        # Check that names of parameters match.
        torch_mlir_keys = []
        for name, param in torch_mlir_params:
            torch_mlir_keys.append(name)

        cpu_keys = []
        for name, param in cpu_params:
            cpu_keys.append(name)

        self.assertEqual(torch_mlir_keys, cpu_keys)

        # Check contents of parameters match.
        for idx in range(len(torch_mlir_params)):
            self.assert_tensors_almost_equal(torch_mlir_params[idx][1], cpu_params[idx][1],
                                             f'Parameters {torch_mlir_keys[idx]} do not match!')

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
