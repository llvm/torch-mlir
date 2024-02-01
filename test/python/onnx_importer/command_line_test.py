# Based on code Copyright (c) Advanced Micro Devices, Inc.
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s --output %t

from pathlib import Path

import logging
import shutil
import sys
import subprocess
import unittest
import unittest.mock

import onnx

from torch_mlir.tools.import_onnx import __main__

# For ONNX models

import numpy
from onnx import numpy_helper, TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.external_data_helper import convert_model_to_external_data
from onnx.checker import check_model

# Accept the output path on the command line or default to a sibling
# to this file. We have to pop this off explicitly or else unittest
# won't understand.
if len(sys.argv) > 1 and sys.argv[1] == "--output":
    OUTPUT_PATH = Path(sys.argv[2])
    del sys.argv[1:3]
else:
    OUTPUT_PATH = Path(__file__).resolve().parent / "output"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


def const_model() -> onnx.ModelProto:
    # Note: data_path must be relative to model_file

    const = make_node(
        'Constant', [], ['c_shape'], 'const',
        value=numpy_helper.from_array(numpy.array([4], dtype=numpy.int64)))
    cofshape = make_node(
        'ConstantOfShape', ['c_shape'], ['c_out'], 'cofshape',
        value=numpy_helper.from_array(numpy.array([1], dtype=numpy.int64)))

    outval = make_tensor_value_info('c_out', TensorProto.INT64, [None])
    graph = make_graph([const, cofshape], 'constgraph', [], [outval])

    onnx_model = make_model(graph)
    check_model(onnx_model)
    return onnx_model


def linear_model() -> onnx.ModelProto:
    # initializers
    k_dim = 32
    value = numpy.arange(k_dim).reshape([k_dim, 1])
    value = numpy.asarray(value, dtype=numpy.float32)
    A = numpy_helper.from_array(value, name='A')

    value = numpy.array([0.4], dtype=numpy.float32).reshape([1, 1])
    C = numpy_helper.from_array(value, name='C')

    # the part which does not change
    X = make_tensor_value_info('X', TensorProto.FLOAT, [1, k_dim])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None])
    node1 = make_node('MatMul', ['X', 'A'], ['AX'])
    node2 = make_node('Add', ['AX', 'C'], ['Y'])
    graph = make_graph([node1, node2], 'lr', [X], [Y], [A, C])
    onnx_model = make_model(graph)
    check_model(onnx_model)
    return onnx_model


ALL_MODELS = [
    const_model,
    linear_model
]


class CommandLineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = OUTPUT_PATH / "command-line"
        shutil.rmtree(cls.test_dir, ignore_errors=True)
        cls.test_dir.mkdir(parents=True, exist_ok=True)

    def get_run_path(self, model_name: str) -> Path:
        run_path = CommandLineTest.test_dir / model_name
        run_path.mkdir(exist_ok=True)
        return run_path

    def run_model_intern(self, onnx_model: onnx.ModelProto, model_name: str):
        run_path = self.get_run_path(model_name)
        model_file = run_path / f"{model_name}-i.onnx"
        mlir_file = run_path / f"{model_name}-i.torch.mlir"
        onnx.save(onnx_model, model_file)
        args = __main__.parse_arguments([
            str(model_file), "-o", str(mlir_file)])
        __main__.main(args)

    def run_model_extern(self, onnx_model: onnx.ModelProto, model_name: str):
        run_path = self.get_run_path(model_name)
        model_file = run_path / f"{model_name}-e.onnx"
        mlir_file = run_path / f"{model_name}-e.torch.mlir"
        data_dir_name = f"{model_name}-data"
        model_data_dir = run_path / data_dir_name
        model_data_dir.mkdir(exist_ok=True)
        convert_model_to_external_data(
            onnx_model, all_tensors_to_one_file=True,
            location=data_dir_name + "/data.bin",
            size_threshold=48,
            convert_attribute=True)
        onnx.save(onnx_model, model_file)
        temp_dir = run_path / "temp"
        temp_dir.mkdir(exist_ok=True)
        args = __main__.parse_arguments([
            str(model_file), "-o", str(mlir_file), "--keep-temps", "--temp-dir",
            str(temp_dir), "--data-dir", str(run_path)])
        __main__.main(args)

    def test_all(self):
        for model_func in ALL_MODELS:
            model_name = model_func.__name__
            model = model_func()
            with self.subTest(f"model {model_name}", model_name=model_name):
                with self.subTest("Internal data"):
                    self.run_model_intern(model, model_name)
                with self.subTest("External data"):
                    self.run_model_extern(model, model_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
