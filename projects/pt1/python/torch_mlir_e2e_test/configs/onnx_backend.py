# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from pathlib import Path
from typing import Any

import io
import onnx
import torch
from torch.onnx._constants import ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET as max_opset_ver
import torch_mlir

from torch_mlir_e2e_test.framework import TestConfig, Trace, TraceItem
from torch_mlir_e2e_test.utils import convert_annotations_to_placeholders
from .utils import (
    recursively_convert_to_numpy,
    recursively_convert_from_numpy,
)

from torch_mlir.extras import onnx_importer
from torch_mlir.dialects import torch as torch_d
from torch_mlir.ir import Context, Module
from torch_mlir.compiler_utils import (
    OutputType,
    run_pipeline_with_repro_report,
    lower_mlir_module,
)

# The pipeline of func.func passes that lower the ONNX backend contract to the
# Linalg-on-Tensors backend contract accepted by RefBackend or another user
# defined backend.
ONNX_TO_TORCH_FUNC_PIPELINE = ",".join(
    [
        "convert-torch-onnx-to-torch",
    ]
)


def import_onnx(contents):
    # Import the ONNX model proto from the file contents:
    raw_model = onnx.load_from_string(contents)
    # since it does not affect current e2e tests, data_prop is left false here
    model_proto = onnx.shape_inference.infer_shapes(raw_model, data_prop=True)

    # Import the ONNX module into an MLIR module:
    context = Context()
    torch_d.register_dialect(context)
    model_info = onnx_importer.ModelInfo(model_proto)
    m = model_info.create_module(context=context)
    imp = onnx_importer.NodeImporter.define_function(model_info.main_graph, m.operation)
    imp.import_all()
    return m


def convert_onnx(model, inputs):
    buffer = io.BytesIO()

    # Process the type information so we export with the dynamic shape information
    examples = []
    input_names = []
    dynamic_tensors = {}
    for index, arg in enumerate(inputs):
        shape = map(lambda d: d if d >= 0 else 1, arg.shape)
        shape = tuple(shape)
        examples.append(torch.zeros(size=shape, dtype=arg.dtype))

        input_name = "input_{}".format(index)
        input_names.append(input_name)

        dynamic_dims = {}
        for dimindex, dim in enumerate(arg.shape):
            if dim < 0:
                dynamic_dims[dimindex] = "dim_{}_{}".format(index, dimindex)

        if dynamic_dims:
            dynamic_tensors[input_name] = dynamic_dims

    examples = tuple(examples)
    torch.onnx.export(
        model,
        examples,
        buffer,
        input_names=input_names,
        dynamic_axes=dynamic_tensors,
        opset_version=max_opset_ver,
    )
    buffer = buffer.getvalue()
    return import_onnx(buffer)


def _module_lowering(
    verbose,
    output_type,
    torch_mod,
):
    if verbose:
        print("\n====================")
        print("ONNX RAW IR")
        print(torch_mod)

    backend_legal_ops = [
        "aten.flatten.using_ints",
        "aten.adaptive_avg_pool1d",
        "aten.unflatten.int",
    ]
    option_string = "{backend-legal-ops=" + ",".join(backend_legal_ops) + "}"

    # Lower from ONNX to Torch
    run_pipeline_with_repro_report(
        torch_mod,
        f"builtin.module(torch-onnx-to-torch-backend-pipeline{option_string})",
        "Lowering Onnx Raw IR -> Torch Backend IR",
    )

    if verbose:
        print("\n====================")
        print("Torch IR")
        print(torch_mod)

    return lower_mlir_module(verbose, output_type, torch_mod)


class OnnxBackendTestConfig(TestConfig):
    """Base class for TestConfig's that are implemented with ONNX.

    This class handles all the common lowering that torch-mlir does before
    reaching the ONNX abstraction level.
    """

    def __init__(
        self,
        backend,
        use_make_fx: bool = False,
        output_type="linalg-on-tensors",
    ):
        super().__init__()
        self.backend = backend
        self.use_make_fx = use_make_fx
        self.output_type = output_type

    def compile(self, program: torch.nn.Module, verbose: bool = False) -> Any:
        example_args = convert_annotations_to_placeholders(program.forward)
        onnx_module = convert_onnx(program, example_args)
        backend_module = _module_lowering(
            verbose, OutputType.get(self.output_type), onnx_module
        )
        compiled_module = self.backend.compile(backend_module)
        return compiled_module

    def run(self, artifact: Any, trace: Trace) -> Trace:
        backend_module = self.backend.load(artifact)
        result: Trace = []
        for item in trace:
            numpy_inputs = recursively_convert_to_numpy(item.inputs)
            outputs = getattr(backend_module, "main_graph")(*numpy_inputs)
            output = recursively_convert_from_numpy(outputs)
            result.append(
                TraceItem(symbol=item.symbol, inputs=item.inputs, output=output)
            )
        return result
