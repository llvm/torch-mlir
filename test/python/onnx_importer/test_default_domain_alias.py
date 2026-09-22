# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s

"""Tests that "" and "ai.onnx" both denote the default ONNX domain."""

import unittest

import onnx
from onnx import TensorProto, helper

from _torch_mlir_config import configure_context, ir, onnx_importer


def _import_model(model: onnx.ModelProto) -> str:
    context = ir.Context()
    configure_context(context)
    model_info = onnx_importer.ModelInfo(model)
    module = model_info.create_module(context=context).operation
    onnx_importer.NodeImporter.define_function(
        model_info.main_graph, module
    ).import_all()
    return module.get_asm()


def _make_relu_model(node_domain: str, opset_domain: str) -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    output_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
    node = helper.make_node("Relu", ["X"], ["Y"], domain=node_domain)
    graph = helper.make_graph([node], "relu", [input_info], [output_info])
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid(opset_domain, 18)]
    )


class DefaultDomainAliasTest(unittest.TestCase):
    def test_default_domain_spellings_are_interchangeable(self):
        for node_domain, opset_domain in (
            ("", ""),
            ("", "ai.onnx"),
            ("ai.onnx", ""),
            ("ai.onnx", "ai.onnx"),
        ):
            with self.subTest(node_domain=node_domain, opset_domain=opset_domain):
                asm = _import_model(_make_relu_model(node_domain, opset_domain))
                self.assertIn("torch.onnx_meta.opset_version = 18 : si64", asm)
                self.assertNotIn("torch.onnx_meta.opset_versions", asm)
                self.assertIn('torch.operator "onnx.Relu"', asm)

    def test_ai_onnx_function_is_expanded(self):
        input_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 4, 4])
        output_info = helper.make_tensor_value_info(
            "Y", TensorProto.FLOAT, [1, 3, 4, 4]
        )
        node = helper.make_node(
            "MeanVarianceNormalization",
            ["X"],
            ["Y"],
            domain="ai.onnx",
        )
        graph = helper.make_graph([node], "mvn", [input_info], [output_info])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

        asm = _import_model(model)
        self.assertNotIn('torch.operator "onnx.MeanVarianceNormalization"', asm)
        self.assertIn('torch.operator "onnx.ReduceMean"', asm)

    def test_custom_domain_is_unchanged(self):
        input_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        output_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
        node = helper.make_node("CustomIdentity", ["X"], ["Y"], domain="test.custom")
        graph = helper.make_graph([node], "custom", [input_info], [output_info])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("test.custom", 1)]
        )

        asm = _import_model(model)
        self.assertIn("torch.onnx_meta.opset_versions", asm)
        self.assertIn("test.custom = 1 : si64", asm)
        self.assertIn('torch.operator "onnx.CustomIdentity"', asm)


if __name__ == "__main__":
    unittest.main()
