# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s

"""Regression for NodeImporter: ONNX input name '' means omitted optional.

The importer must not conflate that with _nv_map[\"\"] when an earlier node binds
a real tensor to the empty-string output name (see onnx_importer empty-string
collision fix).
"""

import unittest

import onnx
from onnx import TensorProto, helper

from _torch_mlir_config import configure_context, ir, onnx_importer


def _minimal_collision_model() -> onnx.ModelProto:
    """Identity writes to output \"\"; second node lists '', '' as omitted inputs."""
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    n1 = helper.make_node("Identity", ["x"], [""])
    n2 = helper.make_node(
        "ReproEmptyStringCollision",
        ["x", "", ""],
        ["y"],
        domain="zmc.repro",
    )
    graph = helper.make_graph([n1, n2], "g", [inp], [out])
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 21),
            helper.make_opsetid("zmc.repro", 1),
        ],
    )


class EmptyStringOptionalInputsTest(unittest.TestCase):
    def test_optional_slots_use_constant_none_not_prior_tensor(self):
        model = _minimal_collision_model()
        ctx = ir.Context()
        configure_context(ctx)
        mi = onnx_importer.ModelInfo(model)
        m = mi.create_module(context=ctx).operation
        onnx_importer.NodeImporter.define_function(mi.main_graph, m).import_all()
        asm = m.get_asm()
        lines = [ln.strip() for ln in asm.splitlines() if "ReproEmptyStringCollision" in ln]
        self.assertEqual(
            len(lines),
            1,
            msg="expected exactly one onnx.ReproEmptyStringCollision operator line",
        )
        line = lines[0]
        # Correct: trailing optionals are torch.constant.none uses (printed as %none, %none).
        self.assertGreaterEqual(
            line.count("%none"),
            2,
            msg=f"expected at least two %none operands for omitted inputs, got:\n{line}",
        )


if __name__ == "__main__":
    unittest.main()
