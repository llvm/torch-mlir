# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s

import gc
import io
import re
import unittest
from unittest.mock import patch

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.passes.shape_prop import _extract_tensor_metadata

from torch_mlir import ir
from torch_mlir.dialects import torch as torch_dialect
from torch_mlir.extras.fx_importer import FxImporter


def import_literals(tensors, *, fake_mode=False):
    root = torch.nn.Module()
    graph = torch.fx.Graph()
    outputs = []
    for index, tensor in enumerate(tensors):
        name = f"value{index}"
        root.register_buffer(name, tensor)
        node = graph.get_attr(name)
        node.meta["tensor_meta"] = _extract_tensor_metadata(tensor)
        outputs.append(node)
    graph.output(tuple(outputs))
    module = torch.fx.GraphModule(root, graph)
    with ir.Context() as context:
        torch_dialect.register_dialect(context)
        importer = FxImporter(context=context)
        if fake_mode:
            with FakeTensorMode():
                importer.import_graph_module(module)
                assert isinstance(torch.empty(2), FakeTensor)
        else:
            importer.import_graph_module(module)
        assert importer.module.operation.verify()
        return importer.module


def resource_blobs(module):
    return [
        bytes.fromhex(blob) for blob in re.findall(r': "0x([0-9A-Fa-f]+)"', str(module))
    ]


def resource_bytes(module):
    # The serialized resource starts with a four-byte alignment header.
    return [blob[4:] for blob in resource_blobs(module)]


class TensorLiteralTest(unittest.TestCase):
    def test_exact_dense_bytes_without_tolist(self):
        # Include signed zero and a non-default NaN payload: converting through
        # Python scalars can lose the original encoding.
        fp32 = torch.tensor(
            [0, -2147483648, 0x7FC01234, 0x7F800000], dtype=torch.int32
        ).view(torch.float32)
        tensors = [
            torch.nn.Parameter(fp32.reshape(2, 2).t()),
            torch.arange(6, dtype=torch.int64).reshape(2, 3).t(),
            torch.arange(6, dtype=torch.float64).reshape(2, 3).t(),
            torch.arange(6, dtype=torch.float16).reshape(2, 3).t(),
            torch.arange(6, dtype=torch.bfloat16).reshape(2, 3).t(),
            torch.arange(6, dtype=torch.uint8).reshape(2, 3).t(),
            torch.tensor([1 + 2j, 3 - 4j]).conj(),
            torch._neg_view(torch.tensor([1.0, -2.0])),
        ]
        for name in ("float8_e4m3fn", "float8_e5m2", "float8_e8m0fnu"):
            if hasattr(torch, name):
                tensors.append(
                    torch.arange(256, dtype=torch.uint8)
                    .view(getattr(torch, name))
                    .reshape(16, 16)
                    .t()
                )
        for tensor in tensors:
            with self.subTest(dtype=tensor.dtype, shape=tensor.shape):
                expected = (
                    tensor.detach()
                    .resolve_conj()
                    .resolve_neg()
                    .contiguous()
                    .view(torch.uint8)
                    .numpy()
                    .tobytes()
                )
                with patch.object(
                    torch.Tensor, "tolist", side_effect=AssertionError("tolist used")
                ):
                    module = import_literals([tensor], fake_mode=True)
                self.assertEqual(resource_bytes(module), [expected])
                self.assertEqual(
                    int.from_bytes(resource_blobs(module)[0][:4], "little"),
                    tensor.element_size(),
                )
                literal = module.body.operations[0].regions[0].blocks[0].operations[0]
                self.assertEqual(
                    tuple(ir.ShapedType(literal.attributes["value"].type).shape),
                    tuple(tensor.shape),
                )
                bytecode = io.BytesIO()
                module.operation.write_bytecode(bytecode)
                for serialized in (str(module), bytecode.getvalue()):
                    with ir.Context() as context:
                        torch_dialect.register_dialect(context)
                        reloaded = ir.Module.parse(serialized)
                        self.assertTrue(reloaded.operation.verify())
                        self.assertEqual(
                            resource_blobs(reloaded), resource_blobs(module)
                        )

    def test_shared_resource_and_snapshot_lifetime(self):
        value = torch.tensor([1.0, 2.0])
        other = value.clone()
        expected = value.view(torch.uint8).numpy().tobytes()
        module = import_literals([value, value, other])
        literals = list(module.body.operations[0].regions[0].blocks[0].operations)[:3]
        self.assertEqual(
            literals[0].attributes["value"], literals[1].attributes["value"]
        )
        self.assertNotEqual(
            literals[0].attributes["value"], literals[2].attributes["value"]
        )
        value.fill_(9)
        other.fill_(8)
        del value, other
        gc.collect()
        self.assertEqual(resource_bytes(module), [expected, expected])

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_literal_under_fake_mode(self):
        value = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        tensor = value.t().cuda()
        expected = value.t().contiguous().view(torch.uint8).numpy().tobytes()
        with patch.object(
            torch.Tensor, "tolist", side_effect=AssertionError("tolist used")
        ):
            module = import_literals([tensor], fake_mode=True)
        self.assertEqual(resource_bytes(module), [expected])

    def test_scalar_and_single_element_literals(self):
        for shape in ((), (1,), (1, 1)):
            with self.subTest(shape=shape):
                module = import_literals([torch.tensor(2.0).reshape(shape)])
                literal = module.body.operations[0].regions[0].blocks[0].operations[0]
                self.assertIsInstance(literal.attributes["value"], ir.DenseElementsAttr)
                self.assertEqual(resource_bytes(module), [])

    def test_boolean_import(self):
        module = import_literals([torch.tensor([True, False, True])])
        self.assertIn("!torch.vtensor<[3],i1>", str(module))
        self.assertTrue(module.operation.verify())

    def test_fake_and_meta_keep_existing_conversion(self):
        fake = FakeTensorMode().from_tensor(torch.ones(2, 3))
        meta = torch.empty(2, 3, device="meta")
        for tensor in (fake, meta):
            with self.subTest(device=tensor.device):
                # Do not attempt to read nonexistent storage or fabricate data.
                with patch.object(
                    type(tensor), "tolist", side_effect=RuntimeError("legacy path")
                ), self.assertRaisesRegex(RuntimeError, "legacy path"):
                    import_literals([tensor])


if __name__ == "__main__":
    unittest.main()
