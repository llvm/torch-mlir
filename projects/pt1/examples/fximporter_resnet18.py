# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import sys
from pathlib import Path

import torch
import torch.utils._pytree as pytree
import torchvision.models as models
from torch_mlir import fx
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend
from torch_mlir_e2e_test.configs.utils import (
    recursively_convert_to_numpy,
)

sys.path.append(str(Path(__file__).absolute().parent))
from _example_utils import (
    top3_possibilities,
    load_and_preprocess_image,
    load_labels,
    DEFAULT_IMAGE_URL,
)


print("load image from " + DEFAULT_IMAGE_URL, file=sys.stderr)
img = load_and_preprocess_image(DEFAULT_IMAGE_URL)
labels = load_labels()

resnet18 = models.resnet18(pretrained=True).eval()
module = fx.export_and_import(
    resnet18,
    torch.ones(1, 3, 224, 224),
    output_type="linalg-on-tensors",
    func_name=resnet18.__class__.__name__,
)
backend = refbackend.RefBackendLinalgOnTensorsBackend()
compiled = backend.compile(module)
fx_module = backend.load(compiled)

params = {
    **dict(resnet18.named_buffers(remove_duplicate=False)),
}
params_flat, params_spec = pytree.tree_flatten(params)
params_flat = list(params_flat)
with torch.no_grad():
    numpy_inputs = recursively_convert_to_numpy(params_flat + [img])

golden_prediction = top3_possibilities(resnet18.forward(img), labels)
print("PyTorch prediction")
print(golden_prediction)

prediction = top3_possibilities(
    torch.from_numpy(getattr(fx_module, resnet18.__class__.__name__)(*numpy_inputs)),
    labels,
)
print("torch-mlir prediction")
print(prediction)
