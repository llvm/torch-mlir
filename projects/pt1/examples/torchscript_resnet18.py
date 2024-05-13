# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import sys
from pathlib import Path

import torch
import torchvision.models as models
from torch_mlir import torchscript
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend

sys.path.append(str(Path(__file__).absolute().parent))
from _example_utils import (
    top3_possibilities,
    load_and_preprocess_image,
    load_labels,
    DEFAULT_IMAGE_URL,
)


def predictions(torch_func, jit_func, img, labels):
    golden_prediction = top3_possibilities(torch_func(img), labels)
    print("PyTorch prediction")
    print(golden_prediction)
    prediction = top3_possibilities(torch.from_numpy(jit_func(img.numpy())), labels)
    print("torch-mlir prediction")
    print(prediction)


print("load image from " + DEFAULT_IMAGE_URL, file=sys.stderr)
img = load_and_preprocess_image(DEFAULT_IMAGE_URL)
labels = load_labels()

resnet18 = models.resnet18(pretrained=True).eval()
module = torchscript.compile(
    resnet18, torch.ones(1, 3, 224, 224), output_type="linalg-on-tensors"
)
backend = refbackend.RefBackendLinalgOnTensorsBackend()
compiled = backend.compile(module)
jit_module = backend.load(compiled)

predictions(resnet18.forward, jit_module.forward, img, labels)
