# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import sys

import requests
import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms

from torch_mlir.eager_mode.torch_mlir_tensor import TorchMLIRTensor


def load_and_preprocess_image(url: str):
    headers = {
        'User-Agent':
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
    }
    img = Image.open(requests.get(url, headers=headers,
                                  stream=True).raw).convert("RGB")
    # preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img_preprocessed = preprocess(img)
    return torch.unsqueeze(img_preprocessed, 0)


def load_labels():
    classes_text = requests.get(
        "https://raw.githubusercontent.com/cathyzhyi/ml-data/main/imagenet-classes.txt",
        stream=True,
    ).text
    labels = [line.strip() for line in classes_text.splitlines()]
    return labels


def top3_possibilities(res):
    _, indexes = torch.sort(res, descending=True)
    percentage = torch.nn.functional.softmax(res, dim=1)[0] * 100
    top3 = [(labels[idx], percentage[idx].item()) for idx in indexes[0][:3]]
    return top3


def predictions(torch_func, img, labels):
    golden_prediction = top3_possibilities(torch_func(img))
    print("PyTorch prediction")
    print(golden_prediction)
    prediction = top3_possibilities(torch_func(TorchMLIRTensor(img)))
    print("torch-mlir prediction")
    print(prediction)


class ResNet18Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.train(False)

    def forward(self, img):
        return self.resnet.forward(img)


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.s = ResNet18Module()

    def forward(self, x):
        return self.s.forward(x)


image_url = (
    "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg"
)

print("load image from " + image_url, file=sys.stderr)
img = load_and_preprocess_image(image_url)
labels = load_labels()

test_module = TestModule()
predictions(test_module.forward, img, labels)
