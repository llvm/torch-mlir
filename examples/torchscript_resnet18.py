# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from PIL import Image
import requests
import torch
import torchvision.models as models
from torchvision import transforms

from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder

from torch_mlir.passmanager import PassManager
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend


mb = ModuleBuilder()

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


def predictions(torch_func, jit_func, img, labels):
    golden_prediction = top3_possibilities(torch_func(img))
    print("PyTorch prediction")
    print(golden_prediction)
    prediction = top3_possibilities(torch.from_numpy(jit_func(img.numpy())))
    print("torch-mlir prediction")
    print(prediction)


class ResNet18Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.train(False)

    def forward(self, img):
        return self.resnet.forward(img)


image_url = (
    "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg"
)
import sys

print("load image from " + image_url, file=sys.stderr)
img = load_and_preprocess_image(image_url)
labels = load_labels()

test_module = ResNet18Module()
class_annotator = ClassAnnotator()
recursivescriptmodule = torch.jit.script(test_module)
torch.jit.save(recursivescriptmodule, "/tmp/foo.pt")

class_annotator.exportNone(recursivescriptmodule._c._type())
class_annotator.exportPath(recursivescriptmodule._c._type(), ["forward"])
class_annotator.annotateArgs(
    recursivescriptmodule._c._type(),
    ["forward"],
    [
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ],
)
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c, class_annotator)

backend = refbackend.RefBackendLinalgOnTensorsBackend()
with mb.module.context:
    pm = PassManager.parse('torchscript-module-to-torch-backend-pipeline,torch-backend-to-linalg-on-tensors-backend-pipeline')
    pm.run(mb.module)

compiled = backend.compile(mb.module)
jit_module = backend.load(compiled)

predictions(test_module.forward, jit_module.forward, img, labels)
