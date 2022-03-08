# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from PIL import Image
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder
from torch_mlir.passmanager import PassManager
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend
from module import Net

mb = ModuleBuilder()

import sys
import numpy as np
import skimage.io


def mnist(pretrained=False, **kwargs):
    """Constructs a mnist-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Net()
    model.load_state_dict(torch.load("mnist_cnn_wt.pt"))
    return model

class MnistModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mnist = mnist(pretrained=True)
        self.train(False)

    def forward(self, img):
        return self.mnist.forward(img)



# MNIST sample images
IMAGE_URLS = [
    'https://i.imgur.com/SdYYBDt.png',  # 0
    'https://i.imgur.com/Wy7mad6.png',  # 1
    'https://i.imgur.com/nhBZndj.png',  # 2
    'https://i.imgur.com/V6XeoWZ.png',  # 3
    'https://i.imgur.com/EdxBM1B.png',  # 4
    'https://i.imgur.com/zWSDIuV.png',  # 5
    'https://i.imgur.com/Y28rZho.png',  # 6
    'https://i.imgur.com/6qsCz2W.png',  # 7
    'https://i.imgur.com/BVorzCP.png',  # 8
    'https://i.imgur.com/vt5Edjb.png',  # 9
]

def load_images():
    """Load MNIST sample images from the web and return them in an array.
    Returns:
        Numpy array of size (10, 28, 28, 1) with MNIST sample images.
    """
    images = np.zeros((10, 28, 28, 1))
    for idx, url in enumerate(IMAGE_URLS):
        images[idx, :, :, 0] = skimage.io.imread(url)
    return images[0,:,:,:]


def load_and_preprocess_image(img_url: str):
    #img = Image.open(img).convert("RGB")
    #images = np.zeros((10, 28, 28, 1))
    #for idx, url in enumerate(IMAGE_URLS):
    #    images[idx, :, :, 0] = skimage.io.imread(url)
    image=skimage.io.imread(img_url)
    # preprocessing pipeline
    preprocess=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    img_preprocessed = preprocess(image)
    return torch.unsqueeze(img_preprocessed, 0)


def top1_possibilities(res):
    _, indexes = torch.sort(res, descending=True)
    percentage = torch.nn.functional.softmax(res, dim=1)[0] * 100
    top1 = [(labels[idx], percentage[idx].item()) for idx in indexes[0][:1]]
    return top1


def predictions(torch_func, jit_func, img, labels):
    golden_prediction = top1_possibilities(torch_func(img))
    print("PyTorch prediction")
    print(golden_prediction)
    prediction = top1_possibilities(torch.from_numpy(jit_func(img.numpy())))
    print("torch-mlir prediction")
    print(prediction)

processedImage=IMAGE_URLS[5]  # specify the image url number that you want to set as the input
print("load image from " + processedImage, file=sys.stderr)
img = load_and_preprocess_image(processedImage)
labels = ['0','1','2','3','4','5','6','7','8','9']

test_module = MnistModule()
class_annotator = ClassAnnotator()
recursivescriptmodule = torch.jit.script(test_module)
#torch.jit.save(recursivescriptmodule, "/tmp/foo.pt")
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
print("import mnist module")
mb.import_module(recursivescriptmodule._c, class_annotator)
#print(mb.module)

original_output = sys.stdout
with open('mnist_torchscript_import.mlir', 'w') as f:
    sys.stdout = f
    print(mb.module)
    sys.stdout = original_output 
print('write imported torchscript module done')


backend = refbackend.RefBackendLinalgOnTensorsBackend()
with mb.module.context:
    pm = PassManager.parse('torchscript-module-to-torch-backend-pipeline,torch-backend-to-linalg-on-tensors-backend-pipeline')
    pm.run(mb.module)
print('pass manager done')

compiled = backend.compile(mb.module)
print('backend compile done')

original_output = sys.stdout
with open('mnist_compiled.mlir', 'w') as f:
    sys.stdout = f
    print(compiled)
    sys.stdout = original_output 
print('write compiled mnist module done')

jit_module = backend.load(compiled)

print(img.type)
predictions(test_module.forward, jit_module.forward, img, labels)
