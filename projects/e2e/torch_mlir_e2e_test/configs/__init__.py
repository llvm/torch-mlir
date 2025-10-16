# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

__all__ = ["load_config"]

from importlib import import_module

CONFIG_LOCATIONS = {
    "LazyTensorCoreTestConfig": "lazy_tensor_core",
    "NativeTorchTestConfig": "native_torch",
    "OnnxBackendTestConfig": "onnx_backend",
    "TorchScriptTestConfig": "torchscript",
    "TorchDynamoTestConfig": "torchdynamo",
    "JITImporterTestConfig": "jit_importer_backend",
    "FxImporterTestConfig": "fx_importer_backend",
}

def load_config(name: str) -> type:
    source = CONFIG_LOCATIONS.get(name)
    assert source is not None, f"Could not find TestConfig named {name}."
    module = import_module(f".{source}", __package__)
    return getattr(module, name)
