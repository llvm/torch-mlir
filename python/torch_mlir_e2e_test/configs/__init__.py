# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from .lazy_tensor_core import LazyTensorCoreTestConfig
from .linalg_on_tensors_backend import LinalgOnTensorsBackendTestConfig
from .native_torch import NativeTorchTestConfig
from .torchscript import TorchScriptTestConfig
from .mhlo_backend import MhloBackendTestConfig
from .tosa_backend import TosaBackendTestConfig
from .torchdynamo import TorchDynamoTestConfig
