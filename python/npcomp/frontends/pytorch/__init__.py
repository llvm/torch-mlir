# -*- Python -*-
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import _torch_mlir
from _torch_mlir import _get_mlir
from _torch_mlir import _op_report
from _torch_mlir import _liveness_report
from _torch_mlir import set_debug
from _torch_mlir import lower_to_std

import json

_torch_mlir._initialize_aten_bindings()
_torch_mlir.set_debug(False, "")


def get_mlir(t):
  if not isinstance(t, list):
    t = [t]
  return _get_mlir(t)


def op_report(mlir):
  return json.loads(_op_report(mlir))


def liveness_report(mlir):
  return json.loads(_liveness_report(mlir))


def get_mlir_supported_devices(devkind=None):
  # TODO: define our own device and stop hijacking the xla device.
  return ["xla:0"]


def mlir_device(devkind=None):
  devices = get_mlir_supported_devices(devkind=devkind)
  device = devices[0]
  return torch.device(device)


__all__ = ['get_mlir', 'mlir_device', 'op_report', 'liveness_report']
