# -*- Python -*-
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import npcomp.frontends.pytorch as torch_mlir
import copy


def compare(a, b, test):
  print("Computing:" + test)
  err = (a.to('cpu') - b.to('cpu')).abs().max()
  if (err <= 1e-5):
    print("PASS! " + test + " check")
  else:
    print("FAILED " + test + " check")


def compare_eq(a, b, test):
  print("Computing:" + test)
  if (a == b):
    print("PASS! " + test + " check")
  else:
    print("FAILED " + test + " check")


def check_fwd(model, tensor):
  device = torch_mlir.mlir_device()
  result = model(tensor)
  device_model = copy.deepcopy(model).to(device)
  device_tensor = tensor.clone().to(device)
  device_result = device_model(device_tensor)

  compare(result, device_result, "fwd")
  return (device_model, device_result, result)


def check_ref(model, tensor):
  return check_fwd(model, tensor)


def check_back(fwd_path, target, lossmodel):
  device = torch_mlir.mlir_device()
  (device_model, device_result, result) = fwd_path
  device_target = target.clone().to(device)
  ref_loss = lossmodel(result, target)
  ref_loss.backward()
  device_loss = lossmodel(device_result, device_target)
  device_loss.backward()

  compare(ref_loss, device_loss, "back")
  return (device_model, device_result)
