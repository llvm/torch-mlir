# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import torch_mlir

# RUN: %PYTHON %s | FileCheck %s

f32 = torch.randn((1,2,3,4))
f64 = f32.double()
f16 = f32.half()
i64 = f32.to(dtype=torch.int64)
i32 = f32.to(dtype=torch.int32)
i16 = f32.to(dtype=torch.int16)
i8 = f32.to(dtype=torch.int8)

# uint8 is the only unsigned type.
ui8 = f32.to(dtype=torch.uint8)

mb = torch_mlir.ModuleBuilder()
with mb.capture_function("float32_arg_results", [f32]) as f:
  f.returns([f32 * f32])
with mb.capture_function("float16_arg_results", [f16]) as f:
  f.returns([f16 * f16])
with mb.capture_function("float64_arg_results", [f64]) as f:
  f.returns([f32 * f64])
with mb.capture_function("int64_arg_results", [i64]) as f:
  f.returns([i64 * i64])
with mb.capture_function("int32_arg_results", [i32]) as f:
  f.returns([i32 * i32])
with mb.capture_function("int16_arg_results", [i16]) as f:
  f.returns([i16 * i16])
with mb.capture_function("int8_arg_results", [i8]) as f:
  f.returns([i8 * i8])
with mb.capture_function("uint8_arg_results", [ui8]) as f:
  f.returns([ui8 * ui8])


# CHECK: >> Symbol Table:
# CHECK: 'float16_arg_results' -> specialized func @float16_arg_results signature (NdArray[DType(float16), Shape(1, 2, 3, 4)]) -> NdArray[DType(float16), Shape(1, 2, 3, 4)]:
# CHECK: 'float32_arg_results' -> specialized func @float32_arg_results signature (NdArray[DType(float32), Shape(1, 2, 3, 4)]) -> NdArray[DType(float32), Shape(1, 2, 3, 4)]:
# CHECK: 'float64_arg_results' -> specialized func @float64_arg_results signature (NdArray[DType(float64), Shape(1, 2, 3, 4)]) -> NdArray[DType(float64), Shape(1, 2, 3, 4)]:
# CHECK: 'int16_arg_results' -> specialized func @int16_arg_results signature (NdArray[DType(int16), Shape(1, 2, 3, 4)]) -> NdArray[DType(int16), Shape(1, 2, 3, 4)]:
# CHECK: 'int32_arg_results' -> specialized func @int32_arg_results signature (NdArray[DType(int32), Shape(1, 2, 3, 4)]) -> NdArray[DType(int32), Shape(1, 2, 3, 4)]:
# CHECK: 'int64_arg_results' -> specialized func @int64_arg_results signature (NdArray[DType(int64), Shape(1, 2, 3, 4)]) -> NdArray[DType(int64), Shape(1, 2, 3, 4)]:
# CHECK: 'int8_arg_results' -> specialized func @int8_arg_results signature (NdArray[DType(int8), Shape(1, 2, 3, 4)]) -> NdArray[DType(int8), Shape(1, 2, 3, 4)]:
# CHECK: 'uint8_arg_results' -> specialized func @uint8_arg_results signature (NdArray[DType(int8), Shape(1, 2, 3, 4)]) -> NdArray[DType(int8), Shape(1, 2, 3, 4)]:
print(mb.meta_module)
