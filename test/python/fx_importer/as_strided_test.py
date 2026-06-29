# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

import torch
import torch.nn as nn
from torch.export import Dim

from torch_mlir import fx


def run(f):
    print(f"{f.__name__}")
    print("-" * len(f.__name__))
    f()
    print()


def import_module(module, *args):
    imported = fx.export_and_import(module, *args)
    print(str(imported).split("{-#", 1)[0])


def expect_reject(module, *args, **kwargs):
    try:
        fx.export_and_import(module, *args, **kwargs)
    except ValueError as e:
        print(f"ValueError: {e}")
        return
    raise AssertionError("expected aten.as_strided.default import to fail")


@run
# CHECK-LABEL: test_as_strided_after_transpose
# CHECK: func.func @main(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[2,2],f32>
# CHECK: %[[T_INDEX0:.*]] = torch.vtensor.literal{{.*}}tensor<2x2xsi64>
# CHECK: %[[T_INDEX1:.*]] = torch.vtensor.literal{{.*}}tensor<2x2xsi64>
# CHECK: %[[T_INDICES:.*]] = torch.prim.ListConstruct %[[T_INDEX0]], %[[T_INDEX1]] : (!torch.vtensor<[2,2],si64>, !torch.vtensor<[2,2],si64>) -> !torch.list<optional<vtensor>>
# CHECK-NOT: torch.aten.as_strided
# CHECK: %[[T_RESULT:.*]] = torch.aten.index.Tensor %arg0, %[[T_INDICES]] : !torch.vtensor<[3,4],f32>, !torch.list<optional<vtensor>> -> !torch.vtensor<[2,2],f32>
# CHECK-NOT: torch.aten.as_strided
# CHECK: return %[[T_RESULT]] : !torch.vtensor<[2,2],f32>
def test_as_strided_after_transpose():
    class M(nn.Module):
        def forward(self, x):
            view = x.transpose(0, 1)
            return torch.ops.aten.as_strided.default(view, [2, 2], [4, 1], 0)

    import_module(M(), torch.arange(12, dtype=torch.float32).reshape(3, 4))


@run
# CHECK-LABEL: test_as_strided_channels_last_input
# CHECK: func.func @main(%arg0: !torch.vtensor<[1,3,4,5],f32>) -> !torch.vtensor<[1,4,5,3],f32>
# CHECK: %[[IN_INDEX0:.*]] = torch.vtensor.literal{{.*}}tensor<1x4x5x3xsi64>
# CHECK: %[[IN_INDEX1:.*]] = torch.vtensor.literal{{.*}}tensor<1x4x5x3xsi64>
# CHECK: %[[IN_INDEX2:.*]] = torch.vtensor.literal{{.*}}tensor<1x4x5x3xsi64>
# CHECK: %[[IN_INDEX3:.*]] = torch.vtensor.literal{{.*}}tensor<1x4x5x3xsi64>
# CHECK: %[[IN_INDICES:.*]] = torch.prim.ListConstruct %[[IN_INDEX0]], %[[IN_INDEX1]], %[[IN_INDEX2]], %[[IN_INDEX3]] : (!torch.vtensor<[1,4,5,3],si64>, !torch.vtensor<[1,4,5,3],si64>, !torch.vtensor<[1,4,5,3],si64>, !torch.vtensor<[1,4,5,3],si64>) -> !torch.list<optional<vtensor>>
# CHECK-NOT: torch.aten.as_strided
# CHECK: %[[IN_RESULT:.*]] = torch.aten.index.Tensor %arg0, %[[IN_INDICES]] : !torch.vtensor<[1,3,4,5],f32>, !torch.list<optional<vtensor>> -> !torch.vtensor<[1,4,5,3],f32>
# CHECK-NOT: torch.aten.as_strided
# CHECK: return %[[IN_RESULT]] : !torch.vtensor<[1,4,5,3],f32>
def test_as_strided_channels_last_input():
    class M(nn.Module):
        def forward(self, x):
            return torch.ops.aten.as_strided.default(x, [1, 4, 5, 3], [60, 15, 3, 1], 0)

    x = torch.arange(60, dtype=torch.float32).reshape(1, 3, 4, 5)
    x = x.contiguous(memory_format=torch.channels_last)
    import_module(M(), x)


@run
# CHECK-LABEL: test_as_strided_channels_last_parameter
# CHECK: func.func @main() -> !torch.vtensor<[1,4,5,3],f32>
# CHECK: %[[PARAM_BASE:.*]] = torch.vtensor.literal{{.*}}tensor<1x3x4x5xf32>
# CHECK: %[[PARAM_INDEX0:.*]] = torch.vtensor.literal{{.*}}tensor<1x4x5x3xsi64>
# CHECK: %[[PARAM_INDEX1:.*]] = torch.vtensor.literal{{.*}}tensor<1x4x5x3xsi64>
# CHECK: %[[PARAM_INDEX2:.*]] = torch.vtensor.literal{{.*}}tensor<1x4x5x3xsi64>
# CHECK: %[[PARAM_INDEX3:.*]] = torch.vtensor.literal{{.*}}tensor<1x4x5x3xsi64>
# CHECK: %[[PARAM_INDICES:.*]] = torch.prim.ListConstruct %[[PARAM_INDEX0]], %[[PARAM_INDEX1]], %[[PARAM_INDEX2]], %[[PARAM_INDEX3]] : (!torch.vtensor<[1,4,5,3],si64>, !torch.vtensor<[1,4,5,3],si64>, !torch.vtensor<[1,4,5,3],si64>, !torch.vtensor<[1,4,5,3],si64>) -> !torch.list<optional<vtensor>>
# CHECK-NOT: torch.aten.as_strided
# CHECK: %[[PARAM_RESULT:.*]] = torch.aten.index.Tensor %[[PARAM_BASE]], %[[PARAM_INDICES]] : !torch.vtensor<[1,3,4,5],f32>, !torch.list<optional<vtensor>> -> !torch.vtensor<[1,4,5,3],f32>
# CHECK-NOT: torch.aten.as_strided
# CHECK: return %[[PARAM_RESULT]] : !torch.vtensor<[1,4,5,3],f32>
def test_as_strided_channels_last_parameter():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            weight = torch.arange(60, dtype=torch.float32).reshape(1, 3, 4, 5)
            weight = weight.contiguous(memory_format=torch.channels_last)
            self.weight = nn.Parameter(weight, requires_grad=False)

        def forward(self):
            return torch.ops.aten.as_strided.default(
                self.weight, [1, 4, 5, 3], [60, 15, 3, 1], 0
            )

    import_module(M())


@run
# CHECK-LABEL: test_as_strided_after_to_channels_last
# CHECK: func.func @main(%arg0: !torch.vtensor<[1,3,4,5],f32>) -> !torch.vtensor<[1,4,5,3],f32>
# CHECK: %[[CL_CONVERT:.*]] = torch.prims.convert_element_type %arg0
# CHECK: %[[CL_BASE:.*]] = torch.aten.clone %[[CL_CONVERT]]
# CHECK: %[[CL_INDEX0:.*]] = torch.vtensor.literal{{.*}}tensor<1x4x5x3xsi64>
# CHECK: %[[CL_INDEX1:.*]] = torch.vtensor.literal{{.*}}tensor<1x4x5x3xsi64>
# CHECK: %[[CL_INDEX2:.*]] = torch.vtensor.literal{{.*}}tensor<1x4x5x3xsi64>
# CHECK: %[[CL_INDEX3:.*]] = torch.vtensor.literal{{.*}}tensor<1x4x5x3xsi64>
# CHECK: %[[CL_INDICES:.*]] = torch.prim.ListConstruct %[[CL_INDEX0]], %[[CL_INDEX1]], %[[CL_INDEX2]], %[[CL_INDEX3]] : (!torch.vtensor<[1,4,5,3],si64>, !torch.vtensor<[1,4,5,3],si64>, !torch.vtensor<[1,4,5,3],si64>, !torch.vtensor<[1,4,5,3],si64>) -> !torch.list<optional<vtensor>>
# CHECK-NOT: torch.aten.as_strided
# CHECK: %[[CL_RESULT:.*]] = torch.aten.index.Tensor %[[CL_BASE]], %[[CL_INDICES]] : !torch.vtensor<[1,3,4,5],f32>, !torch.list<optional<vtensor>> -> !torch.vtensor<[1,4,5,3],f32>
# CHECK-NOT: torch.aten.as_strided
# CHECK: return %[[CL_RESULT]] : !torch.vtensor<[1,4,5,3],f32>
def test_as_strided_after_to_channels_last():
    class M(nn.Module):
        def forward(self, x):
            view = x.to(memory_format=torch.channels_last)
            return torch.ops.aten.as_strided.default(
                view, [1, 4, 5, 3], [60, 15, 3, 1], 0
            )

    import_module(M(), torch.arange(60, dtype=torch.float32).reshape(1, 3, 4, 5))


@run
# CHECK-LABEL: test_as_strided_after_contiguous
# CHECK: func.func @main(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[2,2],f32>
# CHECK: %[[CONTIG_VIEW:.*]] = torch.aten.transpose.int %arg0
# CHECK: %[[CONTIG_BASE:.*]] = torch.aten.clone %[[CONTIG_VIEW]]
# CHECK: %[[CONTIG_INDEX0:.*]] = torch.vtensor.literal{{.*}}tensor<2x2xsi64>
# CHECK: %[[CONTIG_INDEX1:.*]] = torch.vtensor.literal{{.*}}tensor<2x2xsi64>
# CHECK: %[[CONTIG_INDICES:.*]] = torch.prim.ListConstruct %[[CONTIG_INDEX0]], %[[CONTIG_INDEX1]] : (!torch.vtensor<[2,2],si64>, !torch.vtensor<[2,2],si64>) -> !torch.list<optional<vtensor>>
# CHECK-NOT: torch.aten.as_strided
# CHECK: %[[CONTIG_RESULT:.*]] = torch.aten.index.Tensor %[[CONTIG_BASE]], %[[CONTIG_INDICES]] : !torch.vtensor<[4,3],f32>, !torch.list<optional<vtensor>> -> !torch.vtensor<[2,2],f32>
# CHECK-NOT: torch.aten.as_strided
# CHECK: return %[[CONTIG_RESULT]] : !torch.vtensor<[2,2],f32>
def test_as_strided_after_contiguous():
    class M(nn.Module):
        def forward(self, x):
            view = x.transpose(0, 1).contiguous()
            return torch.ops.aten.as_strided.default(view, [2, 2], [3, 1], 0)

    import_module(M(), torch.arange(12, dtype=torch.float32).reshape(3, 4))


@run
# CHECK-LABEL: test_as_strided_after_slice_explicit_offset
# CHECK: func.func @main(%arg0: !torch.vtensor<[8],f32>) -> !torch.vtensor<[2],f32>
# CHECK: %[[SLICE_INDEX0:.*]] = torch.vtensor.literal{{.*}}tensor<2xsi64>
# CHECK: %[[SLICE_INDICES:.*]] = torch.prim.ListConstruct %[[SLICE_INDEX0]] : (!torch.vtensor<[2],si64>) -> !torch.list<optional<vtensor>>
# CHECK-NOT: torch.aten.as_strided
# CHECK: %[[SLICE_RESULT:.*]] = torch.aten.index.Tensor %arg0, %[[SLICE_INDICES]] : !torch.vtensor<[8],f32>, !torch.list<optional<vtensor>> -> !torch.vtensor<[2],f32>
# CHECK-NOT: torch.aten.as_strided
# CHECK: return %[[SLICE_RESULT]] : !torch.vtensor<[2],f32>
def test_as_strided_after_slice_explicit_offset():
    class M(nn.Module):
        def forward(self, x):
            view = x[1:]
            return torch.ops.aten.as_strided.default(view, [2], [1], 0)

    import_module(M(), torch.arange(8, dtype=torch.float32))


@run
# CHECK-LABEL: test_as_strided_nested_explicit_offset
# CHECK: func.func @main(%arg0: !torch.vtensor<[10],f32>) -> !torch.vtensor<[2],f32>
# CHECK: %[[NESTED_INDEX0:.*]] = torch.vtensor.literal{{.*}}tensor<2xsi64>
# CHECK: %[[NESTED_INDICES:.*]] = torch.prim.ListConstruct %[[NESTED_INDEX0]] : (!torch.vtensor<[2],si64>) -> !torch.list<optional<vtensor>>
# CHECK-NOT: torch.aten.as_strided
# CHECK: %[[NESTED_RESULT:.*]] = torch.aten.index.Tensor %arg0, %[[NESTED_INDICES]] : !torch.vtensor<[10],f32>, !torch.list<optional<vtensor>> -> !torch.vtensor<[2],f32>
# CHECK-NOT: torch.aten.as_strided
# CHECK: return %[[NESTED_RESULT]] : !torch.vtensor<[2],f32>
def test_as_strided_nested_explicit_offset():
    class M(nn.Module):
        def forward(self, x):
            view = torch.ops.aten.as_strided.default(x, [2], [1], 2)
            return torch.ops.aten.as_strided.default(view, [2], [1], 0)

    import_module(M(), torch.arange(10, dtype=torch.float32))


@run
# CHECK-LABEL: test_as_strided_rejects_dynamic_input_metadata
# CHECK: ValueError: aten.as_strided.default `base.shape` must be static before Torch IR import
def test_as_strided_rejects_dynamic_input_metadata():
    class M(nn.Module):
        def forward(self, x):
            return torch.ops.aten.as_strided.default(x, [2], [1], 0)

    expect_reject(
        M(),
        torch.arange(4, dtype=torch.float32),
        dynamic_shapes={"x": {0: Dim("n", min=2, max=8)}},
        import_symbolic_shape_expressions=True,
    )


@run
# CHECK-LABEL: test_as_strided_rejects_dynamic_size
# CHECK: ValueError: aten.as_strided.default `size` must be a static int list
def test_as_strided_rejects_dynamic_size():
    class M(nn.Module):
        def forward(self, x):
            return torch.ops.aten.as_strided.default(x, [x.shape[0]], [1], 0)

    expect_reject(
        M(),
        torch.arange(4, dtype=torch.float32),
        dynamic_shapes={"x": {0: Dim("n", min=2, max=8)}},
    )


@run
# CHECK-LABEL: test_as_strided_rejects_dynamic_stride
# CHECK: ValueError: aten.as_strided.default `stride` must be a static int list
def test_as_strided_rejects_dynamic_stride():
    class M(nn.Module):
        def forward(self, x):
            return torch.ops.aten.as_strided.default(x, [2], [x.shape[0]], 0)

    expect_reject(
        M(),
        torch.arange(4, dtype=torch.float32),
        dynamic_shapes={"x": {0: Dim("n", min=2, max=8)}},
    )


@run
# CHECK-LABEL: test_as_strided_rejects_dynamic_storage_offset
# CHECK: ValueError: aten.as_strided.default `storage_offset` must be static before Torch IR import
def test_as_strided_rejects_dynamic_storage_offset():
    class M(nn.Module):
        def forward(self, x):
            return torch.ops.aten.as_strided.default(x, [2], [1], x.shape[0] - 2)

    expect_reject(
        M(),
        torch.arange(4, dtype=torch.float32),
        dynamic_shapes={"x": {0: Dim("n", min=2, max=8)}},
    )


@run
# CHECK-LABEL: test_as_strided_rejects_oob
# CHECK: ValueError: aten.as_strided.default indexes outside the base tensor
def test_as_strided_rejects_oob():
    class M(nn.Module):
        def forward(self, x):
            return torch.ops.aten.as_strided.default(x, [4], [1], 2)

    expect_reject(
        M(),
        torch.arange(4, dtype=torch.float32),
    )


@run
# CHECK-LABEL: test_as_strided_rejects_unmappable_program_input
# CHECK: ValueError: aten.as_strided.default storage offsets cannot be mapped to base indices
def test_as_strided_rejects_unmappable_program_input():
    class M(nn.Module):
        def forward(self, x):
            return torch.ops.aten.as_strided.default(x, [4], [1], 0)

    expect_reject(
        M(),
        torch.arange(8, dtype=torch.float32)[::2],
    )
