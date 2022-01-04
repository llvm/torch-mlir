# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export


# ==============================================================================

class HistogramBinningCalibrationByFeature(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # HistogramBinningCalibrationByFeature
        self._num_segments = 42
        self._num_bins = 5000
        _num_interval = (self._num_segments + 1) * self._num_bins
        _lower_bound = 0
        _upper_bound = 1
        l, u = _lower_bound, _upper_bound
        w = (u - l) / self._num_bins
        self.step=w
        self.register_buffer("_boundaries", torch.arange(l + w, u - w / 2, w))
        self.register_buffer(
            "_bin_num_examples",
            torch.empty([_num_interval], dtype=torch.float64).fill_(0.0),
        )  # ConstantFill
        self.register_buffer(
            "_bin_num_positives",
            torch.empty([_num_interval], dtype=torch.float64).fill_(0.0),
        )  # ConstantFill
        self.register_buffer("_bin_ids", torch.arange(_num_interval))

        self._iteration = 0

    @export
    @annotate_args([
        None,
        ([-1], torch.int64, True),
        ([-1, -1], torch.int64, True),
        ([5, 1], torch.float32, True),
    ])
    def forward(self, segment_value, segment_lengths, logit):
        origin_prediction = torch.sigmoid(logit - 0.9162907600402832)
        # HistogramBinningCalibrationByFeature
        _3251 = origin_prediction.view(-1)  # Reshape
        dense_segment_value = torch.zeros(logit.numel(), dtype=torch.int64)
        offsets = torch.arange(segment_lengths.numel())
        offsets = offsets.unsqueeze(1)
        dense_segment_value[offsets] = segment_value[offsets] + 1
        _3257 = dense_segment_value
        _3253 = segment_lengths.view(-1)
        _3258 = _3257.view(-1)  # Reshape
        _3259 = _3258.long()  # Cast
        _3260 = torch.empty_like(_3253, dtype=torch.int64).fill_(0)  # ConstantFill
        _3261 = torch.empty_like(_3253, dtype=torch.int64).fill_(1)  # ConstantFill
        _3262 = torch.gt(_3259, self._num_segments)  # GT
        _3263 = torch.gt(_3260, _3259)  # GT
        _3264 = _3253 == _3261  # EQ
        _3265 = torch.where(_3262, _3260, _3259)  # Conditional
        _3266 = torch.where(_3263, _3260, _3265)  # Conditional
        _3267 = torch.where(_3264, _3266, _3260)  # Conditional
        _3268 = torch.ceil(_3251/self.step)-1  # Bucketize
        _3269 = _3268.long()  # Cast
        _3270 = _3267 * self._num_bins  # Mul
        _3271 = _3269 + _3270  # Add
        _3272 = _3271.int()  # Cast
        _3273 = self._bin_num_positives[_3272.long()]  # Gather
        _3274 = self._bin_num_examples[_3272.long()]  # Gather
        _3275 = _3273 / _3274  # Div
        _3276 = _3275.float()  # Cast
        _3277 = _3276 * 0.9995 + _3251 * 0.0005  # WeightedSum
        _3278 = torch.gt(_3274, 10000.0)  # GT
        _3279 = torch.where(_3278, _3277, _3251.float())  # Conditional
        _3280 = _3279.view(-1)  # Reshape
        prediction =_3280.unsqueeze(1)  # Reshape
        return prediction


@register_test_case(module_factory=lambda: HistogramBinningCalibrationByFeature())
def HBC_basic(module, tu: TestUtils):
    segment_value = torch.tensor([40, 31, 32, 13, 31])
    segment_lengths = torch.tensor([[1], [1], [1], [1], [1]])
    logit = torch.tensor([[-0.0018], [0.0085], [0.0090], [0.0003], [0.0029]])

    module.forward(segment_value, segment_lengths, logit)
