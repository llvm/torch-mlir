# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export


# ==============================================================================
# Global parameters
NUM_SEGMENTS = 42
NUM_BINS = 5000
NUM_LOGITS = 5000

class HistogramBinningCalibrationByFeature(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._num_segments = NUM_SEGMENTS
        self._num_bins = NUM_BINS
        self._num_logits = NUM_LOGITS
        _num_interval = (self._num_segments + 1) * self._num_bins
        _lower_bound = 0
        _upper_bound = 1
        l, u = _lower_bound, _upper_bound
        w = (u - l) / self._num_bins
        self.step = w
        self.register_buffer("_boundaries", torch.arange(l + w, u - w / 2, w))
        self.register_buffer(
            "_bin_num_examples",
            torch.empty([_num_interval], dtype=torch.float64).fill_(0.0),
        )
        self.register_buffer(
            "_bin_num_positives",
            torch.empty([_num_interval], dtype=torch.float64).fill_(0.0),
        )
        self.register_buffer("_bin_ids", torch.arange(_num_interval))
        self.register_buffer("positive_weight", torch.tensor([0.4]))
        self.bin_ctr_in_use_after = 0
        self.bin_ctr_weight_value = 0.9995
        self.oneminusbin_ctr_weight_value = 0.0005
        self._iteration = 0

    @export
    @annotate_args([
        None,
        ([-1], torch.int32, True),
        ([-1], torch.int32, True),
        ([-1], torch.float32, True),
    ])
    def forward(self, segment_value, segment_lengths, logit):
        origin_prediction = torch.sigmoid(
            logit + torch.log(self.positive_weight))
        # TODO: If in the future this test is removed from xfail for LTC, we will probably hit some device related
        #       issues below when new tensors are created on the CPU, which is currently the default behaviour.
        #       The solution would be to move these tensors to ensure they are on the same device as the existing ones.
        dense_segment_value = torch.zeros(logit.numel(), dtype=torch.int32)
        validoffsets = torch.gt(
            segment_lengths[1:self._num_logits+1], segment_lengths[0:self._num_logits])
        gathered_segment_values = (
            segment_value[segment_lengths[0:self._num_logits].long()]+1).int()
        dense_segment_value = torch.where(
            validoffsets, gathered_segment_values, dense_segment_value)
        zeros = torch.empty_like(
            dense_segment_value, dtype=torch.int32).fill_(0)
        isnotvalid = torch.gt(dense_segment_value, self._num_segments)
        dense_segment_value = torch.where(
            isnotvalid, zeros, dense_segment_value)
        bin_ids_data = torch.ceil(origin_prediction/self.step)-1
        bin_ids_data = bin_ids_data.long()
        curr_segment_value = dense_segment_value * self._num_bins
        bin_ids_data2 = bin_ids_data
        bin_ids_data = bin_ids_data + curr_segment_value
        curr_segment_value = self._bin_num_positives[bin_ids_data]
        curr_bin_num_examples = self._bin_num_examples[bin_ids_data]
        curr_segment_value = curr_segment_value / curr_bin_num_examples
        curr_segment_value = curr_segment_value.float()
        curr_segment_value = curr_segment_value * self.bin_ctr_weight_value + \
            origin_prediction * self.oneminusbin_ctr_weight_value
        isvalid = torch.gt(curr_bin_num_examples,
                           self.bin_ctr_in_use_after)
        calibrated_prediction_data = torch.where(
            isvalid, curr_segment_value, origin_prediction.float())
        return calibrated_prediction_data, bin_ids_data


@register_test_case(module_factory=lambda: HistogramBinningCalibrationByFeature())
def HBC_basic(module, tu: TestUtils):
    logits = torch.rand(NUM_LOGITS, dtype=torch.float)
    segment_lengths: Tensor = tu.randint(NUM_LOGITS, high=2).to(torch.int)
    segment_offsets: Tensor = torch.cumsum(segment_lengths, 0)
    segment_offsets: Tensor = torch.cat(
        (torch.tensor([0]), segment_offsets), 0)
    num_values: int = int(torch.sum(segment_lengths).item())
    segment_values: Tensor = tu.randint(num_values, high=NUM_SEGMENTS)
    segment_values = torch.cat(
        (segment_values, torch.zeros(NUM_LOGITS-segment_values.numel())), 0)
    module.forward(segment_values.int(), segment_offsets.int(), logits)
    #input shape (5000, 5001, 5000)
