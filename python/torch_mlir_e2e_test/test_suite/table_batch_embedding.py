# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================

# Reference: https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/bench/split_table_batched_embeddings_benchmark.py#L270

# Global parameters.
NUM_TABLES = 2
NUM_EMBEDDINGS = 10
EMBEDDING_DIM = 4
BATCH_SIZE = 4
BAG_SIZE = 3


class TableBatchEmbeddingModule(torch.nn.Module):
    def __init__(self):
        super(TableBatchEmbeddingModule, self).__init__()
        self.num_tables = NUM_TABLES
        self.num_embeddings = NUM_EMBEDDINGS
        self.embedding_dim = EMBEDDING_DIM
        self.batch_size = BATCH_SIZE
        self.bag_size = BAG_SIZE
        # Currently, pooling_mode is fixed to 'sum'.
        self.nn_embedding_list = torch.nn.ModuleList([
            torch.nn.EmbeddingBag(
                self.num_embeddings, self.embedding_dim, mode="sum", sparse=False)
            for i in range(self.num_tables)
        ])

    @export
    @annotate_args([
        None,
        ([-1], torch.int64, True),
        ([-1], torch.int64, True),
    ])
    def forward(self, indices, offsets):
        indices_list = indices.view(self.num_tables, self.batch_size, self.bag_size)
        final_output = torch.tensor([])
        for i, nn_embedding in enumerate(self.nn_embedding_list):
            indices = indices_list[i].view(-1)
            output = nn_embedding(indices, offsets).view(self.batch_size, -1)
            final_output = torch.cat((final_output, output), dim=1)
        return final_output


@register_test_case(module_factory=lambda: TableBatchEmbeddingModule())
def TableBatchEmbeddingModule_basic(module, tu: TestUtils):
    indices = tu.randint(NUM_TABLES * BATCH_SIZE * BAG_SIZE, high=NUM_EMBEDDINGS)
    offsets = torch.cumsum(
        torch.tensor([0] + [BAG_SIZE for _ in range(BATCH_SIZE - 1)], dtype=torch.int64), 0)
    module.forward(indices, offsets)
