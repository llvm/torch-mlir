# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

from fairseq.models import register_model
from fairseq.models.roberta.hub_interface import RobertaHubInterface
from fairseq.models.roberta.model import RobertaModel
from fairseq import hub_utils

# ==============================================================================


class FairseqXlmrModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        kwargs = {"data" : None, "user_dir" : None, "bpe_codes" : None, "sentencepiece_model" : None, "bpe_merges" : None, "bpe_vocab" : None, "xlmr.base" : None}
        x = hub_utils.from_pretrained(
            model_name_or_path="xlmr.base",
            checkpoint_file="model.pt",
            data_name_or_path=".",
            archive_map={
             "xlmr.base": "http://dl.fbaipublicfiles.com/fairseq/models/xlmr.base.tar.gz",
             "xlmr.large": "http://dl.fbaipublicfiles.com/fairseq/models/xlmr.large.tar.gz",
             "xlmr.xl": "http://dl.fbaipublicfiles.com/fairseq/models/xlmr/xlmr.xl.tar.gz",
             "xlmr.xxl": "http://dl.fbaipublicfiles.com/fairseq/models/xlmr/xlmr.xxl.tar.gz",
           },
            bpe="sentencepiece",
            load_checkpoint_heads=True,
            **kwargs,
        )
        return RobertaHubInterface(x["args"], x["task"], x["models"][0])

@register_test_case(module_factory=lambda: FairseqXlmrModule())
def FairseqXlmrModule_basic(module, tu: TestUtils):
    module.forward()
