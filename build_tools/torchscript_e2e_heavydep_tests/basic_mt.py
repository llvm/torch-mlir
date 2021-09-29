# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# Basic machine translation (MT) program.
#
# This is an ID's to ID's end-to-end program

import argparse
import tempfile
import unittest
import os
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from fairseq.data.dictionary import Dictionary
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules.multihead_attention import MultiheadAttention
from fairseq.models.transformer import TransformerModel
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
)
from fairseq.sequence_generator import SequenceGenerator
from fairseq.tasks.fairseq_task import LegacyFairseqTask
from fairseq import utils

from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

DEFAULT_TEST_VOCAB_SIZE = 100


class DummyTask(LegacyFairseqTask):
    def __init__(self, args):
        super().__init__(args)
        self.dictionary = _get_dummy_dictionary()
        if getattr(self.args, "ctc", False):
            self.dictionary.add_symbol("<ctc_blank>")
        self.src_dict = self.dictionary
        self.tgt_dict = self.dictionary

    @property
    def source_dictionary(self):
        return self.src_dict

    @property
    def target_dictionary(self):
        return self.dictionary


def _get_dummy_dictionary(vocab_size=DEFAULT_TEST_VOCAB_SIZE):
    dummy_dict = Dictionary()
    for id, _ in enumerate(range(vocab_size)):
        dummy_dict.add_symbol("{}".format(id), n=1000)
    return dummy_dict


def _get_dummy_task_and_parser():
    parser = argparse.ArgumentParser(description="test_dummy_s2s_task",
                                     argument_default=argparse.SUPPRESS)
    DummyTask.add_args(parser)
    args = parser.parse_args([])
    task = DummyTask.setup_task(args)
    return task, parser


class BasicMtModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        task, parser = _get_dummy_task_and_parser()
        TransformerModel.add_args(parser)
        args = parser.parse_args([])
        args.encoder_layers = 2
        args.decoder_layers = 1
        transformer_model = TransformerModel.build_model(args, task)
        self.sequence_generator = SequenceGenerator(
            [transformer_model],
            task.tgt_dict,
            beam_size=2,
            no_repeat_ngram_size=2,
            max_len_b=10,
        )

    # TODO: Support list/dict returns from functions.
    # This will allow us to handle a variable number of sentences.
    @export
    @annotate_args([
        None,
        ([2, -1], torch.long, True),
        ([2], torch.long, True),
    ])
    def forward(self, src_tokens, src_lengths):
        sample = {
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths
            }
        }
        result = self.sequence_generator(sample)
        return result[0][0]["tokens"], result[1][0]["tokens"]

EOS = BasicMtModule().sequence_generator.eos

@register_test_case(module_factory=lambda: BasicMtModule())
def BasicMtModule_basic(module, tu: TestUtils):
    # Imagine random sentences from the vocabulary. Use a subset of the
    # vocabulary that doesn't overlap with the EOS (which is usually the number
    # 2).
    MAX_SENTENCE_LENGTH = 10
    src_tokens = torch.randint(DEFAULT_TEST_VOCAB_SIZE // 4,
                               DEFAULT_TEST_VOCAB_SIZE // 3,
                               (2, MAX_SENTENCE_LENGTH)).long()
    # Append end-of-sentence symbol to the end of each sentence.
    src_tokens = torch.cat((src_tokens, torch.LongTensor([[EOS], [EOS]])), -1)
    src_lengths = torch.LongTensor([7, 10])
    module.forward(src_tokens, src_lengths)
