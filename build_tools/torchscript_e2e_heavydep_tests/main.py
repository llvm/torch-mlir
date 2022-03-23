# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import argparse
import os
import pickle

import torch

from torch_mlir_e2e_test.torchscript.registry import GLOBAL_TEST_REGISTRY
from torch_mlir_e2e_test.torchscript.framework import SerializableTest, generate_golden_trace
from torch_mlir_e2e_test.torchscript.annotations import extract_serializable_annotations

from . import minilm_sequence_classification


def _get_argparse():
    parser = argparse.ArgumentParser(
        description="Generate assets for TorchScript E2E tests")
    parser.add_argument("--output_dir", help="The directory to put assets in.")
    return parser


def main():
    args = _get_argparse().parse_args()
    serializable_tests = []
    for test in GLOBAL_TEST_REGISTRY:
        trace = generate_golden_trace(test)
        module = torch.jit.script(test.program_factory())
        torchscript_module_bytes = module.save_to_buffer({
            "annotations.pkl":
            pickle.dumps(extract_serializable_annotations(module))
        })
        serializable_tests.append(
            SerializableTest(unique_name=test.unique_name,
                             program=torchscript_module_bytes,
                             trace=trace))
    for test in serializable_tests:
        with open(os.path.join(args.output_dir, f"{test.unique_name}.pkl"),
                  "wb") as f:
            pickle.dump(test, f)


if __name__ == "__main__":
    main()
