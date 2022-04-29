# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import argparse

from torch_mlir_e2e_test.torchscript.serialization import serialize_all_tests_to

from . import hf_sequence_classification
from . import fully_connected_backward
from . import bert_functorch
from . import vision_models
from . import functorch_inference


def _get_argparse():
    parser = argparse.ArgumentParser(
        description="Generate assets for TorchScript E2E tests")
    parser.add_argument("--output_dir", help="The directory to put assets in.")
    return parser


def main():
    args = _get_argparse().parse_args()
    serialize_all_tests_to(args.output_dir)


if __name__ == "__main__":
    main()
