# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

"""Console tool for converting an ONNX proto to torch IR.

Typically, when installed from a wheel, this can be invoked as:

  torch-mlir-import-onnx some.pb

Or from Python:
  
  python -m torch_mlir.tools.import_onnx ...
"""
import argparse
from pathlib import Path
import sys

import onnx

from ...extras import onnx_importer

from ...dialects import torch as torch_d
from ...ir import (
    Context,
)


def main(args):
    model_proto = load_onnx_model(args.input_file)
    context = Context()
    torch_d.register_dialect(context)
    model_info = onnx_importer.ModelInfo(model_proto)
    m = model_info.create_module(context=context)
    imp = onnx_importer.NodeImporter.define_function(model_info.main_graph, m)
    imp.import_all()
    if not args.no_verify:
        m.verify()

    # TODO: This isn't very efficient output. If these files ever
    # get large, enable bytecode and direct binary emission to save
    # some copies.
    if args.output_file and args.output_file != "-":
        with open(args.output_file, "wt") as f:
            print(m.get_asm(assume_verified=not args.no_verify), file=f)
    else:
        print(m.get_asm(assume_verified=not args.no_verify))


def load_onnx_model(file_path: Path) -> onnx.ModelProto:
    raw_model = onnx.load(file_path)
    inferred_model = onnx.shape_inference.infer_shapes(raw_model)
    return inferred_model


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description="Torch-mlir ONNX import tool")
    parser.add_argument("input_file", help="ONNX protobuf input", type=Path)
    parser.add_argument(
        "-o", dest="output_file", help="Output path (or '-' for stdout)"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Disable verification prior to printing",
    )
    args = parser.parse_args(argv)
    return args


def _cli_main():
    sys.exit(main(parse_arguments()))


if __name__ == "__main__":
    _cli_main()
