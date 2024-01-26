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
import os
from pathlib import Path
import sys
import tempfile

import onnx

from ...extras import onnx_importer

from ...dialects import torch as torch_d
from ...ir import (
    Context,
)


def main(args: argparse.Namespace):
    model_proto = load_onnx_model(args)
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


def load_onnx_model(args: argparse.Namespace) -> onnx.ModelProto:
    # Do shape inference via files instead of in memory in order to handle
    # models > 2 GB. See https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md#shape-inference-a-large-onnx-model-2gb
    # for details about this technique.
    input_dir = os.path.dirname(os.path.abspath(args.input_file))
    temp_dir = input_dir if args.temp_dir is None else args.temp_dir
    with tempfile.TemporaryDirectory(dir=temp_dir, delete=not args.keep_temps) as td:
        # Infer shapes of the input file, saving results to a temp file
        temp_path = Path(td.name, "inferred.onnx")
        onnx.shape_inference.infer_shapes_path(args.file_path, temp_path)

        # Load the temp file and the external data.  External data does not
        # participate in shape inference, so the external data is the same
        # as would be used with the original model.
        inferred_model = onnx.load(temp_path, load_external_data=False)
        data_dir = input_dir if args.data_dir is None else args.data_dir
        onnx.load_external_data_for_model(inferred_model, data_dir)
        return inferred_model


def parse_arguments(argv=None) -> argparse.Namespace:
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
    parser.add_argument(
        "--keep-temps", action="store_true", help="Keep intermediate files"
    )
    parser.add_argument(
        "--temp-dir",
        help="Pre-existing directory in which to create temporary files."
            " Defaults to the directory of the input file.",
        type=Path
    )
    parser.add_argument(
        "--data-path",
        help="Directory containing the external data file(s)."
            " Defaults to the directory of the input file.",
        type=Path
    )
    args = parser.parse_args(argv)
    return args


def _cli_main():
    sys.exit(main(parse_arguments()))


if __name__ == "__main__":
    _cli_main()
