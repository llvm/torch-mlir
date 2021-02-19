#!/usr/bin/env python3
"""
Utility for handling common tasks for exported `.pt` model files.

Usage:
    # Dump PyTorch data structures for .pt file.
    # This does not involve any MLIR code.
    $ pt_util.py --dump model.pt

    # Import the .pt file into MLIR.
    $ pt_util.py --import model.pt
"""

import torch
import torch_mlir

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Utility for .pt files")
    parser.add_argument("pt_file", metavar="PT_FILE", type=str,
                        help="the .pt file to import")
    parser.add_argument("--dump", action="store_true",
                        help="dump the pytorch module")
    parser.add_argument("--import", action="store_true",
                        help="import the pytorch module")
    args = parser.parse_args()
    # TODO: Investigate why "cpu" is needed.
    module = torch.jit.load(args.pt_file, map_location="cpu")
    mb = torch_mlir.ModuleBuilder()
    if args.dump:
        module._c.dump(code=True, attrs=False, params=False)
    # `import` is a Python keyword, so getattr is needed.
    if getattr(args, "import", False):
        mb.import_module(module._c)
        mb.module.operation.print(large_elements_limit=16)


if __name__ == "__main__":
    main()
