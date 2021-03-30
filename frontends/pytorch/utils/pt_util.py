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
    parser.add_argument("--exported-name", action="append",
                        help="""
Name to export, such as `my.submodule.forward`(default = export all).
Can pass repeatedly.
""")
    args = parser.parse_args()
    # TODO: Investigate why "cpu" is needed.
    module = torch.jit.load(args.pt_file, map_location="cpu")

    if args.dump:
        module._c.dump(code=True, attrs=False, params=False)

    # `import` is a Python keyword, so getattr is needed.
    if getattr(args, "import", False):
        class_annotator = torch_mlir.ClassAnnotator()
        if args.exported_name is not None:
            class_annotator.exportNone(module._c._type())
            for name in args.exported_name:
                class_annotator.exportPath(module._c._type(), name.split("."))
        mb = torch_mlir.ModuleBuilder()
        mb.import_module(module._c, class_annotator)
        mb.module.operation.print(large_elements_limit=16)


if __name__ == "__main__":
    main()
