# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from io import StringIO
import os
import sys
import tempfile

from torch_mlir.passmanager import PassManager
from torch_mlir.ir import StringAttr


def get_module_name_for_debug_dump(module):
    """Gets a name suitable for a debug dump.

    The name is not guaranteed to be unique.
    """
    if not "torch.debug_module_name" in module.operation.attributes:
        return "UnnammedModule"
    return StringAttr(module.operation.attributes["torch.debug_module_name"]).value


class TorchMlirCompilerError(Exception):
    pass

def run_pipeline_with_repro_report(module,
                                   pipeline: str,
                                   description: str,
                                   enable_ir_printing: bool = False):
    """Runs `pipeline` on `module`, with a nice repro report if it fails."""
    module_name = get_module_name_for_debug_dump(module)
    try:
        original_stderr = sys.stderr
        sys.stderr = StringIO()
        asm_for_error_report = module.operation.get_asm(
            large_elements_limit=10, enable_debug_info=True)
        # Lower module in place to make it ready for compiler backends.
        with module.context as ctx:
            pm = PassManager.parse(pipeline)
            if enable_ir_printing:
                ctx.enable_multithreading(False)
                pm.enable_ir_printing()
            pm.run(module.operation)
    except Exception as e:
        # TODO: More robust.
        # - don't arbitrarily clutter up /tmp. When a test suite has many
        #   tests, this can be a big disk cost (also, /tmp/ is frequently a
        #   RAM fs, which increases worries about capacity).
        # - don't have colliding filenames (hard to do without cluttering
        #   up /tmp)
        # - if we do have have colliding filenames, writes should at least
        #   avoid being racy.
        filename = os.path.join(tempfile.gettempdir(), module_name + ".mlir")
        with open(filename, 'w') as f:
            f.write(asm_for_error_report)
        debug_options="-mlir-print-ir-after-all -mlir-disable-threading"
        # Put something descriptive here even if description is empty.
        description = description or f"{module_name} compile"

        message = f"""\
            {description} failed with the following diagnostics:
            {sys.stderr.getvalue()}

            python exception: {e}

            For Torch-MLIR developers, the error can be reproduced with:
            $ torch-mlir-opt -pass-pipeline='{pipeline}' {filename}
            Add '{debug_options}' to get the IR dump for debugging purpose.
            """
        trimmed_message = '\n'.join([m.lstrip() for m in message.split('\n')])
        raise TorchMlirCompilerError(trimmed_message) from None
    finally:
        sys.stderr = original_stderr
