import argparse
import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from textwrap import dedent

import yaml

TORCH_MLIR_DIR = Path(__file__).parent.parent.resolve()
TORCH_DIR = TORCH_MLIR_DIR.parent.joinpath("pytorch")

sys.path.append(str(TORCH_DIR.joinpath("tools")))

# PyTorch's LTC backend autogen script
import codegen.dest.lazy_ir
import codegen.gen_lazy_tensor
from codegen.api.lazy import LazyIrSchema
from codegen.gen import get_grouped_native_functions, parse_native_yaml
from codegen.model import NativeFunctionsGroup
from codegen.gen_backend_stubs import parse_backend_yaml
from codegen.api.types import kernel_signature
from codegen.dest.lazy_ir import ComputeShapeSignature
from codegen.gen_lazy_tensor import parse_full_codegen_ops


def generate_native_functions(aten_ops_file: Path, out_file: Path):
    print("Generating Native Functions Yaml")

    native_yaml_path = TORCH_DIR.joinpath(
        "aten", "src", "ATen", "native", "native_functions.yaml"
    )

    parsed_yaml = parse_native_yaml(native_yaml_path)
    native_functions = parsed_yaml.native_functions
    grouped_native_functions = get_grouped_native_functions(native_functions)

    def get_native_function_name(f):
        func = f.func if hasattr(f, "func") else f.functional.func
        return str(func.name)

    aten_funcs = set(map(get_native_function_name, grouped_native_functions))

    # List of unsupported ops in LTC autogen because of some error
    blacklist = {
        "arange",  # Error: Code below assumes there is at least one tensor arg
        "bernoulli",  # Error: TODO add support for type BaseType(name=<BaseTy.Generator: 1>)
        "bernoulli_",  # Error: TODO add support for type BaseType(name=<BaseTy.Generator: 1>)
        "cat",  # Error: TODO not sure if there are other valid types to handle here
        "clone",  # Error: TODO add support for type BaseType(name=<BaseTy.MemoryFormat: 12>)
        "contiguous",  # Error: TODO add support for type BaseType(name=<BaseTy.MemoryFormat: 12>)
        "empty_like",  # Error: TODO add support for type BaseType(name=<BaseTy.MemoryFormat: 12>)
        "empty.memory_format",  # Error: TODO add support for type BaseType(name=<BaseTy.MemoryFormat: 12>)
        "index.Tensor",  # Error: TODO not sure if there are other valid types to handle here
        "index_put",  # Error: TODO not sure if there are other valid types to handle here
        "index_put_",  # Error: TODO not sure if there are other valid types to handle here
        "ones",  # Error: Code below assumes there is at least one tensor arg
        "ones_like",  # Error: TODO add support for type BaseType(name=<BaseTy.MemoryFormat: 12>)
        "resize_",  # Error: TODO add support for type BaseType(name=<BaseTy.MemoryFormat: 12>)
        "stack",  # Error: TODO not sure if there are other valid types to handle here
        "to.dtype",  # Error: TODO add support for type BaseType(name=<BaseTy.MemoryFormat: 12>)
        "to.other",  # Error: TODO add support for type BaseType(name=<BaseTy.MemoryFormat: 12>)
        "uniform_",  # Error: TODO add support for type BaseType(name=<BaseTy.MemoryFormat: 12>)
        "zeros",  # Error: Code below assumes there is at least one tensor arg
        "zeros_like",  # Error: TODO add support for type BaseType(name=<BaseTy.MemoryFormat: 12>)
    }

    # Additional ops which autogen is supported for but don't compile yet
    blacklist |= {"item", "size", "where"}

    # List of supported ops that we don't want to do the full codegen for
    # primarily view ops
    supported = {
        "expand",
        # "native_batch_norm_backward",
        "native_batch_norm",
        "permute",
        "repeat",
        "squeeze",
        "t",
        "unsqueeze",
        "view",
    }

    if which("rg") is not None:  # use ripgrep if available as its much faster
        cmd = ["rg", "-o", "-N", r"aten::[0-9a-zA-Z_\.]+"]
    else:
        cmd = ["grep", "-o", r"aten::[0-9a-zA-Z_\.]\+"]

    output = (
        subprocess.check_output(
            cmd + [str(aten_ops_file)],
            encoding="utf-8",
        )
        .strip()
        .split(os.linesep)
    )

    # process ops list
    ops = []
    supported_ops = []
    skipped = []

    for op in output:
        op = op[6:]
        opname = op.split(".")[0]

        if opname in blacklist or op in blacklist:
            continue

        if opname in supported:
            supported_ops.append(op)
            continue

        if op not in aten_funcs:
            skipped.append(op)
            continue

        ops.append(op)

    opnames = sorted(set(ops))

    with out_file.open("w") as f:
        yaml.dump(
            {
                "backend": "Lazy",
                "cpp_namespace": "torch_lazy_tensors",
                "full_codegen": opnames,
                "supported": sorted(supported_ops),
            },
            f,
            default_flow_style=False,
        )
        f.write(
            dedent(
                """

                # Skipped ops (supported by Torch-MLIR but no equivalent native function)
                """
            )
            + os.linesep.join(f"#  - {op}" for op in sorted(skipped))
        )

    return parsed_yaml, grouped_native_functions


@dataclass(frozen=True)
class MlirLazyIr(codegen.gen_lazy_tensor.dest.LazyIR):
    lowering_function_type: str = "torch::lazy::MlirFunction"
    lowering_context_type: str = "torch::lazy::MlirLoweringContext*"
    lowering_return_type: str = "torch::lazy::MlirOpVector"

    def lowering_body(self, f):
        func = (
            f.functional.func if isinstance(f, NativeFunctionsGroup) else f.func
        )
        schema = LazyIrSchema(func)

        return f"""
        UNIMPLEMENTED_ERROR(
            "'{func}' lowering not yet implemented"
        );
    """.rstrip()


def generate_backend(
    source_yaml: Path, backend_path: Path, parsed_yaml: dict, grouped_native_functions: list
):
    print("Running Lazy Tensor Autogen")

    # No fallback code allowed
    def gen_fallback_code(*args, **kwargs):
        return ""

    codegen.dest.lazy_ir.gen_fallback_code = gen_fallback_code

    codegen.gen_lazy_tensor.run(
        source_yaml=str(source_yaml),
        output_dir=str(backend_path),
        dry_run=False,
        impl_path=str(backend_path.joinpath("aten_ltc_mlir_type.cpp")),
        gen_ts_lowerings=False,
        node_base="torch::lazy::MlirNode",
        node_base_hdr=str(backend_path.joinpath("mlir_node.h")),
        tensor_class="torch::lazy::LazyTensor",
        tensor_class_hdr="torch/csrc/lazy/core/tensor.h",
        lazy_ir_cls=MlirLazyIr,
    )

    # Remove lazy_tensor_core imports
    subprocess.check_call(
        [
            "sed",
            "-i",
            "/lazy_tensor_core/d",
            str(backend_path.joinpath("LazyNativeFunctions.cpp")),
        ]
    )

    # Autogenerate shape inference placeholders
    import re

    sig_re = re.compile(f"std::vector<Shape> (?P<name>[_a-zA-Z0-9]+)\((?P<signature>.+)\);")
    shape_inference_decls = backend_path.joinpath("LazyShapeInference.h").read_text()

    shape_inference_defs = []
    for name, signature in sig_re.findall(shape_inference_decls):
        shape_inference_defs.append(
            dedent(
                f"""
                std::vector<Shape> {name}({signature}) {{
                    UNIMPLEMENTED_ERROR("{name}");
                }}
                """
            )
        )

    backend_path.joinpath("LazyShapeInference.cpp").write_text(
        dedent(
            """
            // This file contains autogenerated Lazy Shape Inference placeholders
            // for ops that dont have a corresponding structured kernel
            #include "LazyShapeInference.h"
            #include "../utils/exception.h"


            namespace torch_lazy_tensors {{
            namespace ir {{
            namespace ops {{

            using Shape = torch::lazy::Shape;

            {}

            }} // namespace ops
            }} // namespace ir
            }} // namespace torch_lazy_tensors
            """
        ).format("".join(shape_inference_defs))
    )


def main(args):
    script_path = Path(__file__).resolve()
    aten_ops_file = TORCH_MLIR_DIR.joinpath(
        "include", "torch-mlir", "Dialect", "Torch", "IR", "GeneratedAtenOps.td"
    )
    assert aten_ops_file.exists()
    native_functions = TORCH_MLIR_DIR.joinpath(
        "generated_native_functions.yaml"
    )

    prev_hash = None
    hash_file = TORCH_MLIR_DIR.joinpath("generated_backend.hash")
    if hash_file.exists():
        prev_hash = hash_file.read_text().strip()

    m = hashlib.sha256()
    m.update(script_path.read_bytes())
    m.update(aten_ops_file.read_bytes())
    if native_functions.exists():
        m.update(native_functions.read_bytes())

    new_hash = m.hexdigest().strip()

    if args.force or new_hash != prev_hash:
        hash_file.write_text(new_hash)
        parsed_yaml, grouped_native_functions = generate_native_functions(
            aten_ops_file, native_functions
        )

        backend_path = TORCH_MLIR_DIR.joinpath(
            "python", "torch_mlir", "csrc", "backend"
        )
        generate_backend(
            native_functions, backend_path, parsed_yaml, grouped_native_functions
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
    )
    main(parser.parse_args())
