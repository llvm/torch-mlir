import argparse
import hashlib
import os
import subprocess
import sys
import warnings
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


def generate_native_functions(
    config_path: Path, torch_ops_file: Path, out_file: Path
):
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

    with config_path.open() as f:
        config = yaml.load(f, yaml.CLoader)

    # List of unsupported ops in LTC autogen because of some error
    blacklist = config.get("blacklist", [])

    # List of supported ops that we don't want to do the full codegen for
    # primarily view ops
    supported = config.get("supported", [])

    if which("rg") is not None:  # use ripgrep if available as its much faster
        cmd = ["rg", "-o", "-N", r"aten::[0-9a-zA-Z_\.]+"]
    else:
        cmd = ["grep", "-o", r"aten::[0-9a-zA-Z_\.]\+"]

    output = (
        subprocess.check_output(
            cmd + [str(torch_ops_file)],
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

    # Additional ops to support that are not supported by Torch-MLIR explicitly
    supported_ops.extend(config.get("additional_ops", []))

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
    source_yaml: Path,
    backend_path: Path,
    parsed_yaml: dict,
    grouped_native_functions: list,
):
    print("Running Lazy Tensor Autogen")

    # No fallback code allowed
    def gen_fallback_code(*args, **kwargs):
        return ""

    codegen.dest.lazy_ir.gen_fallback_code = gen_fallback_code

    codegen.gen_lazy_tensor.run(
        backend_name="TorchMlir",
        source_yaml=str(source_yaml),
        output_dir=str(backend_path),
        dry_run=False,
        impl_path=str(backend_path.joinpath("aten_ltc_mlir_type.cpp")),
        gen_ts_lowerings=False,
        node_base="torch::lazy::MlirNode",
        node_base_hdr=str(backend_path.joinpath("mlir_node.h")),
        tensor_class="torch::lazy::LazyTensor",
        tensor_class_hdr="torch/csrc/lazy/core/tensor.h",
        shape_inference_hdr=str(backend_path.joinpath("LazyShapeInference.h")),
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

    # programmatically check shape inference declarations
    import re

    sig_re = re.compile(
        r"std::vector<Shape>\s+(?P<name>\w+)\((?P<signature>[^\)]+)\)"
    )
    global_signatures = {}

    def extract_signatures(path):
        signatures = set()
        for name, args in sig_re.findall(path.read_text()):
            signature = re.sub(r"\s+", "", f"{name}({args})")
            global_signatures[signature] = (name, args)
            signatures.add(signature)
        return signatures

    upstream_shape_inference_decls = extract_signatures(
        TORCH_DIR.joinpath("torch", "csrc", "lazy", "core", "shape_inference.h")
    )
    assert len(upstream_shape_inference_decls) > 0
    shape_inference_decls = extract_signatures(
        backend_path.joinpath("LazyShapeInference.h")
    )
    assert len(shape_inference_decls) > 0
    shape_inference_defs = extract_signatures(
        backend_path.joinpath("LazyShapeInference.cpp")
    )
    assert len(shape_inference_defs) > 0
    assert len(shape_inference_decls) > len(shape_inference_defs)

    missing_defs = (
        shape_inference_decls
        - upstream_shape_inference_decls
        - shape_inference_defs
    )
    if missing_defs:
        backend_path.joinpath("GenLazyShapeInference.cpp").write_text(
            dedent(
                """
                // This file contains autogenerated Lazy Shape Inference placeholders
                // for ops that dont have a corresponding structured kernel or shape definition

                #include "LazyShapeInference.h"
                #include "../utils/exception.h"
                namespace torch {{
                namespace lazy {{
                {}
                }}  // namespace lazy
                }}  // namespace torch
                """
            ).format(
                "".join(
                    dedent(
                        f"""
                        std::vector<Shape> {name}({args}) {{
                            UNIMPLEMENTED_FUNCTION_ERROR();
                        }}
                        """
                    )
                    for name, args in map(
                        global_signatures.get, sorted(missing_defs)
                    )
                )
            )
        )

    unnecessary_defs = shape_inference_defs - shape_inference_decls
    if unnecessary_defs:
        unnecessary_defs = "\n\t".join(
            f"{name}({args})"
            for name, args in map(global_signatures.get, unnecessary_defs)
        )
        warnings.warn(
            f"Unnecessary shape inference definitions found for:\n\t{unnecessary_defs}"
        )


def main(args):
    script_path = Path(__file__).resolve()
    config_path = (
        Path(__file__).resolve().parent.joinpath("autogen_ltc_backend.yaml")
    )
    torch_ops_file = TORCH_MLIR_DIR.joinpath(
        "include",
        "torch-mlir",
        "Dialect",
        "Torch",
        "IR",
        "GeneratedTorchOps.td",
    )
    assert torch_ops_file.exists()
    native_functions = TORCH_MLIR_DIR.joinpath(
        "generated_native_functions.yaml"
    )
    backend_path = TORCH_MLIR_DIR.joinpath(
        "python", "torch_mlir", "csrc", "backend"
    )
    assert backend_path.is_dir()

    prev_hash = None
    hash_file = TORCH_MLIR_DIR.joinpath("generated_backend.hash")
    if hash_file.exists():
        prev_hash = hash_file.read_text().strip()

    m = hashlib.sha256()
    m.update(script_path.read_bytes())
    m.update(config_path.read_bytes())
    m.update(torch_ops_file.read_bytes())
    if native_functions.exists():
        m.update(native_functions.read_bytes())

    shape_inference_headers = backend_path.joinpath("LazyShapeInference.h")
    if shape_inference_headers.exists():
        m.update(shape_inference_headers.read_bytes())

    shape_inference_defs = backend_path.joinpath("LazyShapeInference.cpp")
    if shape_inference_defs.exists():
        m.update(shape_inference_defs.read_bytes())

    new_hash = m.hexdigest().strip()

    if args.force or new_hash != prev_hash:
        hash_file.write_text(new_hash)
        parsed_yaml, grouped_native_functions = generate_native_functions(
            config_path, torch_ops_file, native_functions
        )

        generate_backend(
            native_functions,
            backend_path,
            parsed_yaml,
            grouped_native_functions,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
    )
    main(parser.parse_args())
