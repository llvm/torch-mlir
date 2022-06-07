import argparse
import hashlib
import importlib
import os
import subprocess
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from textwrap import dedent

import yaml

# PyTorch's LTC backend autogen script
import torchgen
import torchgen.dest.lazy_ir
import torchgen.gen_lazy_tensor
from torchgen.api.lazy import LazyIrSchema
from torchgen.gen import get_grouped_native_functions, parse_native_yaml
from torchgen.model import NativeFunctionsGroup

TORCH_DIR = Path(importlib.util.find_spec('torch').origin).resolve().parent.parent
TORCH_MLIR_DIR = Path(__file__).resolve().parent.parent

def isOptionalCType(arg):
    return str(type(arg)) == "<class 'torchgen.api.types.OptionalCType'>"


def generate_native_functions(
    config_path: Path, torch_ops_file: Path, out_file: Path
):
    print("Generating Native Functions Yaml")

    native_path = TORCH_DIR.joinpath("aten", "src", "ATen", "native")
    native_yaml_path = native_path.joinpath("native_functions.yaml")
    tags_yaml_path = native_path.joinpath("tags.yaml")

    parsed_yaml = parse_native_yaml(native_yaml_path, tags_yaml_path)
    native_functions = parsed_yaml.native_functions
    grouped_native_functions = get_grouped_native_functions(native_functions)

    def get_native_function_name(f):
        func = f if hasattr(f, "func") else f.functional
        return str(func.func.name), func

    def get_opnames(ops):
        opnames = defaultdict(set)
        for op in ops:
            opname = op.split(".")[0]
            opnames[opname].add(op)
        return opnames

    native_functions = dict(map(get_native_function_name, native_functions))
    grouped_native_functions = dict(map(get_native_function_name, grouped_native_functions))
    aten_funcs = get_opnames(set(grouped_native_functions.keys()))

    with config_path.open() as f:
        config = yaml.load(f, yaml.CLoader)

    # List of unsupported ops in LTC autogen because of some error
    blacklist = set(config.get("blacklist", []))

    # List of supported ops that we don't want to do the full codegen for
    # primarily view ops
    supported = set(config.get("supported", []))

    # List of non-native ops to do IR codegen for
    non_native = config.get("non_native", [])

    if which("rg") is not None:  # use ripgrep if available as its much faster
        cmd = ["rg", "-o", "-N", r"aten::[0-9a-zA-Z_\.]+"]
    else:
        cmd = ["grep", "-o", r"aten::[0-9a-zA-Z_\.]\+"]

    torch_ops = set(
        op[6:]
        for op in subprocess.check_output(
            cmd + [str(torch_ops_file)],
            encoding="utf-8",
        )
        .strip()
        .split(os.linesep)
    )
    torch_opnames = get_opnames(torch_ops)

    # process ops list
    ops = set()
    composite_implicit = set()

    for op in torch_ops:
        if op not in native_functions:
            continue

        func = native_functions[op]
        base = func.func.name.name.base

        if base in blacklist or op in blacklist:
            continue
        if base in supported or op in supported:
            continue

        if func.has_composite_implicit_autograd_kernel and f"{op}_backward" not in torch_ops:
            composite_implicit.add(op)
        elif func.func.name.name.inplace:
            for autogen in func.autogen:
                if "functional" in autogen.overload_name:
                    ops.add(str(autogen))
        else:
            ops.add(op)

    skipped = set(torch_ops) - ops - supported - composite_implicit

    # Additional ops to support that are not supported by Torch-MLIR explicitly
    supported |= set(config.get("additional_ops", []))

    with out_file.open("w") as f:
        yaml.dump(
            {
                "backend": "Lazy",
                "cpp_namespace": "torch::lazy",
                "full_codegen": sorted(ops),
                "supported": sorted(supported),
                "non_native": non_native,
            },
            f,
            default_flow_style=False,
        )
        f.write(
            dedent(
                """

                # Composite implicit ops (supported by Torch-MLIR but not differentiable)
                {composite_implicit}
                # Skipped ops (supported by Torch-MLIR but no equivalent native function)
                {skipped}
                """
            ).format(
                composite_implicit=os.linesep.join(f"#  - {op}" for op in sorted(composite_implicit)),
                skipped=os.linesep.join(f"#  - {op}" for op in sorted(skipped)),
            )
        )

    return parsed_yaml, grouped_native_functions


@dataclass(frozen=True)
class GenMlirLazyIr(torchgen.dest.GenLazyIR):

    def lowering_function(self, schema):
        signature = "TorchMlirOpVector Lower(TorchMlirFunction function, TorchMlirLoweringContext* loctx) const override"

        if schema.properties.LowerDeclOnly:
            return f"{signature};"
        elif not schema.properties.Lower:
            return ""

        emplace_arguments = []
        for arg in schema.positional_args:
            if arg.is_lazy_value:
                if isOptionalCType(arg.lazy_type):
                    emplace_arguments.append(f"has_{arg.name} ? loctx->GetOutputOp(operand(i++)) : nullptr")
                    continue
                emplace_arguments.append('loctx->GetOutputOp(operand(i++))')
                continue
            emplace_arguments.append(f'"{arg.name}", {arg.name}')

        emplace_arguments_str = "\n        ".join(
            [f"arguments.emplace_back({a});" for a in emplace_arguments])
        emplace_kwarg_values = [f'"{t.name}", loctx->GetOutputOp(operand(i++))' for t in schema.keyword_values]
        emplace_kwarg_scalars = [f'"{t.name}", {t.name}' for t in schema.keyword_scalars]
        emplace_kwarguments = "\n    ".join(
            [f"kwarguments.emplace_back({a});" for a in emplace_kwarg_values + emplace_kwarg_scalars])

        return f"""
  {signature} {{
    PRINT_FUNCTION();
    std::vector<torch::jit::NamedValue> arguments;
    std::vector<torch::jit::NamedValue> kwarguments;
    arguments.reserve({len(emplace_arguments)});
    kwarguments.reserve({len(emplace_kwarg_values + emplace_kwarg_scalars)});
    size_t i = 0;
    {emplace_arguments_str}
    {emplace_kwarguments}
    torch::lazy::TorchMlirOpVector {schema.aten_name}_out = torch::lazy::LowerTorchMlirBuiltin(function, op().op, shapes(), arguments, kwarguments);
    CHECK_EQ({schema.aten_name}_out.size(), {len(schema.returns)});
  
    return {schema.aten_name}_out;
  }}
        """.strip()


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

    torchgen.dest.lazy_ir.gen_fallback_code = gen_fallback_code

    torchgen.gen_lazy_tensor.run_gen_lazy_tensor(
        backend_name="TorchMlir",
        aten_path=str(TORCH_DIR.joinpath("aten", "src", "ATen")),
        source_yaml=str(source_yaml),
        output_dir=str(backend_path.joinpath("generated")),
        dry_run=False,
        impl_path=str(backend_path.joinpath("mlir_native_functions.cpp")),
        node_base="torch::lazy::TorchMlirNode",
        node_base_hdr=str(backend_path.joinpath("mlir_node.h")),
        tensor_class="torch::lazy::LazyTensor",
        tensor_class_hdr="torch/csrc/lazy/core/tensor.h",
        shape_inference_hdr=str(backend_path.joinpath("LazyShapeInference.h")),
        lazy_ir_generator=GenMlirLazyIr,
    )

    # Remove lazy_tensor_core imports
    subprocess.check_call(
        [
            "sed",
            "-i",
            "/lazy_tensor_core/d",
            str(backend_path.joinpath("generated", "LazyNativeFunctions.cpp")),
        ]
    )

    # programmatically check shape inference declarations
    import re

    sig_re = re.compile(
        r"std::vector<torch::lazy::Shape>\s+(?P<name>\w+)\((?P<signature>[^\)]+)\)"
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
    assert len(shape_inference_decls) > len(shape_inference_defs)

    missing_defs = (
        shape_inference_decls
        - upstream_shape_inference_decls
        - shape_inference_defs
    )
    if missing_defs:
        backend_path.joinpath("generated", "GenLazyShapeInference.cpp").write_text(
            dedent(
                """
                // This file contains autogenerated Lazy Shape Inference placeholders
                // for ops that dont have a corresponding structured kernel or shape definition

                #include "../LazyShapeInference.h"
                #include "../../utils/exception.h"
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
        "python", "torch_mlir", "csrc", "base_lazy_backend"
    )
    assert backend_path.is_dir()

    torchgen_path = Path(torchgen.__path__[0]).resolve()
    assert torchgen_path.is_dir()

    prev_hash = None
    hash_file = TORCH_MLIR_DIR.joinpath("generated_backend.hash")
    if hash_file.exists():
        prev_hash = hash_file.read_text().strip()

    m = hashlib.sha256()

    # Add file contents to hash
    for path in (
        script_path,
        config_path,
        torch_ops_file,
        native_functions,
        backend_path.joinpath("LazyShapeInference.h"),
        backend_path.joinpath("LazyShapeInference.cpp"),
        torchgen_path.joinpath("dest", "lazy_ir.py"),
        torchgen_path.joinpath("api", "lazy.py"),
        torchgen_path.joinpath("model.py"),
    ):
        if path.exists():
            m.update(path.read_bytes())

    new_hash = m.hexdigest().strip()

    if args.force or new_hash != prev_hash:
        parsed_yaml, grouped_native_functions = generate_native_functions(
            config_path, torch_ops_file, native_functions
        )

        generate_backend(
            native_functions,
            backend_path,
            parsed_yaml,
            grouped_native_functions,
        )

        hash_file.write_text(new_hash)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
    )
    main(parser.parse_args())
