import argparse
import hashlib
import importlib.util
import logging
import os
import re
import subprocess
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from textwrap import dedent, indent

# PyTorch's LTC backend autogen script
import torchgen
import torchgen.dest.lazy_ir
import torchgen.gen_lazy_tensor
import yaml
from torchgen.api.lazy import LazyIrSchema, setValueT
from torchgen.api.types import BaseCppType
from torchgen.dest import GenLazyShapeInferenceDefinition
from torchgen.gen import get_grouped_native_functions, parse_native_yaml
from torchgen.gen_backend_stubs import parse_backend_yaml

TORCH_DIR = Path(importlib.util.find_spec("torch").origin).resolve().parent.parent
TORCH_INCLUDE_DIR = TORCH_DIR.joinpath("torch", "include")
if not TORCH_INCLUDE_DIR.is_dir():
    TORCH_INCLUDE_DIR = TORCH_DIR
TORCHGEN_DIR = Path(torchgen.__path__[0]).resolve()
TORCH_MLIR_DIR = Path(__file__).resolve().parent.parent

# Safely load fast C Yaml loader/dumper if they are available
try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader #type:ignore[assignment, misc]
    
dimsa3-reGdoj-ciqbac
def reindent(text, prefix=""):
    return indent(dedent(text), prefix)


@dataclass(frozen=True)
class GenMlirLazyIr(torchgen.dest.GenLazyIR):
    def isOptionalCType(self, arg):
        return str(type(arg)) == "<class 'torchgen.api.types.types.OptionalCType'>"

    def lowering_function(self, schema: LazyIrSchema):
        signature = "TorchMlirOpVector Lower(TorchMlirFunction function, TorchMlirLoweringContext* loctx) const override"

        if schema.properties.LowerDeclOnly:
            return f"{signature};"
        elif not schema.properties.Lower:
            return ""

        emplace_arguments = []
        for arg in schema.positional_args:
            if arg.is_lazy_value:
                if self.isOptionalCType(arg.lazy_type):
                    emplace_arguments.append(
                        f"has_{arg.name} ? loctx->GetOutputOp(operand(i++)) : nullptr"
                    )
                else:
                    emplace_arguments.append("loctx->GetOutputOp(operand(i++))")
            else:
                emplace_arguments.append(f'"{arg.name}", {arg.name}')

        emplace_arguments_str = "\n        ".join(
            f"arguments.emplace_back({a});" for a in emplace_arguments
        )
        emplace_kwarg_values = [
            f'"{t.name}", loctx->GetOutputOp(operand(i++))'
            for t in schema.keyword_values
        ]
        emplace_kwarg_scalars = [
            f'"{t.name}", {t.name}' for t in schema.keyword_scalars
        ]
        emplace_kwarguments = "\n    ".join(
            f"kwarguments.emplace_back({a});"
            for a in emplace_kwarg_values + emplace_kwarg_scalars
        )

        # Only create this variable if it's used to avoid Wunused-variable
        operand_idx_counter = (
            "size_t i = 0;"
            if "i++" in (emplace_arguments_str + emplace_kwarguments)
            else ""
        )

        return reindent(
            f"""
            {signature} {{
                PRINT_FUNCTION();
                std::vector<torch::jit::NamedValue> arguments;
                std::vector<torch::jit::NamedValue> kwarguments;
                arguments.reserve({len(emplace_arguments)});
                kwarguments.reserve({len(emplace_kwarg_values + emplace_kwarg_scalars)});
                {operand_idx_counter}
                {emplace_arguments_str}
                {emplace_kwarguments}
                torch::lazy::TorchMlirOpVector {schema.aten_name}_out = torch::lazy::LowerTorchMlirBuiltin(function, op().op, shapes(), arguments, kwarguments);
                TORCH_CHECK_EQ({schema.aten_name}_out.size(), {len(schema.returns)});

                return {schema.aten_name}_out;
            }}
            """,
            "  ",
        )


class GenTorchMlirLTC:
    def __init__(self, binary_dir):
        self.script_path = Path(__file__).resolve()
        self.config_path = (
            Path(__file__).resolve().parent.joinpath("autogen_ltc_backend.yaml")
        )
        self.torch_ops_file = TORCH_MLIR_DIR.joinpath(
            # fmt: off
            "include", "torch-mlir", "Dialect", "Torch", "IR", "GeneratedTorchOps.td",
            # fmt: on
        )
        assert self.torch_ops_file.exists()
        self.binary_dir = Path(binary_dir)
        assert (
            self.binary_dir.is_dir()
        ), f"Binary directory not found: {self.binary_dir}"
        self.source_yaml = self.binary_dir.joinpath("generated_native_functions.yaml")
        self.backend_path = TORCH_MLIR_DIR.joinpath(
            "projects", "ltc", "csrc", "base_lazy_backend"
        )
        assert (
            self.backend_path.is_dir()
        ), f"Backend path not found: {self.backend_path}"
        self.generated_path = self.binary_dir.joinpath(
            "projects", "ltc", "csrc", "base_lazy_backend", "generated"
        )
        self.generated_path.mkdir(parents=True, exist_ok=True)

        # Create symlink to match doc structure
        generated_path = self.backend_path.joinpath("generated")
        generated_path.unlink(missing_ok=True)
        generated_path.symlink_to(
            os.path.relpath(self.generated_path, generated_path.parent),
            target_is_directory=True,
        )

        self.tensor_class = "torch::lazy::LazyTensor"

        # Set the lazy value class
        setValueT(BaseCppType("torch::lazy", "Value"))

    def calculate_hash(self):
        m = hashlib.sha256()

        # Add file contents to hash
        for path in (
            self.script_path,
            self.config_path,
            self.torch_ops_file,
            self.source_yaml,
            self.backend_path.joinpath("shape_inference.cpp"),
            TORCHGEN_DIR.joinpath("dest", "lazy_ir.py"),
            TORCHGEN_DIR.joinpath("api", "lazy.py"),
            TORCHGEN_DIR.joinpath("model.py"),
        ):
            if path.exists():
                m.update(path.read_bytes())

        return m.hexdigest().strip()

    def generate_native_functions(self):
        logging.info("Generating Native Functions Yaml")

        native_path = TORCHGEN_DIR.joinpath("packaged", "ATen", "native")
        native_yaml_path = native_path.joinpath("native_functions.yaml")
        tags_yaml_path = native_path.joinpath("tags.yaml")

        ts_native_yaml_path = TORCH_DIR.joinpath(
            "aten", "src", "ATen", "native", "ts_native_functions.yaml"
        )
        ts_native_yaml = None
        if ts_native_yaml_path.exists():
            ts_native_yaml = yaml.load(ts_native_yaml_path.read_text(), Loader)
        else:
            logging.warning(
                f"Could not find `ts_native_functions.yaml` at {ts_native_yaml_path}"
            )

        parsed_yaml = parse_native_yaml(native_yaml_path, tags_yaml_path)
        self.native_functions = parsed_yaml.native_functions
        self.backend_indices = parsed_yaml.backend_indices
        self.grouped_native_functions = get_grouped_native_functions(
            self.native_functions
        )

        def get_native_function_name(f):
            func = f if hasattr(f, "func") else f.functional
            return str(func.func.name)

        self.native_functions = {
            get_native_function_name(f): f for f in self.native_functions
        }

        def get_opnames(ops):
            opnames = defaultdict(set)
            for op in ops:
                opname = op.split(".")[0]
                opnames[opname].add(op)
            return opnames

        aten_funcs = get_opnames(
            map(get_native_function_name, self.grouped_native_functions)
        )

        with self.config_path.open() as f:
            config = yaml.load(f, Loader)

        # List of unsupported ops in LTC autogen because of some error
        blacklist = set(config.get("blacklist", []))

        # List of supported ops that we don't want to do the full codegen for
        # primarily view ops
        supported = set(config.get("supported", []))

        # List of non-native ops to do IR codegen for
        non_native = config.get("non_native", [])

        # use ripgrep if available as its much faster
        if which("rg") is not None:
            cmd = ["rg", "-o", "-N", r"aten::[0-9a-zA-Z_\.]+"]
        else:
            cmd = ["grep", "-o", r"aten::[0-9a-zA-Z_\.]\+"]

        torch_ops = set(
            op[6:]
            for op in subprocess.check_output(
                cmd + [str(self.torch_ops_file)],
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
            if op not in self.native_functions:
                continue

            func = self.native_functions[op]
            base = func.func.name.name.base

            if base in blacklist or op in blacklist:
                continue
            if base in supported or op in supported:
                continue
            # Blacklist new_/_like ops since they are non-differentiable.
            if any(o.startswith("new_") or o.endswith("_like") for o in (base, op)):
                continue

            if func.has_composite_implicit_autograd_kernel:
                composite_implicit.add(op)
            elif func.func.name.name.inplace:
                for autogen in func.autogen:
                    if "functional" in autogen.overload_name:
                        ops.add(str(autogen))
            else:
                ops.add(op)

        skipped = set(torch_ops) - ops - supported - composite_implicit

        # List of ops autogen even if not explicitly supported by Torch-MLIR explicitly
        ops |= set(config.get("whitelist", []))

        # Additional ops to support that are not supported by Torch-MLIR explicitly
        supported |= set(config.get("additional_ops", []))

        # List of ops that will take in symints for its size
        symint = set(config.get("symint", []))

        self.ops = sorted(ops)

        with self.source_yaml.open("w") as f:
            source_yaml = {
                "backend": "Lazy",
                "cpp_namespace": "torch::lazy",
                "full_codegen": self.ops,
                "supported": sorted(supported),
                "symint": sorted(symint),
                "non_native": non_native,
            }
            yaml.dump(source_yaml, f, default_flow_style=False)
            f.write(
                dedent(
                    """

                    # Composite implicit ops (supported by Torch-MLIR but not differentiable)
                    {composite_implicit}
                    # Skipped ops (supported by Torch-MLIR but no equivalent native function)
                    {skipped}
                    """
                ).format(
                    composite_implicit=os.linesep.join(
                        f"#  - {op}" for op in sorted(composite_implicit)
                    ),
                    skipped=os.linesep.join(f"#  - {op}" for op in sorted(skipped)),
                )
            )

        if ts_native_yaml:
            ts_full_codegen = set(ts_native_yaml["full_codegen"])
            ts_supported = set(ts_native_yaml["supported"])
            mlir_full_codegen = set(self.ops)

            if ts_full_codegen - mlir_full_codegen:
                logging.debug(
                    "Full Codegen ops supported by the TorchScript backend "
                    "but not by the Torch-MLIR backend:\n    {}".format(
                        "\n    ".join(sorted(ts_full_codegen - mlir_full_codegen))
                    )
                )

            if mlir_full_codegen - ts_full_codegen:
                logging.debug(
                    "Full Codegen ops supported by the Torch-MLIR backend "
                    "but not by the TorchScript backend:\n    {}".format(
                        "\n    ".join(sorted(mlir_full_codegen - ts_full_codegen))
                    )
                )

            if ts_supported - supported:
                logging.debug(
                    "Ops supported by the TorchScript backend "
                    "but not by the Torch-MLIR backend:\n    {}".format(
                        "\n    ".join(sorted(ts_supported - supported))
                    )
                )

            if supported - ts_supported:
                logging.debug(
                    "Ops supported by the Torch-MLIR backend "
                    "but not by the TorchScript backend:\n    {}".format(
                        "\n    ".join(sorted(supported - ts_supported))
                    )
                )

    def generate_shape_inference(self):
        parsed_backend_yaml = parse_backend_yaml(
            self.source_yaml,
            self.grouped_native_functions,
            self.backend_indices,
        )
        backend_index = self.backend_indices[parsed_backend_yaml.backend_key]

        shape_gen = GenLazyShapeInferenceDefinition(backend_index, self.tensor_class)

        sig_re = re.compile(
            r"std::vector<torch::lazy::Shape>\s+(?P<name>\w+)\((?P<signature>[^\)]+)\)"
        )
        global_signatures = {}

        def extract_signatures(text):
            signatures = set()
            for name, args in sig_re.findall(text):
                # Remove all whitespace from signature
                signature = re.sub(r"\s+", "", f"{name}({args})")
                # Ignore optional's namespace
                signature = re.sub(r":*\w*:*optional", "optional", signature)
                # Remove const type qualifier
                signature = re.sub(r"const", "", signature)
                # Remove type reference
                signature = re.sub(r"&", "", signature)

                global_signatures[signature] = (name, args)
                signatures.add(signature)
            return signatures

        shape_inference_decls = []
        for op in self.ops:
            f = self.native_functions[op]
            shape_sig = shape_gen(f)
            shape_inference_decls.extend(shape_sig)

        self.generated_path.joinpath("shape_inference.h").write_text(
            dedent(
                """
                // This file contains autogenerated Lazy Shape Inference declarations
                // for ops that dont have a corresponding structured kernel or shape definition

                #include <ATen/Tensor.h>
                #include <c10/core/ScalarType.h>
                #include <c10/util/Optional.h>
                #include <torch/csrc/lazy/core/ir.h>
                #include <torch/csrc/lazy/core/shape.h>
                #include <torch/csrc/lazy/core/shape_inference.h>
                #include <vector>

                namespace torch {{
                namespace lazy {{

                {}

                }}  // namespace lazy
                }}  // namespace torch
                """
            ).format(os.linesep.join(sorted(shape_inference_decls)))
        )

        shape_inference_decls = extract_signatures(
            self.generated_path.joinpath("shape_inference.h").read_text()
        )
        assert len(shape_inference_decls) > 0
        upstream_shape_inference_decls = extract_signatures(
            TORCH_INCLUDE_DIR.joinpath(
                "torch", "csrc", "lazy", "core", "shape_inference.h"
            ).read_text()
        )
        assert len(upstream_shape_inference_decls) > 0
        shape_inference_defs = extract_signatures(
            self.backend_path.joinpath("shape_inference.cpp").read_text()
        )
        assert len(shape_inference_decls) > len(shape_inference_defs)

        missing_defs = (
            shape_inference_decls
            - upstream_shape_inference_decls
            - shape_inference_defs
        )
        if missing_defs:
            self.generated_path.joinpath("shape_inference.cpp").write_text(
                dedent(
                    """
                    // This file contains autogenerated Lazy Shape Inference placeholders
                    // for ops that dont have a corresponding structured kernel or shape definition

                    #include "shape_inference.h"
                    #include "base_lazy_backend/utils/exception.h"
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
                            std::vector<torch::lazy::Shape> {name}({args}) {{
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

    def generate_backend(self):
        logging.info("Running Lazy Tensor Autogen")

        # No fallback code allowed
        def gen_fallback_code(*args, **kwargs):
            return ""

        torchgen.dest.lazy_ir.gen_fallback_code = gen_fallback_code

        torchgen.gen_lazy_tensor.run_gen_lazy_tensor(
            backend_name="TorchMlir",
            aten_path=str(TORCHGEN_DIR.joinpath("packaged", "ATen")),
            source_yaml=str(self.source_yaml),
            output_dir=str(self.generated_path),
            dry_run=False,
            impl_path=str(self.backend_path.joinpath("mlir_native_functions.cpp")),
            node_base="torch::lazy::TorchMlirNode",
            node_base_hdr=str(self.backend_path.joinpath("mlir_node.h")),
            tensor_class=self.tensor_class,
            tensor_class_hdr="base_lazy_backend/tensor.h",
            create_aten_from_ltc_tensor="CreateFunctionalizedAtenFromLtcTensor",
            shape_inference_hdr=str(self.generated_path.joinpath("shape_inference.h")),
            lazy_ir_generator=GenMlirLazyIr,
        )

    def __call__(self):
        self.generate_native_functions()
        self.generate_shape_inference()
        self.generate_backend()


def main(args):
    generator = GenTorchMlirLTC(args.binary_dir)

    hash_file = generator.binary_dir.joinpath("generated_backend.hash")

    prev_hash = None
    if hash_file.exists():
        prev_hash = hash_file.read_text().strip()

    new_hash = generator.calculate_hash()

    if args.force or new_hash != prev_hash:
        generator()
        hash_file.write_text(new_hash)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--binary_dir",
        type=str,
        default=os.getenv(
            "TORCH_MLIR_BINARY_DIR",
            TORCH_MLIR_DIR.joinpath("build"),
        ),
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )
    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    main(args)
