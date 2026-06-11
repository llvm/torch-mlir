# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bazel module extension for configuring torch-mlir repositories.

This module extension sets up the necessary repository dependencies for
torch-mlir, including LLVM and StableHLO. When used from the torch-mlir
overlay, it sources these dependencies via git submodules.
"""

load("@bazel_tools//tools/build_defs/repo:local.bzl", "local_repository", "new_local_repository")

def _torch_mlir_repos_extension_impl(module_ctx):
    if any([m.is_root and m.name == "torch-mlir-overlay" for m in module_ctx.modules]):
        # When invoked from the torch-mlir-overlay use LLVM and StableHLO sourced via git submodules
        # LLVM
        new_local_repository(
            name = "llvm-raw",
            build_file_content = "# empty",
            path = "../../externals/llvm-project",
        )

        # StableHLO
        local_repository(
            name = "stablehlo",
            path = "../../externals/stablehlo",
        )

        # Torch-MLIR Raw to allow overlaying Bazel repo over
        new_local_repository(
            name = "torch-mlir-raw",
            build_file_content = "# empty",
            path = "../..",
        )

torch_mlir_repos_extension = module_extension(
    implementation = _torch_mlir_repos_extension_impl,
)
