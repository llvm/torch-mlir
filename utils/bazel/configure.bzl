# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Helper macros to configure torch-mlir overlay project."""

# This is adapted from llvm-project's utils/bazel/configure.bzl

DEFAULT_OVERLAY_PATH = "torch-mlir-overlay"

def _get_src_path(repository_ctx):
    """Returns the torch-mlir source directory path.

    If src_workspace attr is set, uses that label to find the source directory.
    Otherwise, falls back to finding WORKSPACE in the repository root.
    """
    if repository_ctx.attr.src_workspace:
        return repository_ctx.path(repository_ctx.attr.src_workspace).dirname

    # Fallback for standalone torch-mlir builds
    return repository_ctx.path(Label("//:WORKSPACE.bazel")).dirname

def _overlay_directories(repository_ctx):
    src_path = _get_src_path(repository_ctx)
    bazel_path = src_path.get_child("utils").get_child("bazel")
    overlay_path = bazel_path.get_child("torch-mlir-overlay")
    script_path = bazel_path.get_child("overlay_directories.py")

    python_bin = repository_ctx.which("python3")
    if not python_bin:
        # Windows typically just defines "python" as python3. The script itself
        # contains a check to ensure python3.
        python_bin = repository_ctx.which("python")

    if not python_bin:
        fail("Failed to find python3 binary")

    cmd = [
        python_bin,
        script_path,
        "--src",
        src_path,
        "--overlay",
        overlay_path,
        "--target",
        ".",
    ]
    exec_result = repository_ctx.execute(cmd, timeout = 20)

    if exec_result.return_code != 0:
        fail(("Failed to execute overlay script: '{cmd}'\n" +
              "Exited with code {return_code}\n" +
              "stdout:\n{stdout}\n" +
              "stderr:\n{stderr}\n").format(
            cmd = " ".join([str(arg) for arg in cmd]),
            return_code = exec_result.return_code,
            stdout = exec_result.stdout,
            stderr = exec_result.stderr,
        ))

def _torch_mlir_configure_impl(repository_ctx):
    _overlay_directories(repository_ctx)

torch_mlir_configure = repository_rule(
    implementation = _torch_mlir_configure_impl,
    local = True,
    configure = True,
    attrs = {
        # Label pointing to a file in the torch-mlir source root (e.g., CMakeLists.txt).
        # Used by downstream projects that include torch-mlir as a subdirectory.
        # If not set, defaults to looking for //:WORKSPACE in the current repository.
        "src_workspace": attr.label(
            doc = "Label to a file in torch-mlir source root for path resolution",
        ),
    },
)
