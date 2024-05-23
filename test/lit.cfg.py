# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "TORCH_MLIR"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir", ".py", ".runlit"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.torch_mlir_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

# llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = [
    "Inputs",
    "Examples",
    "CMakeLists.txt",
    "README.txt",
    "LICENSE.txt",
    "lit.cfg.py",
    "lit.site.cfg.py",
]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.torch_mlir_obj_root, "test")
config.standalone_tools_dir = os.path.join(config.torch_mlir_obj_root, "bin")

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
# Tweak the PATH to include the binary build dir, in order to pick up CAPI tests during out-of-tree.
llvm_config.with_environment(
    "PATH", os.path.join(config.llvm_build_dir, "bin"), append_path=True
)

# On Windows the path to python could contains spaces in which case it needs to
# be provided in quotes.  This is the equivalent of how %python is setup in
# llvm/utils/lit/lit/llvm/config.py.
if "Windows" in config.host_os:
    config.python_executable = '"%s"' % (config.python_executable)

tool_dirs = [
    config.standalone_tools_dir,
    config.llvm_tools_dir,
    config.torch_mlir_obj_root,
]
tools = [
    "torch-mlir-opt",
    ToolSubst("%PYTHON", config.python_executable, unresolved="ignore"),
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

if config.enable_bindings_python:
    llvm_config.with_environment(
        "PYTHONPATH",
        [
            os.path.join(config.torch_mlir_python_packages_dir, "torch_mlir"),
        ],
        append_path=True,
    )
