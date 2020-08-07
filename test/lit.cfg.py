# -*- Python -*-

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
config.name = 'NPCOMP_OPT'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir', '.py']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.npcomp_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))
config.substitutions.append(('%PYTHON', config.python_executable))

llvm_config.with_system_environment(
    ['HOME', 'INCLUDE', 'LIB', 'NPCOMP_DEBUG', 'TMP', 'TEMP'])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = [
    'lit.cfg.py', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt'
]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.npcomp_obj_root, 'test')
config.npcomp_tools_dir = os.path.join(config.npcomp_obj_root, 'tools')
config.npcomp_runtime_shlib = os.path.join(
    config.npcomp_obj_root,
    'lib',
    'libNPCOMPCompilerRuntimeShlib' + config.llvm_shlib_ext
)

# Tweak the PATH and PYTHONPATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)
llvm_config.with_environment('PYTHONPATH', [
    os.path.join(config.npcomp_obj_root, "python")
],
                             append_path=True)

tool_dirs = [
        os.path.join(config.npcomp_tools_dir, 'npcomp-opt'),
        os.path.join(config.npcomp_tools_dir, 'npcomp-run-mlir'),
        config.llvm_tools_dir,
]
tools = [
    'npcomp-opt',
    'npcomp-run-mlir',
    ToolSubst('%npcomp_runtime_shlib', config.npcomp_runtime_shlib),
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
