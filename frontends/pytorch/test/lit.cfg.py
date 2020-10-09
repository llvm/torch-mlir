# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

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
config.name = 'FRONTENDS_PYTORCH'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.environment['PYTHONPATH'] = os.path.join(config.npcomp_obj_root, "python")
if 'TEST_SRC_PATH' in os.environ:
   config.environment['TEST_SRC_PATH'] = os.environ['TEST_SRC_PATH']

# path to our python operation library
config.environment['TEST_BUILD_PATH'] = os.path.join(config.npcomp_obj_root)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.py']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.npcomp_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))

llvm_config.with_system_environment(
    ['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['lit.cfg.py', 'Inputs', 'Examples', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.npcomp_obj_root, 'test')
config.npcomp_tools_dir = os.path.join(config.npcomp_obj_root, 'bin')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

tool_dirs = [config.npcomp_tools_dir, config.llvm_tools_dir]
tools = [
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
