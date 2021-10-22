# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# Script for generating the torch-mlir wheel.
# ```
# $ python setup.py bdist_wheel
# ```
#
# It is recommended to build with Ninja and ccache. To do so, set environment
# variables by prefixing to above invocations:
# ```
# CMAKE_GENERATOR=Ninja CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache
# ```
#
# On CIs, it is often advantageous to re-use/control the CMake build directory.
# This can be set with the TORCH_MLIR_CMAKE_BUILD_DIR env var.
# Additionally, the TORCH_MLIR_CMAKE_BUILD_DIR_ALREADY_BUILT env var will
# prevent this script from attempting to build the directory, and will simply
# use the (presumed already built) directory as-is.
#
# Implementation notes:
# The contents of the wheel is just the contents of the `python_packages`
# directory that our CMake build produces. We go through quite a bit of effort
# on the CMake side to organize that directory already, so we avoid duplicating
# that here, and just package up its contents.
import os
import shutil
import subprocess
import sys

from distutils.command.build import build as _build
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py


# Build phase discovery is unreliable. Just tell it what phases to run.
class CustomBuild(_build):

    def run(self):
        self.run_command("build_py")
        self.run_command("build_ext")
        self.run_command("build_scripts")


class CMakeBuild(build_py):

    def run(self):
        target_dir = self.build_lib
        cmake_build_dir = os.getenv("TORCH_MLIR_CMAKE_BUILD_DIR")
        if not cmake_build_dir:
            cmake_build_dir = os.path.join(target_dir, "..", "cmake_build")
        if not os.getenv("TORCH_MLIR_CMAKE_BUILD_DIR_ALREADY_BUILT"):
            src_dir = os.path.abspath(os.path.dirname(__file__))
            llvm_dir = os.path.join(
                src_dir, "external", "llvm-project", "llvm")
            cmake_args = [
                f"-DCMAKE_BUILD_TYPE=Release",
                f"-DPython3_EXECUTABLE={sys.executable}",
                f"-DLLVM_TARGETS_TO_BUILD=host",
                f"-DMLIR_ENABLE_BINDINGS_PYTHON=ON",
                f"-DLLVM_ENABLE_PROJECTS=mlir",
                f"-DLLVM_EXTERNAL_PROJECTS=torch-mlir",
                f"-DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR={src_dir}",
                # Optimization options for building wheels.
                f"-DCMAKE_VISIBILITY_INLINES_HIDDEN=ON",
                f"-DCMAKE_C_VISIBILITY_PRESET=hidden",
                f"-DCMAKE_CXX_VISIBILITY_PRESET=hidden",
            ]
            os.makedirs(cmake_build_dir, exist_ok=True)
            cmake_cache_file = os.path.join(cmake_build_dir, "CMakeCache.txt")
            if os.path.exists(cmake_cache_file):
                os.remove(cmake_cache_file)
            subprocess.check_call(["cmake", llvm_dir] +
                                  cmake_args, cwd=cmake_build_dir)
            subprocess.check_call(["cmake",
                                   "--build",  ".",
                                   "--target", "TorchMLIRPythonModules"],
                                  cwd=cmake_build_dir)
        python_package_dir = os.path.join(cmake_build_dir,
                                          "tools", "torch-mlir", "python_packages",
                                          "torch_mlir")
        shutil.copytree(python_package_dir,
                        target_dir,
                        symlinks=False,
                        dirs_exist_ok=True)


class CMakeExtension(Extension):

  def __init__(self, name, sourcedir=""):
    Extension.__init__(self, name, sources=[])
    self.sourcedir = os.path.abspath(sourcedir)


class NoopBuildExtension(build_ext):

    def build_extension(self, ext):
        pass


setup(
    name="torch-mlir",
    version="0.0.1",
    author="Sean Silva",
    author_email="silvasean@google.com",
    description="First-class interop between PyTorch and MLIR",
    long_description="",
    include_package_data=True,
    cmdclass={
        "build": CustomBuild,
        "built_ext": NoopBuildExtension,
        "build_py": CMakeBuild,
    },
    ext_modules=[
        CMakeExtension("torch_mlir._mlir_libs._jit_ir_importer"),
    ],
    zip_safe=False,
)
