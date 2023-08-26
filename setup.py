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
# The package version can be set with the TORCH_MLIR_PYTHON_PACKAGE_VERSION
# environment variable. For example, this can be "20220330.357" for a snapshot
# release on 2022-03-30 with build number 357.
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
import sysconfig

from distutils.command.build import build as _build
from distutils.sysconfig import get_python_inc
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py


PACKAGE_VERSION = os.environ.get("TORCH_MLIR_PYTHON_PACKAGE_VERSION") or "0.0.1"

# If true, enable LTC build by default
TORCH_MLIR_ENABLE_LTC_DEFAULT = True
TORCH_MLIR_ENABLE_ONLY_MLIR_PYTHON_BINDINGS = int(os.environ.get('TORCH_MLIR_ENABLE_ONLY_MLIR_PYTHON_BINDINGS', False))
if not TORCH_MLIR_ENABLE_ONLY_MLIR_PYTHON_BINDINGS:
    import torch

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
            cmake_build_dir = os.path.abspath(
                os.path.join(target_dir, "..", "cmake_build"))
        python_package_dir = os.path.join(cmake_build_dir,
                                          "tools", "torch-mlir", "python_packages",
                                          "torch_mlir")
        if not os.getenv("TORCH_MLIR_CMAKE_BUILD_DIR_ALREADY_BUILT"):
            src_dir = os.path.abspath(os.path.dirname(__file__))
            llvm_dir = os.path.join(
                src_dir, "externals", "llvm-project", "llvm")

            enable_ltc = int(os.environ.get('TORCH_MLIR_ENABLE_LTC', TORCH_MLIR_ENABLE_LTC_DEFAULT))

            cmake_args = [
                f"-DCMAKE_BUILD_TYPE=Release",
                f"-DPython3_EXECUTABLE={sys.executable}",
                f"-DPython3_FIND_VIRTUALENV=ONLY",
                f"-DLLVM_TARGETS_TO_BUILD=host",
                f"-DMLIR_ENABLE_BINDINGS_PYTHON=ON",
                f"-DLLVM_ENABLE_PROJECTS=mlir",
                f"-DLLVM_ENABLE_ZSTD=OFF",
                f"-DLLVM_EXTERNAL_PROJECTS=torch-mlir;torch-mlir-dialects",
                f"-DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR={src_dir}",
                f"-DLLVM_EXTERNAL_TORCH_MLIR_DIALECTS_SOURCE_DIR={src_dir}/externals/llvm-external-projects/torch-mlir-dialects",
                # Optimization options for building wheels.
                f"-DCMAKE_VISIBILITY_INLINES_HIDDEN=ON",
                f"-DCMAKE_C_VISIBILITY_PRESET=hidden",
                f"-DCMAKE_CXX_VISIBILITY_PRESET=hidden",
                f"-DTORCH_MLIR_ENABLE_LTC={'ON' if enable_ltc else 'OFF'}",
                f"-DTORCH_MLIR_ENABLE_ONLY_MLIR_PYTHON_BINDINGS={'ON' if TORCH_MLIR_ENABLE_ONLY_MLIR_PYTHON_BINDINGS else 'OFF'}",
            ]

            os.makedirs(cmake_build_dir, exist_ok=True)
            cmake_cache_file = os.path.join(cmake_build_dir, "CMakeCache.txt")
            if os.path.exists(cmake_cache_file):
                os.remove(cmake_cache_file)
            # NOTE: With repeated builds for different Python versions, the
            # prior version binaries will continue to accumulate. IREE uses
            # a separate install step and cleans the install directory to
            # keep this from happening. That is the most robust. Here we just
            # delete the directory where we build native extensions to keep
            # this from happening but still take advantage of most of the
            # build cache.
            mlir_libs_dir = os.path.join(python_package_dir, "torch_mlir", "_mlir_libs")
            if os.path.exists(mlir_libs_dir):
                print(f"Removing _mlir_mlibs dir to force rebuild: {mlir_libs_dir}")
                shutil.rmtree(mlir_libs_dir)
            else:
                print(f"Not removing _mlir_libs dir (does not exist): {mlir_libs_dir}")

            subprocess.check_call(["cmake", llvm_dir] +
                                  cmake_args, cwd=cmake_build_dir)
            subprocess.check_call(["cmake",
                                   "--build",  ".",
                                   "--config", "Release",
                                   "--target", "TorchMLIRPythonModules"],
                                  cwd=cmake_build_dir)

        if os.path.exists(target_dir):
            shutil.rmtree(target_dir, ignore_errors=False, onerror=None)

        shutil.copytree(python_package_dir,
                        target_dir,
                        symlinks=False)


class CMakeExtension(Extension):

  def __init__(self, name, sourcedir=""):
    Extension.__init__(self, name, sources=[])
    self.sourcedir = os.path.abspath(sourcedir)


class NoopBuildExtension(build_ext):

    def build_extension(self, ext):
        pass


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="torch-mlir" if not TORCH_MLIR_ENABLE_ONLY_MLIR_PYTHON_BINDINGS else "torch-mlir-core",
    version=f"{PACKAGE_VERSION}",
    author="Sean Silva",
    author_email="silvasean@google.com",
    description="First-class interop between PyTorch and MLIR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    cmdclass={
        "build": CustomBuild,
        "built_ext": NoopBuildExtension,
        "build_py": CMakeBuild,
    },
    ext_modules=[
        CMakeExtension("torch_mlir._mlir_libs._jit_ir_importer"),
    ] if not TORCH_MLIR_ENABLE_ONLY_MLIR_PYTHON_BINDINGS else [CMakeExtension("torch_mlir._mlir_libs._torchMlir")],
    install_requires=["numpy", "packaging"] + (
        [f"torch=={torch.__version__}".split("+", 1)[0], ] if not TORCH_MLIR_ENABLE_ONLY_MLIR_PYTHON_BINDINGS else []),
    zip_safe=False,
)
