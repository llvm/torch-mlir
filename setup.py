# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# Script for generating the torch-mlir wheel.
# ```
# $ python setup.py bdist_wheel
# ```
# Environment variables you are probably interested in:
#
#   TORCH_MLIR_PYTHON_PACKAGE_VERSION:
#       specify the version of torch-mlir, for example, this can be "20220330.357"
#       for a snapshot release on 2022-03-30 with build number 357.
#
#   TORCH_MLIR_ENABLE_LTC:
#       enables the Lazy Tensor Core Backend
#
#   LLVM_INSTALL_DIR:
#       build the project *out-of-tree* using the built llvm-project
#
#   CMAKE_BUILD_TYPE:
#       specify the build type: DEBUG/RelWithDebInfo/Release
#
#   TORCH_MLIR_CMAKE_BUILD_DIR:
#       specify the cmake build directory
#
#   TORCH_MLIR_CMAKE_ALREADY_BUILT:
#       the `TORCH_MLIR_CMAKE_BUILD_DIR` directory has already been compiled,
#       and the CMake compilation process will not be executed again.
#       On CIs, it is often advantageous to re-use/control the CMake build directory.
#
#   MAX_JOBS:
#       maximum number of compile jobs we should use to compile your code
#
# It is recommended to build with Ninja and ccache. To do so, set environment
# variables by prefixing to above invocations:
# ```
# CMAKE_GENERATOR=Ninja CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache
# ```
#
# Implementation notes:
# The contents of the wheel is just the contents of the `python_packages`
# directory that our CMake build produces. We go through quite a bit of effort
# on the CMake side to organize that directory already, so we avoid duplicating
# that here, and just package up its contents.
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import multiprocessing

from distutils.command.build import build as _build
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py


if "develop" in sys.argv:
    print("Warning: The setup.py script is only used for building the wheel package.")
    print(
        "For initializing the development environment,"
        "please use the cmake commands introduced in the docs/development.md."
    )
    sys.exit(1)


def _check_env_flag(name: str, default=None) -> bool:
    return str(os.getenv(name, default)).upper() in ["ON", "1", "YES", "TRUE", "Y"]


PACKAGE_VERSION = os.getenv("TORCH_MLIR_PYTHON_PACKAGE_VERSION", "0.0.1")

# If true, enable LTC build by default
TORCH_MLIR_ENABLE_LTC = _check_env_flag("TORCH_MLIR_ENABLE_LTC", True)
TORCH_MLIR_ENABLE_ONLY_MLIR_PYTHON_BINDINGS = _check_env_flag(
    "TORCH_MLIR_ENABLE_ONLY_MLIR_PYTHON_BINDINGS", True
)
LLVM_INSTALL_DIR = os.getenv("LLVM_INSTALL_DIR", None)
SRC_DIR = pathlib.Path(__file__).parent.absolute()
CMAKE_BUILD_TYPE = os.getenv("CMAKE_BUILD_TYPE", "Release")

TORCH_MLIR_CMAKE_ALREADY_BUILT = _check_env_flag(
    "TORCH_MLIR_CMAKE_ALREADY_BUILT", False
)
TORCH_MLIR_CMAKE_BUILD_DIR = os.getenv("TORCH_MLIR_CMAKE_BUILD_DIR")
MAX_JOBS = os.getenv("MAX_JOBS", str(multiprocessing.cpu_count()))


# Build phase discovery is unreliable. Just tell it what phases to run.
class CustomBuild(_build):
    def initialize_options(self):
        _build.initialize_options(self)
        # Make setuptools not steal the build directory name,
        # because the mlir c++ developers are quite
        # used to having build/ be for cmake
        self.build_base = "setup_build"

    def run(self):
        self.run_command("build_py")
        self.run_command("build_ext")
        self.run_command("build_scripts")


class CMakeBuild(build_py):
    def cmake_build(self, cmake_build_dir):
        llvm_dir = str(SRC_DIR / "externals" / "llvm-project" / "llvm")

        cmake_config_args = [
            f"cmake",
            f"-DCMAKE_BUILD_TYPE={CMAKE_BUILD_TYPE}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DPython3_FIND_VIRTUALENV=ONLY",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DPython_FIND_VIRTUALENV=ONLY",
            f"-DMLIR_ENABLE_BINDINGS_PYTHON=ON",
            f"-DLLVM_TARGETS_TO_BUILD=host",
            f"-DLLVM_ENABLE_ZSTD=OFF",
            # Optimization options for building wheels.
            f"-DCMAKE_VISIBILITY_INLINES_HIDDEN=ON",
            f"-DCMAKE_C_VISIBILITY_PRESET=hidden",
            f"-DCMAKE_CXX_VISIBILITY_PRESET=hidden",
            f"-DTORCH_MLIR_ENABLE_LTC={'ON' if TORCH_MLIR_ENABLE_LTC else 'OFF'}",
            f"-DTORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS={'OFF' if TORCH_MLIR_ENABLE_ONLY_MLIR_PYTHON_BINDINGS else 'ON'}",
        ]
        if LLVM_INSTALL_DIR:
            cmake_config_args += [
                f"-DMLIR_DIR='{LLVM_INSTALL_DIR}/lib/cmake/mlir/'",
                f"-DLLVM_DIR='{LLVM_INSTALL_DIR}/lib/cmake/llvm/'",
                f"{SRC_DIR}",
            ]
        else:
            cmake_config_args += [
                f"-DLLVM_ENABLE_PROJECTS=mlir",
                f"-DLLVM_EXTERNAL_PROJECTS='torch-mlir'",
                f"-DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR={SRC_DIR}",
                f"{llvm_dir}",
            ]
        cmake_build_args = [
            f"cmake",
            f"--build",
            f".",
            f"--config",
            f"{CMAKE_BUILD_TYPE}",
            f"--target",
            f"TorchMLIRPythonModules",
            f"--",
            f"-j{MAX_JOBS}",
        ]
        try:
            subprocess.check_call(cmake_config_args, cwd=cmake_build_dir)
            subprocess.check_call(cmake_build_args, cwd=cmake_build_dir)
        except subprocess.CalledProcessError as e:
            print("cmake build failed with\n", e)
            print("debug by follow cmake command:")
            sys.exit(e.returncode)
        finally:
            print(f"cmake config: {' '.join(cmake_config_args)}")
            print(f"cmake build: {' '.join(cmake_build_args)}")
            print(f"cmake workspace: {cmake_build_dir}")

    def run(self):
        target_dir = self.build_lib
        cmake_build_dir = TORCH_MLIR_CMAKE_BUILD_DIR
        if not cmake_build_dir:
            cmake_build_dir = os.path.abspath(
                os.path.join(target_dir, "..", "cmake_build")
            )
        if LLVM_INSTALL_DIR:
            python_package_dir = os.path.join(
                cmake_build_dir, "python_packages", "torch_mlir"
            )
        else:
            python_package_dir = os.path.join(
                cmake_build_dir, "tools", "torch-mlir", "python_packages", "torch_mlir"
            )
        if not TORCH_MLIR_CMAKE_ALREADY_BUILT:
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
            self.cmake_build(cmake_build_dir)

        if os.path.exists(target_dir):
            shutil.rmtree(target_dir, ignore_errors=False, onerror=None)

        python_package_dir = os.path.join(python_package_dir, "torch_mlir")
        shutil.copytree(
            python_package_dir, os.path.join(target_dir, "torch_mlir"), symlinks=False
        )

        torch_mlir_opt_src = os.path.join(cmake_build_dir, "bin", "torch-mlir-opt")
        torch_mlir_opt_dst = os.path.join(
            target_dir, "torch_mlir", "_mlir_libs", "torch-mlir-opt"
        )
        if platform.system() == "Windows":
            torch_mlir_opt_src += ".exe"
            torch_mlir_opt_dst += ".exe"
        shutil.copy2(torch_mlir_opt_src, torch_mlir_opt_dst, follow_symlinks=False)


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class NoopBuildExtension(build_ext):
    def build_extension(self, ext):
        pass


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# Requires and extension modules depend on whether building PyTorch
# extensions.
INSTALL_REQUIRES = [
    "numpy",
    "packaging",
]
EXT_MODULES = [
    CMakeExtension("torch_mlir._mlir_libs._torchMlir"),
]
NAME = "torch-mlir"

# If building PyTorch extensions, customize.
if not TORCH_MLIR_ENABLE_ONLY_MLIR_PYTHON_BINDINGS:
    import torch

    NAME = "torch-mlir-ext"
    INSTALL_REQUIRES.extend(
        [
            f"torch=={torch.__version__}".split("+", 1)[0],
        ]
    )
    EXT_MODULES.extend(
        [
            CMakeExtension("torch_mlir._mlir_libs._jit_ir_importer"),
        ]
    )


setup(
    name=NAME,
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
    ext_modules=EXT_MODULES,
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "onnx": [
            "onnx>=1.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "torch-mlir-import-onnx = torch_mlir.tools.import_onnx.__main__:_cli_main",
            "torch-mlir-opt = torch_mlir.tools.opt.__main__:main",
        ],
    },
    zip_safe=False,
)
