# Build/install the npcomp-core python package.
# Note that this includes a relatively large build of LLVM (~2400 C++ files)
# and can take a considerable amount of time, especially with defaults.
# To install:
#   pip install . --use-feature=in-tree-build
# To build a wheel:
#   pip wheel . --use-feature=in-tree-build
#
# It is recommended to build with Ninja and ccache. To do so, set environment
# variables by prefixing to above invocations:
#   CMAKE_GENERATOR=Ninja CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache
#
# On CIs, it is often advantageous to re-use/control the CMake build directory.
# This can be set with the NPCOMP_CMAKE_BUILD_DIR env var.
import os
import shutil
import subprocess
import sys

from setuptools import find_packages, setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py


class CMakeExtension(Extension):

  def __init__(self, name, sourcedir=""):
    Extension.__init__(self, name, sources=[])
    self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_py):

  def run(self):
    target_dir = self.build_lib
    cmake_build_dir = os.getenv("NPCOMP_CMAKE_BUILD_DIR")
    if not cmake_build_dir:
      cmake_build_dir = os.path.join(target_dir, "..", "cmake_build")
    cmake_install_dir = os.path.join(target_dir, "..", "cmake_install")
    src_dir = os.path.abspath(os.path.dirname(__file__))
    cfg = "Release"
    cmake_args = [
        "-DCMAKE_INSTALL_PREFIX={}".format(os.path.abspath(cmake_install_dir)),
        "-DPython3_EXECUTABLE={}".format(sys.executable),
        "-DNPCOMP_VERSION_INFO={}".format(self.distribution.get_version()),
        "-DCMAKE_BUILD_TYPE={}".format(cfg),  # not used on MSVC, but no harm
        "-DLLVM_TARGETS_TO_BUILD=host",
    ]
    build_args = []
    os.makedirs(cmake_build_dir, exist_ok=True)
    if os.path.exists(cmake_install_dir):
      shutil.rmtree(cmake_install_dir)
    cmake_cache_file = os.path.join(cmake_build_dir, "CMakeCache.txt")
    if os.path.exists(cmake_cache_file):
      os.remove(cmake_cache_file)
    subprocess.check_call(["cmake", src_dir] + cmake_args, cwd=cmake_build_dir)
    subprocess.check_call(["cmake", "--build", ".", "--target", "install"] +
                          build_args,
                          cwd=cmake_build_dir)
    shutil.copytree(os.path.join(cmake_install_dir, "python_packages",
                                 "npcomp_core"),
                    target_dir,
                    symlinks=False,
                    dirs_exist_ok=True)


class NoopBuildExtension(build_ext):

  def build_extension(self, ext):
    pass


setup(
    name="npcomp-core",
    version="0.0.1",
    author="Stella Laurenzo",
    author_email="stellaraccident@gmail.com",
    description="NPComp Core",
    long_description="",
    ext_modules=[
        CMakeExtension("mlir._mlir_libs._mlir"),
        CMakeExtension("mlir._mlir_libs._npcomp"),
        # TODO: We don't really want these but they are along for the ride.
        CMakeExtension("mlir._mlir_libs._mlirAsyncPasses"),
        CMakeExtension("mlir._mlir_libs._mlirConversions"),
        CMakeExtension("mlir._mlir_libs._mlirTransforms"),
        CMakeExtension("mlir._mlir_libs._mlirSparseTensorPasses"),
        CMakeExtension("mlir._mlir_libs._mlirAllPassesRegisration"),
        CMakeExtension("mlir._mlir_libs._mlirLinalgPasses"),
        CMakeExtension("mlir._mlir_libs._mlirGPUPasses"),
    ],
    cmdclass={
        "built_ext": NoopBuildExtension,
        "build_py": CMakeBuild,
    },
    zip_safe=False,
    packages=find_packages(),
)
