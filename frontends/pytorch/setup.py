# Build/install the npcomp-torch package.
# This uses PyTorch's setuptools support and requires an existing installation
# of npcomp-core in order to access its headers/libraries.

from pathlib import Path

from setuptools import find_packages, setup, Extension
from torch.utils import cpp_extension

try:
  from npcomp import build as npcomp_build
except ModuleNotFoundError:
  raise ModuleNotFoundError(
    f"Could not import npcomp.build "
    f"(do you have the npcomp-core package installed)")

# Get our sources.
this_dir = Path(__file__).parent
extension_sources = [str(p) for p in this_dir.joinpath("csrc").rglob("*.cpp")]

# Npcomp bits.
include_dirs = npcomp_build.get_include_dirs()
lib_dirs = npcomp_build.get_lib_dirs()
npcomp_libs = [npcomp_build.get_capi_link_library_name()]

setup(
  name="npcomp-torch",
  ext_modules=[
    cpp_extension.CppExtension(
      name="_torch_mlir",
      sources=extension_sources,
      include_dirs=include_dirs,
      library_dirs=lib_dirs,
      libraries=npcomp_libs),
  ],
  cmdclass={
    "build_ext": cpp_extension.BuildExtension
  },
  package_dir={
    "": "./python",
  },
  packages=find_packages("./python", include=[
    "torch_mlir",
    "torch_mlir.*",
    "torch_mlir_torchscript",
    "torch_mlir_torchscript.*",
    "torch_mlir_torchscript_e2e_test_configs",
    "torch_mlir_torchscript_e2e_test_configs.*",
  ]),
)
