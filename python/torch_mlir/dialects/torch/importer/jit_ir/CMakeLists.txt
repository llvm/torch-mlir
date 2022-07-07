#-------------------------------------------------------------------------------
# Setup PyTorch
#-------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
include(TorchMLIRPyTorch)
set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../libtorch/share/cmake/Torch")

option(TORCH_MLIR_USE_INSTALLED_PYTORCH "Build from local PyTorch in environment" ON)

if(TORCH_MLIR_USE_INSTALLED_PYTORCH)
  TorchMLIRProbeForPyTorchInstall()
endif()

find_package(Torch 1.11 REQUIRED)

if(TORCH_MLIR_USE_INSTALLED_PYTORCH)
  TorchMLIRConfigurePyTorch()
else()
  TorchMLIRConfigureLibTorch()
endif()

message(STATUS "libtorch_python CXXFLAGS is ...${TORCH_CXXFLAGS}")
#-------------------------------------------------------------------------------
# Subdirectories
#-------------------------------------------------------------------------------

add_subdirectory(csrc)

## Declare the sources of the Python module.

declare_mlir_python_sources(TorchMLIRPythonSources.JitIRImporter
  ROOT_DIR "${TORCH_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT TorchMLIRPythonSources
  SOURCES_GLOB
    dialects/torch/importer/jit_ir/*.py
)
