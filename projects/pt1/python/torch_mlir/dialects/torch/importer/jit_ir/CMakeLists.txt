#-------------------------------------------------------------------------------
# Setup PyTorch
#-------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/python/torch_mlir/cmake/modules")
include(TorchMLIRPyTorch)

TorchMLIRProbeForPyTorchInstall()
if(TORCH_MLIR_USE_INSTALLED_PYTORCH)
  TorchMLIRConfigurePyTorch()
else()  
  set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../libtorch/share/cmake/Torch")
endif()

find_package(Torch 1.11 REQUIRED)

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
