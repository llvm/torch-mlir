include(AddMLIRPython)

# Configure PyTorch if we have any features enabled which require it.
if(TORCH_MLIR_ENABLE_JIT_IR_IMPORTER OR TORCH_MLIR_ENABLE_LTC)
  message(STATUS "Enabling PyTorch C++ dep (features depend on it)")
  include(TorchMLIRPyTorch)

  TorchMLIRProbeForPyTorchInstall()
  if(TORCH_MLIR_USE_INSTALLED_PYTORCH)
    TorchMLIRConfigurePyTorch()
  else()
    # Assume it is a sibling to the overall project.
    set(Torch_DIR "${PROJECT_SOURCE_DIR}/../libtorch/share/cmake/Torch")
    message(STATUS "Attempting to locate libtorch as a sibling to the project: ${Torch_DIR}")
    if(NOT EXISTS "${Torch_DIR}")
      message(FATAL_ERROR "Without TORCH_MLIR_USE_INSTALLED_PYTORCH, expected to find Torch configuration at ${Torch_DIR}, which does not exist")
    endif()
  endif()
  
  find_package(Torch 1.11 REQUIRED)
  
  set(TORCHGEN_DIR ${Torch_ROOT}/../../../torchgen)
  
  include_directories(BEFORE
    ${TORCH_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
  )
  link_directories("${TORCH_INSTALL_PREFIX}/lib")  
  message(STATUS "TORCH_CXXFLAGS is = ${TORCH_CXXFLAGS}")
  if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux" AND NOT TORCH_CXXFLAGS)
    message(WARNING
      "When building on Linux TORCH_CXXFLAGS are almost always required but were not detected. "
      "It is very likely this this will produce a non-functional installation. "
      "See notes in build_tools/cmake/TorchMLIRPyTorch.cmake.")
  endif()
  message(STATUS "TORCH_LIBRARIES = ${TORCH_LIBRARIES}")
endif()

# Include jit_ir_common if the jit_ir importer or LTC is enabled,
# since they both require it.
if(TORCH_MLIR_ENABLE_JIT_IR_IMPORTER OR TORCH_MLIR_ENABLE_LTC)
  add_subdirectory(jit_ir_common)
endif()

# Include LTC.
if(TORCH_MLIR_ENABLE_LTC)
  add_subdirectory(ltc)
endif()

# Include overall PT1 project.
if(TORCH_MLIR_ENABLE_PROJECT_PT1)
  add_subdirectory(pt1)
endif()
