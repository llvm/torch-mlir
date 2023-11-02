message(STATUS "Building PyTorch1 compatibility project")

if(TORCH_MLIR_ENABLE_LTC)
    set(ENV{TORCH_MLIR_ENABLE_LTC} 1)
    message(STATUS "LTC Backend build is enabled")
else()
    set(ENV{TORCH_MLIR_ENABLE_LTC} 0)
    message(STATUS "LTC Backend build is disabled")
endif()


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/python/torch_mlir/cmake/modules")

################################################################################
# Setup python.
################################################################################

if(MLIR_ENABLE_BINDINGS_PYTHON)
  add_dependencies(check-torch-mlir-all
    check-torch-mlir-pt1
  )
  add_subdirectory(python)
else()
  add_custom_target(TorchMLIRPythonModules)
endif()

add_subdirectory(test)
