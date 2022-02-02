#-------------------------------------------------------------------------------
# Setup PyTorch/LTC
#-------------------------------------------------------------------------------

# Attempts to find a Torch installation and set the Torch_ROOT variable
# based on introspecting the python environment. This allows a subsequent
# call to find_package(Torch) to work.
function(find_torch_cmake)
    if(Torch_ROOT)
        message(STATUS "Using cached Torch cmake directory: ${Torch_ROOT}")
    else()
        message(STATUS "Checking for PyTorch using ${Python3_EXECUTABLE} ...")
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c
            "import torch; print(torch.utils.cmake_prefix_path, end='')"
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE PYTORCH_STATUS
            OUTPUT_VARIABLE PYTORCH_PACKAGE_DIR)
        if(NOT PYTORCH_STATUS EQUAL "0")
            message(STATUS "Unable to 'import torch' with ${Python3_EXECUTABLE} (fallback to explicit config)")
            return()
        endif()
        message(STATUS "Found PyTorch installation at ${PYTORCH_PACKAGE_DIR}")

        set(Torch_ROOT "${PYTORCH_PACKAGE_DIR}" CACHE STRING
            "Torch configure directory" FORCE)
    endif()
endfunction()

find_torch_cmake()
find_package(Torch 1.11 REQUIRED)

include_directories(BEFORE
  ${TORCH_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${Python3_INCLUDE_DIRS}
)
link_directories("${TORCH_INSTALL_PREFIX}/lib")


add_library(torch_mlir_ltc_backend SHARED
  backend/backend_impl.cc
  backend/mlir_lowering_context.cc
  backend/mlir_node.cc
)

target_link_libraries(torch_mlir_ltc_backend
  TorchMLIRAggregateCAPI
  ${TORCH_LIBRARIES}
  ${Python3_LIBRARIES}
  torch_python
)

message(STATUS "TORCH_CXXFLAGS=${TORCH_CXXFLAGS} -Wpedantic")
set_target_properties(torch_mlir_ltc_backend PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${TORCH_MLIR_PYTHON_PACKAGES_DIR}/torch_mlir/"
  OUTPUT_NAME _MLIR_LTC
  PREFIX "${PYTHON_MODULE_PREFIX}"
  SUFFIX "${PYTHON_MODULE_EXTENSION}"
  CXX_VISIBILITY_PRESET "hidden"
  COMPILE_FLAGS "${TORCH_CXXFLAGS} -Wpedantic"
)
