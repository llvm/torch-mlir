###########################################################################
# Setup PyTorch
###########################################################################

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/python/torch_mlir/cmake/modules")
include(TorchMLIRPyTorch)

TorchMLIRProbeForPyTorchInstall()
if(TORCH_MLIR_USE_INSTALLED_PYTORCH)
  TorchMLIRConfigurePyTorch()
else()
  set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../libtorch/share/cmake/Torch")
endif()

find_package(Torch 1.11 REQUIRED)

###########################################################################
# Setup Python development
###########################################################################

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/externals/llvm-project/mlir/cmake/modules")
include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()

###########################################################################
# Library definition
###########################################################################

# We piggyback on the shared library setup/infra used by Torch-MLIR Python bindings for consistency.
# https://github.com/llvm/torch-mlir/pull/1283
set(LIBRARY_OUTPUT_PATH  "${TORCH_MLIR_PYTHON_PACKAGES_DIR}/torch_mlir/torch_mlir/_mlir_libs")
set(OUTPUT_NAME "_REFERENCE_LAZY_BACKEND")

if(TORCH_MLIR_ENABLE_LTC)
  include_directories(BEFORE
          ${TORCH_INCLUDE_DIRS}
          ${CMAKE_CURRENT_SOURCE_DIR}
          ${CMAKE_CURRENT_BINARY_DIR}
          ${Python3_INCLUDE_DIRS}
          ${PYTHON_H_DIR}
          ${PROJECT_SOURCE_DIR}/python
          )
  link_directories("${TORCH_INSTALL_PREFIX}/lib")
  link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)
  add_link_options(-Wl,-rpath,$ORIGIN/lib)

  add_library(reference_lazy_backend MODULE
          backend_impl.cpp
          reference_lazy_backend_pybind.cpp
          )
  add_dependencies(reference_lazy_backend
          torch_mlir_ltc_backend
          )
  target_link_libraries(reference_lazy_backend
          ${TORCH_LIBRARIES}
          torch_python
          torch_mlir_ltc_backend
          )

  message(STATUS "TORCH_CXXFLAGS=${TORCH_CXXFLAGS} -Wno-pedantic")
  set_target_properties(reference_lazy_backend PROPERTIES
          LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
          OUTPUT_NAME ${OUTPUT_NAME}
          PREFIX "${PYTHON_MODULE_PREFIX}"
          SUFFIX "${PYTHON_MODULE_EXTENSION}"
          CXX_VISIBILITY_PRESET "hidden"
          COMPILE_FLAGS "${TORCH_CXXFLAGS} -Wno-pedantic"
          )
  mlir_python_setup_extension_rpath(reference_lazy_backend)

  torch_mlir_python_target_compile_options(reference_lazy_backend)
  mlir_check_all_link_libraries(reference_lazy_backend)
else()
  # To avoid import errors when LTC is disabled (and a bunch of checks
  # associated with that), we will generate a dummy placeholder library.
  execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen_dummy_lib.py ${LIBRARY_OUTPUT_PATH} ${OUTPUT_NAME}
  )
endif()
