###########################################################################
# Setup PyTorch
###########################################################################

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/python/torch_mlir/cmake/modules")
include(TorchMLIRPyTorch)
TorchMLIRProbeForPyTorchInstall()
find_package(Torch 1.11 REQUIRED)

TorchMLIRConfigurePyTorch()

###########################################################################
# Setup Python development
###########################################################################

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/external/llvm-project/mlir/cmake/modules")
include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()

###########################################################################
# Library definition
###########################################################################

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

set(REFERENCE_LAZY_BACKEND_CSRC
        backend_impl.cpp
        reference_lazy_backend_pybind.cpp
        )
add_library(reference_lazy_backend SHARED ${REFERENCE_LAZY_BACKEND_CSRC})
add_dependencies(reference_lazy_backend
        torch_mlir_ltc_backend
        )
target_link_libraries(reference_lazy_backend
        ${TORCH_LIBRARIES}
        ${Python3_LIBRARIES}
        torch_python
        torch_mlir_ltc_backend
        )

message(STATUS "TORCH_CXXFLAGS=${TORCH_CXXFLAGS} -Wno-pedantic")
set_target_properties(reference_lazy_backend PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${TORCH_MLIR_PYTHON_PACKAGES_DIR}/torch_mlir/torch_mlir/reference_lazy_backend"
        OUTPUT_NAME _REFERENCE_LAZY_BACKEND
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
        CXX_VISIBILITY_PRESET "hidden"
        COMPILE_FLAGS "${TORCH_CXXFLAGS} -Wno-pedantic"
        )
