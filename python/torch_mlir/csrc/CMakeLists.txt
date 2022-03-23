#-------------------------------------------------------------------------------
# Setup PyTorch/LTC
#-------------------------------------------------------------------------------


list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/python/torch_mlir/cmake/modules")
include(TorchMLIRPyTorch)
TorchMLIRProbeForPyTorchInstall()
find_package(Torch 1.11 REQUIRED)

TorchMLIRConfigurePyTorch()

include_directories(BEFORE
  ${TORCH_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${Python3_INCLUDE_DIRS}
)
link_directories("${TORCH_INSTALL_PREFIX}/lib")


add_library(torch_mlir_ltc_backend SHARED
  backend/aten_eager_fallback.cpp
  backend/aten_ltc_mlir_type.cpp
  backend/backend_impl.cpp
  backend/LazyNativeFunctions.cpp
  backend/LazyShapeInference.cpp
  backend/GenLazyShapeInference.cpp
  backend/mlir_lowering_context.cpp
  backend/mlir_node.cpp
  backend/RegisterLazy.cpp
  tensor_aten_ops.cpp
)

target_link_libraries(torch_mlir_ltc_backend
  TorchMLIRAggregateCAPI
  ${TORCH_LIBRARIES}
  ${Python3_LIBRARIES}
  torch_python
)

message(STATUS "TORCH_CXXFLAGS=${TORCH_CXXFLAGS} -Wno-pedantic")
set_target_properties(torch_mlir_ltc_backend PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${TORCH_MLIR_PYTHON_PACKAGES_DIR}/torch_mlir/"
  OUTPUT_NAME lib_mlir_ltc
  PREFIX ""
  SUFFIX ".so"
  CXX_VISIBILITY_PRESET "hidden"
  COMPILE_FLAGS "${TORCH_CXXFLAGS} -Wno-pedantic"
  LINK_FLAGS "-rdynamic"
)
