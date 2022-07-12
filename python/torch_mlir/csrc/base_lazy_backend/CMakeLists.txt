#-------------------------------------------------------------------------------
# Setup PyTorch/LTC
#-------------------------------------------------------------------------------


list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/python/torch_mlir/cmake/modules")
include(TorchMLIRPyTorch)
TorchMLIRProbeForPyTorchInstall()
find_package(Torch 1.11 REQUIRED)

TorchMLIRConfigurePyTorch()
set(TORCHGEN_DIR ${Torch_ROOT}/../../../torchgen)

include_directories(BEFORE
  ${TORCH_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${Python3_INCLUDE_DIRS}
)
link_directories("${TORCH_INSTALL_PREFIX}/lib")

set(LTC_GENERATED
  generated/LazyNativeFunctions.cpp
  generated/RegisterLazy.cpp
  generated/shape_inference.cpp
)
set(LTC_BACKEND_DEPENDS
  mlir_lowering_context.cpp
  mlir_native_functions.cpp
  mlir_node_lowering.cpp
  shape_inference.cpp
)

# Generate Lazy IR Nodes

add_custom_command(
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/build_tools/autogen_ltc_backend.py
  OUTPUT
  ${CMAKE_BINARY_DIR}/generated_backend.hash
  DEPENDS
  ${PROJECT_SOURCE_DIR}/build_tools/autogen_ltc_backend.py
  ${PROJECT_SOURCE_DIR}/build_tools/autogen_ltc_backend.yaml
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Dialect/Torch/IR/GeneratedTorchOps.td
  ${LTC_BACKEND_DEPENDS}
  ${TORCHGEN_DIR}/gen_backend_stubs.py
  ${TORCHGEN_DIR}/gen_lazy_tensor.py
  ${TORCHGEN_DIR}/api/lazy.py
  ${TORCHGEN_DIR}/dest/lazy_ir.py
  COMMENT "Generating Lazy Tensor Core IR Nodes"
)
add_custom_target(
  torch_mlir_ltc_backend_generated ALL
  DEPENDS
  ${CMAKE_BINARY_DIR}/generated_backend.hash
)

add_library(torch_mlir_ltc_backend SHARED
  ${LTC_GENERATED}
  ${LTC_BACKEND_DEPENDS}
  backend_impl.cpp
  mlir_node.cpp
  ops/device_data.cpp
  ops/generic.cpp
)
target_compile_features(torch_mlir_ltc_backend PRIVATE cxx_std_17)

add_dependencies(torch_mlir_ltc_backend
  TorchMLIRJITIRImporter
  torch_mlir_ltc_backend_generated
)
target_link_libraries(torch_mlir_ltc_backend
  TorchMLIRAggregateCAPI
  TorchMLIRJITIRImporter
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
