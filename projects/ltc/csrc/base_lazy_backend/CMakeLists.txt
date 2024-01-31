#-------------------------------------------------------------------------------
# Setup PyTorch/LTC
#-------------------------------------------------------------------------------

torch_mlir_enable_werror()

set(LTC_GENERATED
  generated/LazyNativeFunctions.cpp
  generated/RegisterLazy.cpp
  generated/shape_inference.cpp
)

# The auto generated files trigger some warnings we can't do anything about.
if(NOT MSVC)
  set_source_files_properties(${LTC_GENERATED}
    PROPERTIES COMPILE_FLAGS "-Wno-sign-compare -Wno-unused-function"
  )
endif()

set(LTC_BACKEND_DEPENDS
  mlir_lowering_context.cpp
  mlir_native_functions.cpp
  mlir_node_lowering.cpp
  shape_inference.cpp
)

# Generate Lazy IR Nodes

add_custom_command(
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/build_tools/autogen_ltc_backend.py -b ${TORCH_MLIR_BINARY_DIR}
  OUTPUT
  ${TORCH_MLIR_BINARY_DIR}/generated_backend.hash
  ${LTC_GENERATED}
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
  ${TORCH_MLIR_BINARY_DIR}/generated_backend.hash
  ${LTC_GENERATED}
)

add_library(torch_mlir_ltc_backend SHARED
  ${LTC_GENERATED}
  ${LTC_BACKEND_DEPENDS}
  backend_impl.cpp
  dynamic_ir.cpp
  mlir_node.cpp
  tensor.cpp
  ops/device_data.cpp
  ops/generic.cpp
  ops/index.cpp
  ops/ivalue.cpp
  ops/split.cpp
  ops/unbind_int.cpp
  utils/jit_utils.cpp
  utils/tensor_utils.cpp
)
target_compile_features(torch_mlir_ltc_backend PRIVATE cxx_std_17)
# Includes are resolved relative to csrc (i.e. #include "base_lazy_backend/...").
# Add both the source and generated include directories.
target_include_directories(torch_mlir_ltc_backend PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_BINARY_DIR}/..
)

add_dependencies(torch_mlir_ltc_backend
  TorchMLIRJITIRImporter
  torch_mlir_ltc_backend_generated
)
target_link_libraries(torch_mlir_ltc_backend
  TorchMLIRAggregateCAPI
  TorchMLIRJITIRImporter
  ${TORCH_LIBRARIES}
)

message(STATUS "TORCH_CXXFLAGS=${TORCH_CXXFLAGS} -Wno-pedantic")
set_target_properties(torch_mlir_ltc_backend PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${TORCH_MLIR_PYTHON_PACKAGES_DIR}/torch_mlir/torch_mlir/_mlir_libs"
  OUTPUT_NAME lib_torch_mlir_ltc
  PREFIX ""
  SUFFIX ".so"
  CXX_VISIBILITY_PRESET "hidden"
  COMPILE_FLAGS "${TORCH_CXXFLAGS} -Wno-pedantic"
  LINK_FLAGS "-rdynamic"
)

# Copy header files into python package

add_custom_command(
        TARGET torch_mlir_ltc_backend POST_BUILD
        COMMAND mkdir -p
                ${TORCH_MLIR_PYTHON_PACKAGES_DIR}/torch_mlir/torch_mlir/base_lazy_backend/generated/)

add_custom_command(
        TARGET torch_mlir_ltc_backend POST_BUILD
        COMMAND cp
                ${PROJECT_SOURCE_DIR}/projects/ltc/csrc/base_lazy_backend/*.h
                ${TORCH_MLIR_PYTHON_PACKAGES_DIR}/torch_mlir/torch_mlir/base_lazy_backend/)

add_custom_command(
        TARGET torch_mlir_ltc_backend POST_BUILD
        COMMAND cp
                ${PROJECT_SOURCE_DIR}/projects/ltc/csrc/base_lazy_backend/generated/*.h
                ${TORCH_MLIR_PYTHON_PACKAGES_DIR}/torch_mlir/torch_mlir/base_lazy_backend/generated/)

add_custom_command(
        TARGET torch_mlir_ltc_backend POST_BUILD
        COMMAND mkdir -p
                ${TORCH_MLIR_PYTHON_PACKAGES_DIR}/torch_mlir/torch_mlir/base_lazy_backend/ops/)

add_custom_command(
        TARGET torch_mlir_ltc_backend POST_BUILD
        COMMAND cp
                ${PROJECT_SOURCE_DIR}/projects/ltc/csrc/base_lazy_backend/ops/*.h
                ${TORCH_MLIR_PYTHON_PACKAGES_DIR}/torch_mlir/torch_mlir/base_lazy_backend/ops/)

add_custom_command(
        TARGET torch_mlir_ltc_backend POST_BUILD
        COMMAND mkdir -p
                ${TORCH_MLIR_PYTHON_PACKAGES_DIR}/torch_mlir/torch_mlir/base_lazy_backend/utils/)

add_custom_command(
        TARGET torch_mlir_ltc_backend POST_BUILD
        COMMAND cp
                ${PROJECT_SOURCE_DIR}/projects/ltc/csrc/base_lazy_backend/utils/*.h
                ${TORCH_MLIR_PYTHON_PACKAGES_DIR}/torch_mlir/torch_mlir/base_lazy_backend/utils/)
