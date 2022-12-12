if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # The cmake configuration to build torch-mlir-dialects as
  # an out-of-tree project has not been implemented. It can
  # be built as part of LLVM or as a subdirectory of torch-mlir.
  message(FATAL_ERROR
    "This project is intended to be built as part of LLVM via "
    "-DLLVM_EXTERNAL_PROJECTS=torch-mlir-dialects "
    "-DLLVM_EXTERNAL_TORCH_MLIR_DIALECTS_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}")
endif()

option(MLIR_ENABLE_BINDINGS_PYTHON "Enables MLIR Python Bindings" OFF)

if(TORCH_MLIR_ENABLE_TCP)
  option(TORCH_MLIR_DIALECTS_ENABLE_TCP "Add TCP dialect" ON)
  add_definitions(-DTORCH_MLIR_DIALECTS_ENABLE_TCP)
else()
  option(TORCH_MLIR_DIALECTS_ENABLE_TCP "Add TCP dialect" OFF)
endif()

set(TORCH_MLIR_DIALECTS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TORCH_MLIR_DIALECTS_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "Building torch-mlir-dialects project at ${TORCH_MLIR_DIALECTS_SOURCE_DIR} (into ${TORCH_MLIR_DIALECTS_BINARY_DIR})")

if(MLIR_FOUND)
  message(STATUS "LLVM and MLIR packages have already been configured.")
else()
  message(STATUS "torch-mlir-dialect is being built in-tree")
  # An in-tree build, LLVM hasn't been installed yet.
  # We compute these properties manually, they're otherwise
  # contributed by find_package(MLIR...)
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
  set(MLIR_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/../mlir/include)
  set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(MLIR_INCLUDE_DIRS ${MLIR_INCLUDE_DIR} ${MLIR_GENERATED_INCLUDE_DIR})
  set(LLVM_INCLUDE_DIRS ${LLVM_MAIN_INCLUDE_DIR})

  # Configure CMake and tablegen.
  list(APPEND CMAKE_MODULE_PATH ${MLIR_MAIN_SRC_DIR}/cmake/modules)
  list(APPEND CMAKE_MODULE_PATH ${LLVM_MAIN_SRC_DIR}/cmake)

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
endif()

function(torch_mlir_dialects_target_includes target)
  set(_dirs
    $<BUILD_INTERFACE:${MLIR_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${TORCH_MLIR_DIALECTS_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${TORCH_MLIR_DIALECTS_BINARY_DIR}/include>
  )
  # In LLVM parlance, the actual target may just be an interface and may not
  # be responsible for actually compiling anything. The corresponding obj.
  # target, when present, is just used for compilation and does not
  # contribute to the interface properties.
  # TODO: Normalize this upstream.
  target_include_directories(${target} PUBLIC ${_dirs})
  if(TARGET obj.${target})
    target_include_directories(obj.${target} PRIVATE ${_dirs})
  endif()
endfunction()

# TODO: Needed for tablegen. Remove.
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${TORCH_MLIR_DIALECTS_SOURCE_DIR}/include)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)
