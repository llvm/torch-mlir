# TODO: This is copied from frontends/pytorch/csrc/c10_dispatch/CMakeLists.txt
# What is the idiomatic way of sharing this in CMake?
include_directories(
  ${TORCH_INCLUDE_DIRS}
  ${TORCH_INSTALL_PREFIX}/include/TH
  ${TORCH_INSTALL_PREFIX}/include/THC/opt/pytorch/pytorch
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${PYTHON_INCLUDE_DIRS}
  )
link_directories("${TORCH_INSTALL_PREFIX}/lib")


set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  )

add_llvm_tool(mnist-playground
  mnist-playground.cpp
  )
llvm_update_compile_flags(mnist-playground)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
target_link_libraries(mnist-playground PRIVATE
  MLIRAnalysis
  MLIREDSC
  MLIRExecutionEngine
  MLIRIR
  MLIRJitRunner
  MLIRLLVMIR
  MLIRParser
  MLIRTargetLLVMIR
  MLIRSupport
  NPCOMPInitAll
  NPCOMPRefBackendJITHelpers
  ${conversion_libs}
  ${dialect_libs}
  ${TORCH_LIBRARIES}
  )
add_dependencies(mnist-playground
  NPCOMPCompilerRuntimeShlib
  )
