# Sharp edge: Torch extensions need to use the same pybind11 that torch
# was compiled with, or else there will be issues in cross module exception
# handling (which will abort instead of raise). We circumvent the possibility
# by forcing the torch directories first.
include_directories(BEFORE
  ${TORCH_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${Python3_INCLUDE_DIRS}
  )
link_directories("${TORCH_INSTALL_PREFIX}/lib")

add_library(NPCOMPTorchMLIRExt SHARED
  builder/acap_dispatch.cpp
  builder/class_annotator.cpp
  builder/debug.cpp
  builder/func_builder.cpp
  builder/function_importer.cpp
  builder/module_builder.cpp
  builder/node_importer.cpp
  builder/op_builder.cpp
  builder/ivalue_importer.cpp
  builder/python_bindings.cpp
  builder/torch_to_mlir_utils.cpp
  init_python_bindings.cpp
  )

target_link_libraries(NPCOMPTorchMLIRExt
  # NPCOMP shared library.
  # TODO: Debug why order matters here (if NPCOMP is included last a large
  # amount of LLVM/MLIR/NPCOMP ends up compiled into this library).
  NPCOMP

  ${TORCH_LIBRARIES}
  ${Python3_LIBRARIES}
  torch_python
)
add_dependencies(NPCOMPTorchMLIRExt
  # Uses of the torch_mlir extension also require the npcomp extension to
  # be built.
  NPCOMPNativePyExt
)

install(TARGETS NPCOMPTorchMLIRExt
LIBRARY DESTINATION python COMPONENT python
ARCHIVE DESTINATION python COMPONENT python
RUNTIME DESTINATION bin COMPONENT python)

message(STATUS "TORCH_CXXFLAGS=${TORCH_CXXFLAGS}")
set_target_properties(NPCOMPTorchMLIRExt PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/python
  OUTPUT_NAME _torch_mlir
  PREFIX "${PYTHON_MODULE_PREFIX}"
  SUFFIX "${PYTHON_MODULE_EXTENSION}"
  CXX_VISIBILITY_PRESET "hidden"
  COMPILE_FLAGS "${TORCH_CXXFLAGS}"
  )

npcomp_python_target_compile_options(NPCOMPTorchMLIRExt)
mlir_check_all_link_libraries(NPCOMPTorchMLIRExt)
