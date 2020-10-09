if(NPCOMP_ENABLE_TORCH_TYPE_DISPATCH)
  add_subdirectory(type_dispatch)
else()
  add_subdirectory(c10_dispatch)
endif()

include(NpcompPython)

include_directories(
  ${TORCH_INCLUDE_DIRS}
  ${TORCH_INSTALL_PREFIX}/include/TH
  ${TORCH_INSTALL_PREFIX}/include/THC/opt/pytorch/pytorch
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${PYTHON_INCLUDE_DIRS}
  )
link_directories("${TORCH_INSTALL_PREFIX}/lib")

set(torch_mlir_optional_libraries)
if(NPCOMP_ENABLE_TORCH_TYPE_DISPATCH)
  list(APPEND torch_mlir_optional_libraries
    npcomp_torch_type_dispatch_bindings
    )
else()
  list(APPEND torch_mlir_optional_libraries
    npcomp_torch_c10_dispatch_bindings
  )
endif()

add_library(NPCOMPTorchMLIRExt SHARED
  init_python_bindings.cpp
  )
target_link_libraries(NPCOMPTorchMLIRExt
  ${TORCH_LIBRARIES}
  ${PYTHON_LIBRARIES}
  ${torch_mlir_optional_libraries}
  torch_python

  # NPCOMP shared library.
  NPCOMP
)

set_target_properties(NPCOMPTorchMLIRExt PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/python
  OUTPUT_NAME _torch_mlir
  PREFIX "${PYTHON_MODULE_PREFIX}"
  SUFFIX "${PYTHON_MODULE_EXTENSION}"
  CXX_VISIBILITY_PRESET "hidden"
  )

npcomp_python_target_compile_options(NPCOMPTorchMLIRExt)
mlir_check_all_link_libraries(NPCOMPTorchMLIRExt)
