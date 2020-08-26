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
endif()

add_library(_torch_mlir SHARED
  init_python_bindings.cpp
  )
set_target_properties(_torch_mlir PROPERTIES PREFIX "")
get_property(mlir_libs GLOBAL PROPERTY MLIR_ALL_LIBS)
target_link_libraries(_torch_mlir
  NPCOMPATenDialect
  ${TORCH_LIBRARIES}
  ${mlir_libs}
  ${PYTHON_LIBRARIES}
  ${torch_mlir_optional_libraries}
  torch_python
  )

if(NPCOMP_ENABLE_TORCH_TYPE_DISPATCH)
  add_subdirectory(type_dispatch)
endif()
