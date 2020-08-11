include_directories(
  ${TORCH_INCLUDE_DIRS}
  )
add_library(aten_ops SHARED
  aten_ops.cpp
  )

target_link_libraries(aten_ops
  ${TORCH_LIBRARIES}
 )
