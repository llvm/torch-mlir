llvm_canonicalize_cmake_booleans(
  NPCOMP_ENABLE_TORCH_TYPE_DISPATCH
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TEST_DEPENDS
  FileCheck count not
  _torch_mlir
  )


if(NPCOMP_ENABLE_TORCH_TYPE_DISPATCH)
  list(APPEND TEST_DEPENDS
    aten_ops
  )
endif()

add_lit_testsuite(check-frontends-pytorch "Running the frontends-pytorch regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  )
set_target_properties(check-frontends-pytorch PROPERTIES FOLDER "Tests")

add_lit_testsuites(TORCH_MLIR ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${TEST_DEPENDS})
add_dependencies(check-all check-frontends-pytorch)
