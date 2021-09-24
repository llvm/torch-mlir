configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TEST_DEPENDS
  FileCheck count not
  npcomp-opt
  NPCOMPTorchSupportPythonModules
  TorchMLIRPythonModules
  )

add_lit_testsuite(check-npcomp-python "Running the npcomp-python regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  )
set_target_properties(check-npcomp-python PROPERTIES FOLDER "Tests")

add_lit_testsuites(NPCOMP_PYTHON ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${TEST_DEPENDS})
add_dependencies(check-npcomp-all check-npcomp-python)
