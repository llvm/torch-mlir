configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(NPCOMP_TEST_DEPENDS
        FileCheck count not
        npcomp-opt
        refback-run
        NPCOMPPythonModules
)

add_lit_testsuite(check-npcomp-lit "Running the npcomp regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${NPCOMP_TEST_DEPENDS}
        )
set_target_properties(check-npcomp-lit PROPERTIES FOLDER "Tests")

add_lit_testsuites(NPCOMP_OPT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${NPCOMP_TEST_DEPENDS})

add_dependencies(check-npcomp check-npcomp-lit)
