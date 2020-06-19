configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(NPCOMP_TEST_DEPENDS
        FileCheck count not
        npcomp-opt
        NPCOMPNativePyExt
        # TODO: Fix this so it has an IREE prefix
        bindings_python_pyiree_rt_rt
        # TODO: Why is this separate?
        bindings_python_pyiree_rt_system_api
)

add_lit_testsuite(check-npcomp-backend-iree-lit "Running npcomp IREE tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${NPCOMP_TEST_DEPENDS}
        )
set_target_properties(check-npcomp-backend-iree-lit PROPERTIES FOLDER "Tests")

add_lit_testsuites(NPCOMP_PYTEST ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${NPCOMP_TEST_DEPENDS})

add_dependencies(check-npcomp check-npcomp-backend-iree-lit)
