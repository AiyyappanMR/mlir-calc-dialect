llvm_canonicalize_cmake_booleans(
        MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(CALC_TEST_DEPENDS
        FileCheck count not
        calc-opt
        )

add_lit_testsuite(check-calc "Running the calc regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${CALC_TEST_DEPENDS}
)

set_target_properties(check-calc PROPERTIES FOLDER "Tests")

add_lit_testsuites(CALC ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${CALC_TEST_DEPENDS})
