message(STATUS "Building end-to-end testing package.")

################################################################################
# Setup python.
################################################################################

add_subdirectory(torch_mlir_e2e_test)
