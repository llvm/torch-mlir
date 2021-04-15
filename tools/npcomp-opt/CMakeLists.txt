# npcomp-opt is always linked dynamically as we want to distribute the
# binaries with the python packages for hacking/debugging.
add_npcomp_executable(npcomp-opt npcomp-opt.cpp)

target_link_libraries(npcomp-opt PRIVATE
  # Shared library deps first ensure we get most of what we need from libraries.
  NPCOMP
  MLIROptLib
)

mlir_check_all_link_libraries(npcomp-opt)
