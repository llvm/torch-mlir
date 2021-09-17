add_subdirectory(Common)

# We include this unconditionally, because we use the `iree-dialects` public
# op surface area to target IREE. No "external" C++ dependency is needed
# because we copy those files into npcomp and build them into our compiler
add_subdirectory(IREE)
