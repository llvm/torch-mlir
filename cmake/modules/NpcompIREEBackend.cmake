  function(npcomp_iree_target_compile_options target)
  target_compile_options(${target} PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wno-sign-compare
  >
  $<$<CXX_COMPILER_ID:MSVC>:>
  )
endfunction()
