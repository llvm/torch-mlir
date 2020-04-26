add_llvm_tool(npcomp-dummy-runner
  dummy-runner.cpp
)

target_link_libraries(npcomp-dummy-runner PRIVATE
  LLVMSupport
)
