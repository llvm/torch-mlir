# Sharp edge: Torch extensions need to use the same pybind11 that torch
# was compiled with, or else there will be issues in cross module exception
# handling (which will abort instead of raise). We circumvent the possibility
# by forcing the torch directories first.
include_directories(BEFORE
  ${TORCH_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${Python3_INCLUDE_DIRS}
  )
link_directories("${TORCH_INSTALL_PREFIX}/lib")

add_library(TorchMLIRJITIRImporter MODULE
  class_annotator.cpp
  class_annotator_pybind.cpp
  get_registered_ops.cpp
  function_importer.cpp
  module_builder.cpp
  node_importer.cpp
  ivalue_importer.cpp
  init_python_bindings.cpp
  torch_to_mlir_utils.cpp
  )


function(find_static_library LIB_NAME OUT)
  if (WIN32 OR MSVC)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib")
  elseif (UNIX)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
  endif ()
  find_library(
          FOUND_${LIB_NAME}_STATIC
          ${LIB_NAME}
          PATHS "${TORCH_INSTALL_PREFIX}/lib"
  )
  if (FOUND_${LIB_NAME}_STATIC)
    get_filename_component(ABS_FILE ${FOUND_${LIB_NAME}_STATIC} ABSOLUTE)
  else ()
    message(SEND_ERROR "Unable to find library ${LIB_NAME}")
  endif ()
  set(${OUT} ${ABS_FILE} PARENT_SCOPE)
endfunction()

find_static_library(torch TORCH)
find_static_library(torch_cpu TORCH_CPU)
find_static_library(c10 C10)
find_static_library(cpuinfo CPUINFO)
find_static_library(clog CLOG)
find_static_library(pthreadpool PTHREADPOOL)
find_static_library(kineto KINETO)

target_link_libraries(TorchMLIRJITIRImporter
  TorchMLIRAggregateCAPI
  ${TORCH} ${TORCH_CPU} ${C10} ${CPUINFO} ${CLOG} ${PTHREADPOOL} ${KINETO}
)

# On static Python builds, there may not be Python libraries to link against
# (they will late bind at runtime from the executable). We have to condition
# this because in that case it is set to NOTFOUND and CMake will consider
# this an error.
if(Python3_LIBRARIES)
  target_link_libraries(TorchMLIRJITIRImporter
    ${Python3_LIBRARIES}
  )
endif()

message(STATUS "TORCH_CXXFLAGS=${TORCH_CXXFLAGS}")
set_target_properties(TorchMLIRJITIRImporter PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${TORCH_MLIR_PYTHON_PACKAGES_DIR}/torch_mlir/torch_mlir/_mlir_libs"
  OUTPUT_NAME _jit_ir_importer
  PREFIX "${PYTHON_MODULE_PREFIX}"
  SUFFIX "${PYTHON_MODULE_EXTENSION}"
  CXX_VISIBILITY_PRESET "hidden"
  COMPILE_FLAGS "${TORCH_CXXFLAGS}"
  )
mlir_python_setup_extension_rpath(TorchMLIRJITIRImporter)

torch_mlir_python_target_compile_options(TorchMLIRJITIRImporter)
mlir_check_all_link_libraries(TorchMLIRJITIRImporter)
