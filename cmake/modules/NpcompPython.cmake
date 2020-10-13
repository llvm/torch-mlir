function(npcomp_python_target_compile_options target)
  target_compile_options(${target} PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
      # Enable RTTI and exceptions.
      -frtti -fexceptions
      # Noisy pybind warnings
      -Wno-unused-value
      -Wno-covered-switch-default
    >
    $<$<CXX_COMPILER_ID:MSVC>:
    # Enable RTTI and exceptions.
    /EHsc /GR>
  )
endfunction()

function(npcomp_python_create_symlinks binary_dir source_dir)
    # Do nothing if building in-source
    if (${binary_dir} STREQUAL ${source_dir})
        return()
    endif()

    file(GLOB_RECURSE python_files RELATIVE ${source_dir} *.py)
    foreach (path_file ${python_files})
        get_filename_component(folder ${path_file} PATH)

        # Create REAL folder
        file(MAKE_DIRECTORY "${binary_dir}/${folder}")

        # Get OS dependent path to use in `execute_process`
        file(TO_NATIVE_PATH "${binary_dir}/${path_file}" link)
        file(TO_NATIVE_PATH "${source_dir}/${path_file}" target)

        # TODO: Switch to copy on windows if symlink still not supported by
        # then.
        set(cmake_verb create_symlink)
        execute_process(COMMAND ${CMAKE_COMMAND} -E ${cmake_verb} ${target} ${link}
          RESULT_VARIABLE result
          ERROR_VARIABLE output)

        if (NOT ${result} EQUAL 0)
            message(FATAL_ERROR "Could not create symbolic link for: ${target} --> ${output}")
        endif()
    endforeach(path_file)
endfunction(npcomp_python_create_symlinks)
