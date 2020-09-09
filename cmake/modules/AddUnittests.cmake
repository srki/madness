macro(add_unittests _component _sources _libs)

  # Add targets and for world_unittests
  add_custom_target_subproject(madness ${_component}_unittests)
  add_dependencies(unittests-madness ${_component}_unittests-madness)

  # Add a test that builds the unit tests
  add_test(build_${_component}_unittests
      "${CMAKE_COMMAND}" --build ${PROJECT_BINARY_DIR} --target ${_component}_unittests-madness)
  
  foreach(_source ${_sources})
    # Get the test name (the file name of the first source)
    string(REGEX MATCH "[A-Za-z_][A-Za-z0-9_]*\\.cc" _test_source "${_source}")
    string(REGEX MATCHALL "[A-Za-z0-9_\\.\\$<:>]+" _source_list "${_source}")
    get_filename_component(_test "${_test_source}" NAME_WE)
    
    # Create test executable
    add_mad_executable(${_test} "${_source_list}" "${_libs}")

    # Add the test and set dependencies
    add_test(NAME ${_component}-${_test} COMMAND ${_test})
    add_dependencies(${_component}_unittests-madness ${_test})
    set_tests_properties(${_component}-${_test} 
        PROPERTIES DEPENDS build_${_component}_unittests)
 
  endforeach()

endmacro()