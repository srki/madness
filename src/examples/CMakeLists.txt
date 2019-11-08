# src/examples

set(EXAMPLE_SOURCES
    h2dft hedft hello hatom_energy h2 he tdse_example heat heat2 csqrt 
    sdf_shape_tester test_gmres tdse1d vnucso nonlinschro sininteg functionio 
    dataloadbal hatom_1d binaryop dielectric hehf 3dharmonic testsolver
    testspectralprop dielectric_external_field mp2 tiny oep h2dynamic newsolver
    cc2 nemo helium_exact density_smoothing siam_example lrccs ac_corr derivatives)
 
if(LIBXC_FOUND)
  list(APPEND EXAMPLE_SOURCES hefxc)
endif()

# Create executables for example applications
foreach(example ${EXAMPLE_SOURCES})
  add_executable(${example} ${example}.cc)
  target_link_libraries(${example} MADchem  ${DISABLEPIE_LINKER_FLAG})

  add_custom_command(TARGET ${example}
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND}
      -Dlocal_dir="${PROJECT_SOURCE_DIR}"
      -Doutput_dir="${PROJECT_BINARY_DIR}/src"
      -P "${PROJECT_SOURCE_DIR}/cmake/git_version.cmake"
  )
  # for finding generated gitversion.h
  target_include_directories(${example} PRIVATE ${CMAKE_BINARY_DIR}/src)
endforeach()
