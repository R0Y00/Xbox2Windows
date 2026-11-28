# Reuse the existing source-built mesh executable and its generated DXIL.
if(EXISTS "${INTAKE_PYTHON}")
    foreach(adapter IN ITEMS warp hardware)
        add_test(NAME startup_mesh_argument_provenance_${adapter}
            COMMAND "${INTAKE_PYTHON}" -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_mesh_argument_provenance_check.py"
                "$<TARGET_FILE:startup_indirect_draw_indexed>" ${adapter})
        set_tests_properties(startup_mesh_argument_provenance_${adapter} PROPERTIES
            TIMEOUT 120 SKIP_RETURN_CODE 77)
    endforeach()
endif()
