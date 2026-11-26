if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME mesh_bitcmp_cpu COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/mesh_bitcmp_check.py")
    foreach(adapter IN ITEMS hardware warp)
        add_test(NAME mesh_bitcmp_draw_${adapter} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_SOURCE_DIR}/tests/mesh_bitcmp_check.py" draw "${DXC_EXECUTABLE}"
            "$<TARGET_FILE:offscreen_check>" ${adapter})
        set_tests_properties(mesh_bitcmp_draw_${adapter} PROPERTIES TIMEOUT 60 SKIP_RETURN_CODE 77)
    endforeach()
endif()
