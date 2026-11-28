# Typed buffer inputs (buffer_load_format_*) on the published compute route (#3224).
if(TARGET compile_dxil AND EXISTS "${INTAKE_PYTHON}"
        AND EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    add_test(NAME startup_compute_typed_buffer_srv_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_compute_typed_buffer_srv_check.py" compile
        "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}")
    set_tests_properties(startup_compute_typed_buffer_srv_controls PROPERTIES TIMEOUT 60)
endif()
if(TARGET compile_dxil AND TARGET startup_stream_consumer AND TARGET offscreen_check AND EXISTS "${INTAKE_PYTHON}"
        AND EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    foreach(adapter IN ITEMS warp hardware)
        add_test(NAME startup_compute_typed_buffer_srv_${adapter} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_compute_typed_buffer_srv_check.py"
            "$<TARGET_FILE:compile_dxil>" "$<TARGET_FILE:startup_stream_consumer>" "$<TARGET_FILE:offscreen_check>"
            "${DXC_BIN_DIR}" ${adapter})
        set_tests_properties(startup_compute_typed_buffer_srv_${adapter} PROPERTIES TIMEOUT 180 SKIP_RETURN_CODE 77)
    endforeach()
endif()
