if(TARGET compile_dxil AND TARGET startup_stream_consumer AND TARGET offscreen_check AND EXISTS "${INTAKE_PYTHON}"
        AND EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    foreach(adapter IN ITEMS warp hardware)
        add_test(NAME startup_compute_mtbuf_window_${adapter} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_compute_mtbuf_window_check.py"
            "$<TARGET_FILE:compile_dxil>" "$<TARGET_FILE:startup_stream_consumer>" "$<TARGET_FILE:offscreen_check>"
            "${DXC_BIN_DIR}" ${adapter})
        set_tests_properties(startup_compute_mtbuf_window_${adapter} PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    endforeach()
endif()
