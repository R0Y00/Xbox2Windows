if(TARGET compile_dxil AND EXISTS "${INTAKE_PYTHON}"
        AND EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    set(mtbuf_window_check "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_compute_mtbuf_window_check.py")
    add_test(NAME startup_compute_mtbuf_window_controls COMMAND "${INTAKE_PYTHON}" -B
        "${mtbuf_window_check}" controls "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}")
    set_tests_properties(startup_compute_mtbuf_window_controls PROPERTIES TIMEOUT 60)
    if(TARGET startup_stream_consumer AND TARGET offscreen_check)
        foreach(adapter IN ITEMS warp hardware)
            add_test(NAME startup_compute_mtbuf_window_${adapter} COMMAND "${INTAKE_PYTHON}" -B
                "${mtbuf_window_check}" "$<TARGET_FILE:compile_dxil>" "$<TARGET_FILE:startup_stream_consumer>"
                "$<TARGET_FILE:offscreen_check>" "${DXC_BIN_DIR}" ${adapter})
            set_tests_properties(startup_compute_mtbuf_window_${adapter} PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
        endforeach()
    endif()
endif()
