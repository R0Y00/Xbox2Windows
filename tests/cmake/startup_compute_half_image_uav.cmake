if(TARGET compile_dxil AND EXISTS "${INTAKE_PYTHON}"
        AND EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    set(half_image_check "${CMAKE_SOURCE_DIR}/tests/startup_compute_half_image_uav_check.py")
    add_test(NAME startup_compute_half_image_uav_controls COMMAND "${INTAKE_PYTHON}" -B
        "${half_image_check}" compile "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}")
    set_tests_properties(startup_compute_half_image_uav_controls PROPERTIES TIMEOUT 60)
    if(TARGET startup_stream_consumer AND TARGET offscreen_check)
        foreach(adapter IN ITEMS warp hardware)
            add_test(NAME startup_compute_half_image_uav_${adapter} COMMAND "${INTAKE_PYTHON}" -B
                "${half_image_check}" "$<TARGET_FILE:compile_dxil>" "$<TARGET_FILE:startup_stream_consumer>"
                "$<TARGET_FILE:offscreen_check>" "${DXC_BIN_DIR}" ${adapter})
            set_tests_properties(startup_compute_half_image_uav_${adapter} PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
        endforeach()
    endif()
endif()
