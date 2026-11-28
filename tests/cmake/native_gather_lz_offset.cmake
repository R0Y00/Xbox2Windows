if(TARGET compile_dxil AND EXISTS "${INTAKE_PYTHON}" AND
   EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    set(checker "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_gather_lz_offset_check.py")
    add_test(NAME native_gather_lz_offset_controls COMMAND "${INTAKE_PYTHON}" -B
        "${checker}" controls "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}")
    set_tests_properties(native_gather_lz_offset_controls PROPERTIES TIMEOUT 120)
    if(TARGET offscreen_check)
        foreach(adapter IN ITEMS warp hardware)
            add_test(NAME native_gather_lz_offset_${adapter} COMMAND "${INTAKE_PYTHON}" -B
                "${checker}" hardware "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}"
                "$<TARGET_FILE:offscreen_check>" ${adapter})
            set_tests_properties(native_gather_lz_offset_${adapter} PROPERTIES TIMEOUT 180 SKIP_RETURN_CODE 77)
        endforeach()
    endif()
    unset(checker)
endif()
