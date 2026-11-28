if(TARGET compile_dxil AND EXISTS "${INTAKE_PYTHON}" AND
   EXISTS "${DXC_BIN_DIR}/dxc.exe" AND EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND
   EXISTS "${DXC_BIN_DIR}/dxil.dll")
    set(checker "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_scalar_mask64_check.py")
    add_test(NAME native_scalar_mask64_controls COMMAND "${INTAKE_PYTHON}" -B
        "${checker}" controls "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}")
    set_tests_properties(native_scalar_mask64_controls PROPERTIES TIMEOUT 120)
    if(TARGET native_store_replay)
        add_test(NAME native_scalar_mask64_hardware COMMAND "${INTAKE_PYTHON}" -B
            "${checker}" hardware "$<TARGET_FILE:native_store_replay>" "${DXC_BIN_DIR}")
        set_tests_properties(native_scalar_mask64_hardware PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    endif()
    unset(checker)
endif()
