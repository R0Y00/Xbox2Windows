if(TARGET compile_dxil AND TARGET native_store_replay AND
   EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    foreach(mode controls hardware warp)
        add_test(NAME native_wave32_lds_order_${mode} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_wave32_lds_order_check.py" ${mode}
            "$<TARGET_FILE:compile_dxil>" "$<TARGET_FILE:native_store_replay>" "${DXC_BIN_DIR}")
        set_tests_properties(native_wave32_lds_order_${mode} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
    endforeach()
endif()
