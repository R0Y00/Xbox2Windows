if(TARGET compile_dxil AND TARGET native_store_replay AND
   EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    foreach(mode controls hardware)
        add_test(NAME native_wave64_lane_pairs_${mode} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_wave64_lane_pairs_check.py" ${mode}
            "$<TARGET_FILE:compile_dxil>" "$<TARGET_FILE:native_store_replay>" "${DXC_BIN_DIR}")
    endforeach()
    set_tests_properties(native_wave64_lane_pairs_controls PROPERTIES TIMEOUT 60)
    set_tests_properties(native_wave64_lane_pairs_hardware PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
endif()
