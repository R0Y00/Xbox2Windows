# Direct C++ compute: v_cvt_f64_i32 and v_cvt_f64_u32 (#3363, Refs #2935).
if(TARGET native_compute_ir_dump AND EXISTS "${INTAKE_PYTHON}")
    add_test(NAME native_cvt_f64_int_ir COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/native_cvt_f64_int_check.py" ir
        "$<TARGET_FILE:native_compute_ir_dump>")
    set_tests_properties(native_cvt_f64_int_ir PROPERTIES TIMEOUT 60)
endif()
if(TARGET compile_dxil AND TARGET native_store_replay AND EXISTS "${INTAKE_PYTHON}" AND
   EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    add_test(NAME native_cvt_f64_int_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/native_cvt_f64_int_check.py" controls
        "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}")
    set_tests_properties(native_cvt_f64_int_controls PROPERTIES TIMEOUT 60)
    add_test(NAME native_cvt_f64_int_hardware COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/native_cvt_f64_int_check.py" hardware
        "$<TARGET_FILE:native_store_replay>" "${DXC_BIN_DIR}")
    set_tests_properties(native_cvt_f64_int_hardware PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
endif()
