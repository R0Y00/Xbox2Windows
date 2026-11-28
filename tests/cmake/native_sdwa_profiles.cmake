# Remaining lowered VOP1/VOP2 SDWA operand profiles in direct compute.
if(TARGET native_compute_ir_dump AND TARGET native_pixel_ir_dump)
    add_test(NAME native_sdwa_profiles_ir COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/native_sdwa_profiles_check.py" ir
        "$<TARGET_FILE:native_compute_ir_dump>" "$<TARGET_FILE:native_pixel_ir_dump>")
    set_tests_properties(native_sdwa_profiles_ir PROPERTIES TIMEOUT 60)
endif()

if(TARGET compile_dxil AND EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    add_test(NAME native_sdwa_profiles_dxil COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/native_sdwa_profiles_check.py" dxil
        "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}")
    set_tests_properties(native_sdwa_profiles_dxil PROPERTIES TIMEOUT 60)
endif()

if(TARGET native_store_replay AND EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    add_test(NAME native_sdwa_profiles_hardware COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/native_sdwa_profiles_check.py" hardware
        "$<TARGET_FILE:native_store_replay>" "${DXC_BIN_DIR}")
    set_tests_properties(native_sdwa_profiles_hardware PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
endif()
