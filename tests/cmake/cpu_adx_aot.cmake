# Dedicated cpu_adx_aot lifter (32/64-bit ADCX and ADOX, reg + flat memory, dual-carry flags)
if(COMMAND cpu_asm_fixture AND COMMAND cpu_translation AND EXISTS "${INTAKE_PYTHON}")
    set(cpu_adx_aot_exports
        "op_adcx32_c0,op_adcx32_c1,op_adcx64_c0,op_adcx64_c1,"
        "op_adox32_o0,op_adox32_o1,op_adox64_o0,op_adox64_o1,"
        "op_adcx32_mem,op_adcx64_mem,op_adox32_mem,op_adox64_mem,"
        "op_adcx32_flags,op_adcx64_flags,op_adox32_flags,op_adox64_flags,"
        "op_adcx64_mem_flags,op_adox64_mem_flags,"
        "op_adx_dual64_adcx,op_adx_dual64_adox,op_adx_dst_alias,op_adx_discarded_read")
    string(CONCAT cpu_adx_aot_exports ${cpu_adx_aot_exports})
    foreach(variant IN ITEMS original altered)
        cpu_asm_fixture(cpu_adx_aot_${variant} adx_aot)
        cpu_translation(cpu_adx_aot_${variant} cpu_adx_aot "${cpu_adx_aot_exports}")
    endforeach()
    target_compile_definitions(cpu_adx_aot_altered PRIVATE ALTER_ADX)
    add_test(NAME cpu_aot_adx_dedicated COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/cpu/adx_aot_check.py"
        "$<TARGET_FILE:cpu_adx_aot_original>"
        "$<TARGET_FILE:cpu_adx_aot_original_lifted>"
        "$<TARGET_FILE:cpu_adx_aot_altered>"
        "$<TARGET_FILE:cpu_adx_aot_altered_lifted>"
        "${CMAKE_CXX_COMPILER}")
    set_tests_properties(cpu_aot_adx_dedicated PROPERTIES
        TIMEOUT 60
        SKIP_RETURN_CODE 77
        ENVIRONMENT "FORM_COVERAGE_RECORD=${CMAKE_BINARY_DIR}/form_coverage/cpu_aot_adx_dedicated.txt")
endif()
