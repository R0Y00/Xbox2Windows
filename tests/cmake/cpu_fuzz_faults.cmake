# Bad-pointer fault-equivalence fuzz area (fault_reads): dead/overwritten 8/16/32/64-bit guest
# memory reads ([rcx], [rdx], [rcx+disp]) plus valid-pointer controls, with a dead-read-dropping
# mutant compiled with /O2.
if(EXISTS "${INTAKE_PYTHON}")
    set(CPU_FUZZ_FAULT_READS_SEED "0xFA17" CACHE STRING "Seed for the CPU differential fuzz fault-reads test")
    set(CPU_FUZZ_FAULT_READS_FUNCTIONS "24" CACHE STRING "Number of generated functions in the CPU differential fuzz fault-reads test")
    set(cpu_fuzz_fault_reads_asm "${CMAKE_CURRENT_BINARY_DIR}/cpu_fuzz_fault_reads.asm")
    set(cpu_fuzz_fault_reads_def "${CMAKE_CURRENT_BINARY_DIR}/cpu_fuzz_fault_reads.def")
    add_custom_command(OUTPUT "${cpu_fuzz_fault_reads_asm}" "${cpu_fuzz_fault_reads_def}"
        COMMAND "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tools/fuzz_cpu.py" emit
            "${CPU_FUZZ_FAULT_READS_SEED}" "${CPU_FUZZ_FAULT_READS_FUNCTIONS}"
            "${cpu_fuzz_fault_reads_asm}" "${cpu_fuzz_fault_reads_def}" --area fault_reads
        DEPENDS "${CMAKE_SOURCE_DIR}/tools/fuzz_cpu.py" VERBATIM)
    add_library(cpu_fuzz_fault_reads SHARED "${cpu_fuzz_fault_reads_asm}" "${cpu_fuzz_fault_reads_def}")
    set_target_properties(cpu_fuzz_fault_reads PROPERTIES LINKER_LANGUAGE CXX)
    target_link_options(cpu_fuzz_fault_reads PRIVATE /NOENTRY /NODEFAULTLIB /INCREMENTAL:NO /OPT:NOICF)

    set(cpu_fuzz_fault_reads_lifted "${CMAKE_CURRENT_BINARY_DIR}/cpu_fuzz_fault_reads_lifted.cpp")
    add_custom_command(OUTPUT "${cpu_fuzz_fault_reads_lifted}"
        COMMAND "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tools/fuzz_cpu.py" translate
            "${CPU_FUZZ_FAULT_READS_SEED}" "${CPU_FUZZ_FAULT_READS_FUNCTIONS}"
            "$<TARGET_FILE:cpu_fuzz_fault_reads>" "${cpu_fuzz_fault_reads_lifted}" --area fault_reads
        DEPENDS cpu_fuzz_fault_reads "${CMAKE_SOURCE_DIR}/tools/fuzz_cpu.py"
            "${CMAKE_SOURCE_DIR}/tools/cpu_switch_aot.py" "${CMAKE_SOURCE_DIR}/tools/cpu_leaf_aot.py"
            "${CMAKE_SOURCE_DIR}/tools/cpu_call_aot.py" "${CMAKE_SOURCE_DIR}/tools/cpu_ranges.py" VERBATIM)
    add_library(cpu_fuzz_fault_reads_lifted SHARED "${cpu_fuzz_fault_reads_lifted}")
    target_compile_features(cpu_fuzz_fault_reads_lifted PRIVATE cxx_std_20)

    set(cpu_fuzz_fault_reads_mutant "${CMAKE_CURRENT_BINARY_DIR}/cpu_fuzz_fault_reads_mutant.cpp")
    add_custom_command(OUTPUT "${cpu_fuzz_fault_reads_mutant}"
        COMMAND "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tools/fuzz_cpu.py" translate
            "${CPU_FUZZ_FAULT_READS_SEED}" "${CPU_FUZZ_FAULT_READS_FUNCTIONS}"
            "$<TARGET_FILE:cpu_fuzz_fault_reads>" "${cpu_fuzz_fault_reads_mutant}" --area fault_reads --mutant
        DEPENDS cpu_fuzz_fault_reads "${CMAKE_SOURCE_DIR}/tools/fuzz_cpu.py"
            "${CMAKE_SOURCE_DIR}/tools/cpu_switch_aot.py" "${CMAKE_SOURCE_DIR}/tools/cpu_leaf_aot.py"
            "${CMAKE_SOURCE_DIR}/tools/cpu_call_aot.py" "${CMAKE_SOURCE_DIR}/tools/cpu_ranges.py" VERBATIM)

    set(cpu_fuzz_fault_reads_o2_dll "${CMAKE_CURRENT_BINARY_DIR}/cpu_fuzz_fault_reads_lifted_o2.dll")
    add_custom_command(OUTPUT "${cpu_fuzz_fault_reads_o2_dll}"
        COMMAND "${CMAKE_CXX_COMPILER}" /nologo /O2 /W4 /LD "${cpu_fuzz_fault_reads_lifted}"
            "/Fo${CMAKE_CURRENT_BINARY_DIR}/cpu_fuzz_fault_reads_lifted_o2.obj" "/Fe${cpu_fuzz_fault_reads_o2_dll}"
        DEPENDS "${cpu_fuzz_fault_reads_lifted}" VERBATIM)

    set(cpu_fuzz_fault_reads_mutant_o2_dll "${CMAKE_CURRENT_BINARY_DIR}/cpu_fuzz_fault_reads_mutant_o2.dll")
    add_custom_command(OUTPUT "${cpu_fuzz_fault_reads_mutant_o2_dll}"
        COMMAND "${CMAKE_CXX_COMPILER}" /nologo /O2 /W4 /LD "${cpu_fuzz_fault_reads_mutant}"
            "/Fo${CMAKE_CURRENT_BINARY_DIR}/cpu_fuzz_fault_reads_mutant_o2.obj" "/Fe${cpu_fuzz_fault_reads_mutant_o2_dll}"
        DEPENDS "${cpu_fuzz_fault_reads_mutant}" VERBATIM)

    add_custom_target(cpu_fuzz_fault_reads_o2 ALL DEPENDS
        "${cpu_fuzz_fault_reads_o2_dll}" "${cpu_fuzz_fault_reads_mutant_o2_dll}")

    add_test(NAME cpu_aot_fuzz_faults COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tools/fuzz_cpu.py" check "${CPU_FUZZ_FAULT_READS_SEED}" "${CPU_FUZZ_FAULT_READS_FUNCTIONS}"
        "$<TARGET_FILE:cpu_fuzz_fault_reads>" "$<TARGET_FILE:cpu_fuzz_fault_reads_lifted>" "${cpu_fuzz_fault_reads_o2_dll}"
        --area fault_reads --mutant "${cpu_fuzz_fault_reads_mutant_o2_dll}")
    set_tests_properties(cpu_aot_fuzz_faults PROPERTIES TIMEOUT 60)

    add_test(NAME tooling_fuzz_cpu_fault_check COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/fuzz_cpu_fault_check.py"
        --native "$<TARGET_FILE:cpu_fuzz_fault_reads>"
        --lifted "$<TARGET_FILE:cpu_fuzz_fault_reads_lifted>"
        --o2 "${cpu_fuzz_fault_reads_o2_dll}"
        --mutant "${cpu_fuzz_fault_reads_mutant_o2_dll}"
        --seed "${CPU_FUZZ_FAULT_READS_SEED}"
        --count "${CPU_FUZZ_FAULT_READS_FUNCTIONS}")
    set_tests_properties(tooling_fuzz_cpu_fault_check PROPERTIES TIMEOUT 60)
endif()
