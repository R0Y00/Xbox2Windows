# Bounded frame-slot pointers passed to non-recursive internal callees in tools/cpu_switch_aot.py:
# verifies differential native vs lifted (/Od and /O2) execution, C++ & Python guard mutations,
# and refusal negative controls (escaping pointers, recursion, tail jumps, aliasing, bounds, uninit).
if(EXISTS "${INTAKE_PYTHON}" AND COMMAND cpu_asm_fixture AND COMMAND cpu_translation)
    cpu_asm_fixture(cpu_switch_frame_ptr switch_frame_ptr)
    cpu_translation(cpu_switch_frame_ptr cpu_switch_aot
        out_u32_rsp,rmw_u64_rbp,all_widths_and_regs,non_anchor_rbp_arith)
    add_test(NAME cpu_switch_frame_ptr COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/cpu/switch_frame_ptr_check.py"
        "$<TARGET_FILE:cpu_register_probe>"
        "$<TARGET_FILE:cpu_switch_frame_ptr>"
        "$<TARGET_FILE:cpu_switch_frame_ptr_lifted>"
        "${CMAKE_CURRENT_BINARY_DIR}/cpu_switch_frame_ptr_lifted.cpp"
        "${CMAKE_CXX_COMPILER}")
endif()
