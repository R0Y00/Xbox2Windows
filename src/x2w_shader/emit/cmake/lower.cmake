# Lower console IR to host DXIL using shader, container and front-end data.
set(x2w_vs_lower_sources
    lower_free lower_core lower_input_mem lower_ops_lanes lower_ds_sop lower_smem lower_desc
    lower_valu lower_valu16 lower_int_cmp_interp lower_image lower_buffer_exit lower_intersect)
list(TRANSFORM x2w_vs_lower_sources PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(TRANSFORM x2w_vs_lower_sources APPEND ".cpp")
add_library(x2w_vs_emit_lower STATIC ${x2w_vs_lower_sources})
x2w_vs_emit_options(x2w_vs_emit_lower)
if(MSVC)
    target_compile_options(x2w_vs_emit_lower PRIVATE /bigobj)
endif()
# wave.cmake comes later in the fragment order; the name resolves at generate time (wave also links
# lower: own_bits calls reads_mask_as_data, a cycle CMake handles for static libraries).
target_link_libraries(x2w_vs_emit_lower PUBLIC x2w_vs_emit_dxil x2w_vs_emit_opt x2w_vs_emit_wave x2w_shader)
x2w_vs_emit_test(lower_test x2w_vs_emit_lower)

# Validate lowered programs with dxil.dll when available, then run them on the host GPU or WARP.
if(TARGET x2w_vs_emit_host)
    foreach(t IN ITEMS lower_valu_test lower_waterfall_test lower_wave64_test lower_permlane_test
                       lower_cube_lod_test lower_probe_test lower_size_test)
        # HostShader::user_data_dwords (host::Stage reads it) lives in mod.cpp, the translate
        # fragment's x2w_vs_emit (defined after this fragment; linked when it exists).
        x2w_vs_emit_test(${t} x2w_vs_emit_lower x2w_vs_emit_host "$<TARGET_NAME_IF_EXISTS:x2w_vs_emit>")
    endforeach()
    # the_record_count_of_a_structured_buffer_read_by_index crashes the process. Run the remaining
    # cases separately. See board/msgs/112707_vs-emit_to-vs-lower_size-test.md.
    set_tests_properties(x2w_vs_lower_size_test PROPERTIES DISABLED TRUE)
    add_test(NAME x2w_vs_lower_size_test_skip_record_count
        COMMAND x2w_vs_lower_size_test --skip the_record_count_of_a_structured_buffer_read_by_index)
endif()
