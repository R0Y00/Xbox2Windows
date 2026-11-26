# Build the shader container, rectangle stage, settings, and signature readers. Requires opt.
add_library(x2w_vs_emit_dxil STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/container.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shader.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/settings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/mod_signature.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rect.cpp")
x2w_vs_emit_options(x2w_vs_emit_dxil)
target_link_libraries(x2w_vs_emit_dxil PUBLIC x2w_vs_emit_opt x2w_vs_emit_base x2w_shader)

# Compare emitted containers against reference outputs in tests/x2w_shader/emit/ref.
set(x2w_vs_ref "${x2w_vs_emit_tests}/ref")
x2w_vs_emit_test(ref_test x2w_vs_emit_dxil)
set_tests_properties(x2w_vs_ref_test PROPERTIES ENVIRONMENT "X2W_REF_DIR=${x2w_vs_ref}/out")
add_test(NAME x2w_vs_ref_test_opt0 COMMAND x2w_vs_ref_test)
set_tests_properties(x2w_vs_ref_test_opt0 PROPERTIES ENVIRONMENT "X2W_REF_DIR=${x2w_vs_ref}/out_opt0;X2W_SHADERCFG_SHADER_OPT=0")
x2w_vs_emit_test(container_test x2w_vs_emit_dxil)
