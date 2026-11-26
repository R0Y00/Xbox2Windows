# DXIL validator (dxil.dll) and D3D12 test host.
# host.cpp reads signature parts and builds the stub shaders (x2w_vs_emit_dxil: shader.cpp,
# mod_signature.cpp; x2w_shader: dxbc.cpp); its HostShader stub_vs/stub_ps wrappers compile once
# x2w_shader/native.h exists.
# Build alone with -DX2W_VS_EMIT_ONLY="opt;dxil;host".
add_library(x2w_vs_emit_host STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/validator.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/host.cpp")
x2w_vs_emit_options(x2w_vs_emit_host)
target_link_libraries(x2w_vs_emit_host PUBLIC x2w_vs_emit_dxil x2w_vs_emit_base x2w_shader d3d12 dxgi)

x2w_vs_emit_test(validator_test x2w_vs_emit_host)
x2w_vs_emit_test(host_test x2w_vs_emit_host)

# Validate the same reference containers used by ref_test.
if(DEFINED x2w_vs_ref)
    x2w_vs_emit_test(ref_validate_test x2w_vs_emit_host)
    set_tests_properties(x2w_vs_ref_validate_test PROPERTIES ENVIRONMENT "X2W_REF_DIR=${x2w_vs_ref}/out")
    add_test(NAME x2w_vs_ref_validate_test_opt0 COMMAND x2w_vs_ref_validate_test)
    set_tests_properties(x2w_vs_ref_validate_test_opt0 PROPERTIES ENVIRONMENT "X2W_REF_DIR=${x2w_vs_ref}/out_opt0")
endif()
