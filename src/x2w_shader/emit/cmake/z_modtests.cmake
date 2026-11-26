# Translation, atomic and barycentric tests run
# over the whole translator (x2w_vs_emit, cmake/x_translate.cmake) and the D3D12 test host and
# validator (x2w_vs_emit_host, cmake/host.cmake). The GPU and dxc/dxil.dll parts skip, and say so,
# where there is no device or Windows SDK.
if(TARGET x2w_vs_emit AND TARGET x2w_vs_emit_host)
    x2w_vs_emit_test(mod_test x2w_vs_emit x2w_vs_emit_host)
    # scalar_buffer_loads_read_the_same_dwords_one_call_or_four: "run the test once plain and once
    # with X2W_SHADERCFG_SHADER_OPT=127" (the wide loads, bitcode::opt::WIDE).
    add_test(NAME x2w_vs_mod_test_opt127 COMMAND x2w_vs_mod_test)
    set_tests_properties(x2w_vs_mod_test_opt127 PROPERTIES ENVIRONMENT "X2W_SHADERCFG_SHADER_OPT=127")
    x2w_vs_emit_test(atomic_test x2w_vs_emit x2w_vs_emit_host)
    x2w_vs_emit_test(barycentric_test x2w_vs_emit x2w_vs_emit_host)
else()
    message(STATUS "z_modtests: no x2w_vs_emit or x2w_vs_emit_host yet: mod_test, atomic_test, barycentric_test left out")
endif()
