# Run translated and lowered shaders on the host GPU. The z_ prefix places these tests after
# the fragments defining their targets.
# rect_test needs only the DXIL writer and the host.
if(TARGET x2w_vs_emit_host)
    x2w_vs_emit_test(rect_test x2w_vs_emit_host)
endif()
# diff_test (the console's dxc against the PC's: X2W_SHADERCFG_CONSOLE_DXC, X2W_SHADERCFG_PC_DXC, X2W_SHADERCFG_WARP)
# and intersect_test (the BVH node test against bvh.cpp's reference) need the translator.
if(TARGET x2w_vs_emit AND TARGET x2w_vs_emit_host)
    x2w_vs_emit_test(diff_test x2w_vs_emit x2w_vs_emit_host)
    x2w_vs_emit_test(intersect_test x2w_vs_emit x2w_vs_emit_lower x2w_vs_emit_host)
endif()
