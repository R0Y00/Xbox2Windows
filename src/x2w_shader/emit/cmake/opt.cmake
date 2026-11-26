# Clean-up passes over a finished module.
add_library(x2w_vs_emit_opt STATIC "${CMAKE_CURRENT_SOURCE_DIR}/opt.cpp")
x2w_vs_emit_options(x2w_vs_emit_opt)
target_link_libraries(x2w_vs_emit_opt PUBLIC x2w_vs_emit_base)

x2w_vs_emit_test(opt_test x2w_vs_emit_opt x2w_vs_emit_base)
