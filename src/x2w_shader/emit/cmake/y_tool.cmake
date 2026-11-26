# The x2w shaders command-line tool (tools/x2w). The parts that need no translator (BLAKE3, SGD5, the container scan and
# the export loop) and their tests are built always; the dump itself and the `x2w` executable only
# once the translator library (x2w_vs_emit, x_translate.cmake) exists. Named y_ so that it is
# included after x_translate.cmake.
set(x2w_tool_dir "${x2w_vs_root}/tools/x2w")
set(x2w_tool_tests "${x2w_vs_root}/tests/x2w_tool")

add_library(x2w_tool_base STATIC
    "${x2w_tool_dir}/blake3.cpp"
    "${x2w_tool_dir}/sgd5.cpp"
    "${x2w_tool_dir}/scan.cpp")
x2w_vs_emit_options(x2w_tool_base)
target_include_directories(x2w_tool_base PUBLIC "${x2w_tool_dir}")
target_link_libraries(x2w_tool_base PUBLIC x2w_shader)

foreach(t blake3_test sgd5_test scan_test)
    add_executable(x2w_tool_${t} "${x2w_tool_tests}/${t}.cpp")
    x2w_vs_emit_options(x2w_tool_${t})
    target_include_directories(x2w_tool_${t} PRIVATE "${x2w_vs_emit_tests}")
    target_link_libraries(x2w_tool_${t} PRIVATE x2w_tool_base)
    add_test(NAME x2w_tool_${t} COMMAND x2w_tool_${t})
endforeach()

if(TARGET x2w_vs_emit)
    add_library(x2w_tool_dump STATIC "${x2w_tool_dir}/shaders_dump.cpp")
    x2w_vs_emit_options(x2w_tool_dump)
    target_compile_definitions(x2w_tool_dump PRIVATE "X2W_TOOL_SOURCE_DIR=\"${x2w_vs_root}\"")
    target_link_libraries(x2w_tool_dump PUBLIC x2w_tool_base x2w_vs_emit d3d12 dxgi)

    add_executable(x2w_tool_exe "${x2w_tool_dir}/main.cpp")
    x2w_vs_emit_options(x2w_tool_exe)
    set_target_properties(x2w_tool_exe PROPERTIES OUTPUT_NAME x2w)
    target_link_libraries(x2w_tool_exe PRIVATE x2w_tool_dump)
endif()
