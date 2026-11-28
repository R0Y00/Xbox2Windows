# #1199, X2W_D3D12_VIEW_CONTENT_TOKENS (docs/startup-runtime.md#d3d12-view-content-tokens).
# startup_view_content_token: the token encoding and decoding without a device (src/startup_stubs/view_content_token.h).
add_executable(startup_view_content_token_check tests/startup_view_content_token_check.cpp)
target_include_directories(startup_view_content_token_check PRIVATE src)
target_compile_features(startup_view_content_token_check PRIVATE cxx_std_17)
add_test(NAME startup_view_content_token COMMAND startup_view_content_token_check)
# startup_view_content_<off|on>_<backend>: title-side and shadow copies of buffer SRV, CBV and texture SRV tokens keep
# naming their view while the descriptor is overwritten 5000 times; with the switch on the buffer views' records are
# parked, revived and reused, and the record count stays bounded.
add_executable(startup_view_content tests/startup_view_content.cpp)
target_compile_features(startup_view_content PRIVATE cxx_std_17)
target_link_libraries(startup_view_content PRIVATE d3d12_xs_shim d3d12 dxgi)
foreach(backend IN ITEMS warp hardware)
    foreach(mode IN ITEMS off on)
        set(content_test startup_view_content_${mode}_${backend})
        add_test(NAME ${content_test} COMMAND startup_view_content ${backend} ${mode})
        set(content_env "X2W_D3D12_BUDGET=1;X2W_D3D12_RELEASE_RETIRED=1;X2W_D3D12_PERMISSIVE=1;X2W_D3D12_RENDER=1")
        list(APPEND content_env "X2W_D3D12_DEBUG=0" "X2W_TRAP_LOG=${CMAKE_CURRENT_BINARY_DIR}/${content_test}.log")
        if(mode STREQUAL "on")
            list(APPEND content_env "X2W_D3D12_VIEW_CONTENT_TOKENS=1")
        else()
            list(APPEND content_env "X2W_D3D12_VIEW_CONTENT_TOKENS=0")
        endif()
        set_tests_properties(${content_test} PROPERTIES TIMEOUT 60 SKIP_RETURN_CODE 77 ENVIRONMENT "${content_env}")
    endforeach()
endforeach()
