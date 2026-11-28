add_executable(startup_fence_values tests/startup_fence_values.cpp)
target_compile_features(startup_fence_values PRIVATE cxx_std_17)
target_link_libraries(startup_fence_values PRIVATE d3d12_xs_shim d3d12 dxgi)
foreach(backend IN ITEMS warp hardware)
    add_test(NAME startup_fence_values_${backend} COMMAND startup_fence_values ${backend})
    set_tests_properties(startup_fence_values_${backend} PROPERTIES TIMEOUT 60 SKIP_RETURN_CODE 77
        ENVIRONMENT "X2W_D3D12_PERMISSIVE=1;X2W_D3D12_LIST_NOOP=1;X2W_D3D12_RENDER=1;X2W_D3D12_DEBUG=1")
endforeach()
