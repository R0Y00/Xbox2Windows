# Slot 133 existing-heap cap (<= 256 MiB), deferred commit_lazy_block, and incremental
# reserve_upload_ring growth (Refs #1199, Fixes #2765).
add_executable(startup_slot133_heap_cap_check tests/startup_slot133_heap_cap_check.cpp)
target_include_directories(startup_slot133_heap_cap_check PRIVATE src)
target_compile_features(startup_slot133_heap_cap_check PRIVATE cxx_std_17)
target_link_libraries(startup_slot133_heap_cap_check PRIVATE d3d12 dxgi)
add_dependencies(startup_slot133_heap_cap_check xmem_shim d3d12_xs_shim)

foreach(backend IN ITEMS warp hardware)
    foreach(debug IN ITEMS 0 1)
        set(case startup_slot133_heap_cap_${backend}_debug${debug})
        add_test(NAME ${case} COMMAND startup_slot133_heap_cap_check ${backend})
        set_tests_properties(${case} PROPERTIES TIMEOUT 60 SKIP_RETURN_CODE 77
            ENVIRONMENT "X2W_D3D12_DEBUG=${debug}")
    endforeach()
endforeach()

add_test(NAME startup_slot133_heap_cap_unbounded_warp COMMAND startup_slot133_heap_cap_check warp unbounded)
set_tests_properties(startup_slot133_heap_cap_unbounded_warp PROPERTIES TIMEOUT 60 SKIP_RETURN_CODE 77
    ENVIRONMENT "X2W_D3D12_DEBUG=0")
