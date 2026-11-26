# WP5 (rawtwins) tests of the descriptor mirror. Included from CMakeLists.txt by WP0's hook line;
# this file is WP5's own.
# The header-level layout and view rules of src/startup_stubs/gpu_heap.h. The mutated runs are
# the negative controls.
add_executable(startup_gpu_heap_layout_check tests/startup_gpu_heap_layout_check.cpp)
target_include_directories(startup_gpu_heap_layout_check PRIVATE src)
target_compile_features(startup_gpu_heap_layout_check PRIVATE cxx_std_17)
add_test(NAME startup_gpu_heap_layout COMMAND startup_gpu_heap_layout_check)
foreach(mutation probe most)
  add_test(NAME startup_gpu_heap_layout_mutate_${mutation}
           COMMAND startup_gpu_heap_layout_check --mutate ${mutation})
  set_tests_properties(startup_gpu_heap_layout_mutate_${mutation} PROPERTIES WILL_FAIL TRUE)
endforeach()

# The mirror heap on a device (src/startup_stubs/gpu_heap.inl) with fixture title memory and our own SM 6.6
# shader: twins, views, and records copied as bytes mirrored at submit. WARP: the layout at half scale (WARP takes
# no 2^22-view heap); hardware: the real one, skipped (77) on a host that takes no 2^22 views.
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/startup_gpu_heap_cs.h"
    COMMAND "${DXC_EXECUTABLE}" -T cs_6_6 -E PickMain -Vn startup_gpu_heap_cs -Fh "${CMAKE_CURRENT_BINARY_DIR}/startup_gpu_heap_cs.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gpu_heap.hlsl"
    DEPENDS tests/startup_gpu_heap.hlsl VERBATIM)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/startup_gpu_memory_cs.h"
    COMMAND "${DXC_EXECUTABLE}" -T cs_6_6 -E MemoryMain -Vn startup_gpu_memory_cs -Fh "${CMAKE_CURRENT_BINARY_DIR}/startup_gpu_memory_cs.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gpu_memory.hlsl"
    DEPENDS tests/startup_gpu_memory.hlsl VERBATIM)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/startup_gpu_memory_store_cs.h"
    COMMAND "${DXC_EXECUTABLE}" -T cs_6_6 -E StoreMain -Vn startup_gpu_memory_store_cs -Fh "${CMAKE_CURRENT_BINARY_DIR}/startup_gpu_memory_store_cs.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gpu_memory_store.hlsl"
    DEPENDS tests/startup_gpu_memory_store.hlsl VERBATIM)
add_executable(startup_gpu_heap tests/startup_gpu_heap.cpp "${CMAKE_CURRENT_BINARY_DIR}/startup_gpu_heap_cs.h" "${CMAKE_CURRENT_BINARY_DIR}/startup_gpu_memory_cs.h" "${CMAKE_CURRENT_BINARY_DIR}/startup_gpu_memory_store_cs.h")
target_include_directories(startup_gpu_heap PRIVATE src "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_features(startup_gpu_heap PRIVATE cxx_std_17)
target_link_libraries(startup_gpu_heap PRIVATE d3d12 dxgi d3dcompiler)
foreach(backend IN ITEMS warp hardware)
  foreach(mode IN ITEMS twins views typed_null submit memory memory_empty memory_store packed_light material_record pixel_subresource_capture compute_texture_capture compute_volume_capture compute_sparse_volume_capture compute_priority_depth_capture ambient resident_mips small_texture_heap list_states)
    add_test(NAME startup_gpu_${mode}_${backend} COMMAND startup_gpu_heap ${backend} ${mode})
    set_tests_properties(startup_gpu_${mode}_${backend} PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
  endforeach()
endforeach()

# The mirror heap through the shim (gpu_views.inl): a shader-visible title heap gets its mirror, a record the
# title copies as bytes is found at submit. The negative control verifies that a non-shader-visible heap has no mirror.
add_executable(startup_gpu_heap_shim tests/startup_gpu_heap_shim.cpp)
add_executable(startup_gpu_depth_twins tests/startup_gpu_depth_twins.cpp)
target_include_directories(startup_gpu_depth_twins PRIVATE src)
target_compile_features(startup_gpu_depth_twins PRIVATE cxx_std_17)
target_link_libraries(startup_gpu_depth_twins PRIVATE d3d12 dxgi)
foreach(backend IN ITEMS warp hardware)
    add_test(NAME startup_gpu_depth_twins_${backend} COMMAND startup_gpu_depth_twins ${backend})
    set_tests_properties(startup_gpu_depth_twins_${backend} PROPERTIES TIMEOUT 60 SKIP_RETURN_CODE 77)
endforeach()
target_compile_features(startup_gpu_heap_shim PRIVATE cxx_std_17)
target_link_libraries(startup_gpu_heap_shim PRIVATE d3d12_xs_shim d3d12 dxgi)
set_target_properties(startup_gpu_heap_shim PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
add_test(NAME startup_gpu_heap_shim COMMAND startup_gpu_heap_shim)
add_test(NAME startup_gpu_heap_shim_lazy_detile COMMAND startup_gpu_heap_shim --lazy)
set_tests_properties(startup_gpu_heap_shim_lazy_detile PROPERTIES TIMEOUT 120 ENVIRONMENT
    "X2W_GPU_ABI=1;X2W_D3D12_RENDER=1;X2W_D3D12_PERMISSIVE=1;X2W_D3D12_LIST_NOOP=1;X2W_GPU_LATE_TEXTURES=1;X2W_GPU_LAZY_DETILE=1;X2W_MEMORY_SRV_DETILE=1")
add_test(NAME startup_gpu_heap_shim_native_detile COMMAND startup_gpu_heap_shim --native)
set_tests_properties(startup_gpu_heap_shim_native_detile PROPERTIES TIMEOUT 120 ENVIRONMENT
    "X2W_GPU_ABI=0;X2W_D3D12_RENDER=1;X2W_D3D12_PERMISSIVE=1;X2W_D3D12_LIST_NOOP=1;X2W_GPU_LATE_TEXTURES=1;X2W_GPU_LAZY_DETILE=1;X2W_MEMORY_SRV_DETILE=1")
add_test(NAME startup_gpu_heap_shim_mutate COMMAND startup_gpu_heap_shim --mutate)
set_tests_properties(startup_gpu_heap_shim startup_gpu_heap_shim_mutate PROPERTIES
    TIMEOUT 120 ENVIRONMENT "X2W_GPU_ABI=1;X2W_D3D12_RENDER=1;X2W_D3D12_PERMISSIVE=1;X2W_D3D12_LIST_NOOP=1")

add_executable(startup_gpu_trace_request tests/startup_gpu_trace_request.cpp)
target_include_directories(startup_gpu_trace_request PRIVATE src)
target_compile_features(startup_gpu_trace_request PRIVATE cxx_std_17)
add_test(NAME startup_gpu_trace_request COMMAND startup_gpu_trace_request)
