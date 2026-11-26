# WP3 vbind (PLAN.md WP3): Root arguments, binding and draw state. Included from CMakeLists.txt.
#
# Header checks (no device): the per-list root record and the console topologies (startup_gpu_roots_check); user
# data, table planning, raw view bounds, find_bounded, RecordHash, dispatch arithmetic (startup_gpu_user_data_check).
# Each has a --mutate negative control that must fail.
foreach(check IN ITEMS roots user_data)
    add_executable(startup_gpu_${check}_check tests/startup_gpu_${check}_check.cpp)
    target_include_directories(startup_gpu_${check}_check PRIVATE src)
    target_compile_features(startup_gpu_${check}_check PRIVATE cxx_std_17)
    add_test(NAME startup_gpu_${check} COMMAND startup_gpu_${check}_check)
    add_test(NAME startup_gpu_${check}_mutate_control COMMAND startup_gpu_${check}_check --mutate)
    set_tests_properties(startup_gpu_${check}_mutate_control PROPERTIES WILL_FAIL TRUE)
    set_tests_properties(startup_gpu_${check} startup_gpu_${check}_mutate_control PROPERTIES TIMEOUT 30)
endforeach()

# WARP binding checks with our own ABI shaders (tests/startup_gpu_bind.hlsl, cs_6_6, ResourceDescriptorHeap):
# startup_gpu_bind (table dword in console units, raw views from the 16 bytes below with the bias, a CBV made afresh
# in 256-byte steps) and startup_gpu_bounds (a root UAV bounded by its title buffer, null over unknown memory, cut at
# the window's end). Each has a --mutate negative control. 77: no WARP / SM 6.6 / binding tier 3.
add_executable(startup_gpu_bind_check tests/startup_gpu_bind_check.cpp)
foreach(entry IN ITEMS bind_main bounds_main)
    set(header "${CMAKE_CURRENT_BINARY_DIR}/startup_gpu_bind_${entry}.h")
    add_custom_command(OUTPUT "${header}" COMMAND "${DXC_EXECUTABLE}" -T cs_6_6 -E ${entry}
        -Vn startup_gpu_bind_${entry} -Fh "${header}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gpu_bind.hlsl"
        DEPENDS tests/startup_gpu_bind.hlsl VERBATIM)
    target_sources(startup_gpu_bind_check PRIVATE "${header}")
endforeach()
target_include_directories(startup_gpu_bind_check PRIVATE src "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_features(startup_gpu_bind_check PRIVATE cxx_std_17)
target_link_libraries(startup_gpu_bind_check PRIVATE d3d12 dxgi)
foreach(mode IN ITEMS bind bounds)
    add_test(NAME startup_gpu_${mode} COMMAND startup_gpu_bind_check ${mode})
    add_test(NAME startup_gpu_${mode}_mutate_control COMMAND startup_gpu_bind_check ${mode} --mutate)
    set_tests_properties(startup_gpu_${mode} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
    set_tests_properties(startup_gpu_${mode}_mutate_control PROPERTIES WILL_FAIL TRUE SKIP_RETURN_CODE 77 TIMEOUT 60)
endforeach()
