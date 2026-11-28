add_test(NAME startup_debug_noise_check COMMAND startup_debug_noise_check)
foreach(backend IN ITEMS warp hardware)
    add_test(NAME startup_debug_noise_device_${backend} COMMAND startup_debug_noise_device ${backend})
    set_tests_properties(startup_debug_noise_device_${backend} PROPERTIES TIMEOUT 60
        ENVIRONMENT "X2W_D3D12_PERMISSIVE=1;X2W_D3D12_DEBUG=1")
endforeach()
set_tests_properties(startup_debug_noise_device_hardware PROPERTIES SKIP_RETURN_CODE 77)
