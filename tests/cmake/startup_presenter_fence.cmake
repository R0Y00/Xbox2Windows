add_test(NAME startup_presenter_fence COMMAND startup_presenter_fence)
foreach(backend IN ITEMS warp hardware)
    add_test(NAME startup_presenter_fence_${backend} COMMAND startup_presenter_fence ${backend})
    set_tests_properties(startup_presenter_fence_${backend} PROPERTIES TIMEOUT 60 SKIP_RETURN_CODE 77
        ENVIRONMENT "X2W_D3D12_PERMISSIVE=1;X2W_D3D12_DEBUG=1")
endforeach()
