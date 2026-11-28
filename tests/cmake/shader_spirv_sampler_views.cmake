# Unit, comment-stripping, table-splitting, refusal, and temp-copy mutation checks for tools/spirv_sampler_views.py.
if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME shader_spirv_sampler_views_check COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/spirv_sampler_views_check.py")
    set_tests_properties(shader_spirv_sampler_views_check PROPERTIES TIMEOUT 60)
endif()
