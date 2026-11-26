# tools/startup/debug_launch.py rejects malformed --timeout, --memory-gib and --continue-asserts
# values with an "expected ..." message and exit code 2 before it launches anything.
if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME startup_debug_launch_args COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/startup_debug_launch_args_check.py")
    set_tests_properties(startup_debug_launch_args PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
endif()
