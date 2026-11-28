if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME runtime_ui_regression_matrix_cli_check COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/startup_ui_regression_matrix_cli_check.py")
    set_tests_properties(runtime_ui_regression_matrix_cli_check PROPERTIES TIMEOUT 60)
endif()
