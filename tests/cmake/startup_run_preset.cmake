# Shared run presets (tools/startup/run_preset.py, tools/startup/run_presets.json): every console x OS x GPU
# choice names only switches that run.ps1 and run.sh define; staging is checked on a synthetic package.
if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME startup_run_preset COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_run_preset_check.py")
    set_tests_properties(startup_run_preset PROPERTIES TIMEOUT 60)
endif()
