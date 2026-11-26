# CTest registration fragments

Every `tests/cmake/*.cmake` file is included by `CMakeLists.txt` after the existing test
registrations (in name order, and a new file triggers a re-configure). Put new `add_test` and
`set_tests_properties` calls for a family in `tests/cmake/<family>.cmake`, one file per family
or pull request, so two pull requests do not edit the same lines of `CMakeLists.txt`. A
fragment sees the targets, the `INTAKE_PYTHON` variable and the `add_test` wrapper defined above
it, and can set a `COST` property for a test that is not in `X2W_SLOW_TEST_COSTS`. Existing
registrations stay where they are.
