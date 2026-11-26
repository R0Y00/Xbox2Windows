# Translation and emission entry points. Requires lowering
# (vs-lower's x2w_vs_emit_lower), the front end (vs-front) and wave/threading.
if(TARGET x2w_vs_emit_lower AND TARGET x2w_vs_emit_wave)
    # The translator's source fingerprint (fingerprint.cmake), made at every build so that an edit of any
    # x2w_shader source changes translation_fingerprint(); the header is rewritten only when it changes.
    file(GLOB_RECURSE x2w_vs_fp_files CONFIGURE_DEPENDS "${x2w_vs_src}/x2w_shader/*.h" "${x2w_vs_src}/x2w_shader/*.cpp")
    set(x2w_vs_fp_dir "${CMAKE_CURRENT_BINARY_DIR}/x2w_vs_fingerprint")
    add_custom_command(OUTPUT "${x2w_vs_fp_dir}/x2w_vs_fingerprint.stamp"
        COMMAND "${CMAKE_COMMAND}" "-DSRC=${x2w_vs_src}" "-DOUT=${x2w_vs_fp_dir}/x2w_vs_fingerprint.h"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/fingerprint.cmake"
        COMMAND "${CMAKE_COMMAND}" -E touch "${x2w_vs_fp_dir}/x2w_vs_fingerprint.stamp"
        DEPENDS ${x2w_vs_fp_files} "${CMAKE_CURRENT_SOURCE_DIR}/fingerprint.cmake"
        BYPRODUCTS "${x2w_vs_fp_dir}/x2w_vs_fingerprint.h"
        COMMENT "x2w_shader translator fingerprint")
    add_custom_target(x2w_vs_fingerprint DEPENDS "${x2w_vs_fp_dir}/x2w_vs_fingerprint.stamp")

    add_library(x2w_vs_emit STATIC "${CMAKE_CURRENT_SOURCE_DIR}/mod.cpp")
    x2w_vs_emit_options(x2w_vs_emit)
    add_dependencies(x2w_vs_emit x2w_vs_fingerprint)
    target_include_directories(x2w_vs_emit PRIVATE "${x2w_vs_fp_dir}")
    target_compile_definitions(x2w_vs_emit PRIVATE X2W_VS_FINGERPRINT_HEADER)
    target_link_libraries(x2w_vs_emit PUBLIC x2w_vs_emit_lower x2w_vs_emit_wave x2w_vs_emit_dxil)
endif()
