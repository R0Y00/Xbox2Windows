# Whole-wave sections, jump threading, and per-lane bit tracking.
#
# x2w_vs_emit_wave needs vs-core's ir (x2w_shader); threading.cpp and own_bits.cpp also need
# vs-front's structure.h/.cpp (in x2w_shader), and own_bits.cpp vs-lower's
# lower::reads_mask_as_data (x2w_vs_emit_lower, whose fragment lower.cmake comes before this one).
# What is not in the tree yet is left out, with a message.

set(x2w_vs_wave_sources wave.cpp)
set(x2w_vs_wave_have_structure OFF)
if(EXISTS "${x2w_vs_src}/x2w_shader/structure.h")
    set(x2w_vs_wave_have_structure ON)
    list(APPEND x2w_vs_wave_sources threading.cpp own_bits.cpp)
else()
    message(STATUS "x2w_vs_emit_wave: no x2w_shader/structure.h (vs-front) yet: threading and own_bits left out")
endif()
add_library(x2w_vs_emit_wave STATIC ${x2w_vs_wave_sources})
x2w_vs_emit_options(x2w_vs_emit_wave)
target_link_libraries(x2w_vs_emit_wave PUBLIC x2w_vs_emit_base x2w_shader)

x2w_vs_emit_test(wave_test x2w_vs_emit_wave)
# (all of wave_test's tests: ir::build is in x2w_shader)
target_compile_definitions(x2w_vs_wave_test PRIVATE X2W_VS_HAVE_IR)

# threading_test and own_bits_test need structure.cpp linked (in x2w_shader).
set(x2w_vs_wave_have_structure_impl OFF)
if(x2w_vs_wave_have_structure)
    get_target_property(x2w_vs_shader_sources x2w_shader SOURCES)
    foreach(s IN LISTS x2w_vs_shader_sources)
        if(s MATCHES "structure\\.cpp$")
            set(x2w_vs_wave_have_structure_impl ON)
        endif()
    endforeach()
endif()
if(x2w_vs_wave_have_structure_impl)
    x2w_vs_emit_test(threading_test x2w_vs_emit_wave)
    if(TARGET x2w_vs_emit_lower)
        x2w_vs_emit_test(own_bits_test x2w_vs_emit_wave x2w_vs_emit_lower)
    else()
        message(STATUS "x2w_vs_emit_wave: no x2w_vs_emit_lower (vs-lower) yet: own_bits_test left out")
    endif()
elseif(x2w_vs_wave_have_structure)
    message(STATUS "x2w_vs_emit_wave: structure.cpp (vs-front) not in x2w_shader: threading_test, own_bits_test left out")
endif()

# The lowering reads wave sections and own bits, and own_bits.cpp reads the lowering's
# reads_mask_as_data: the two static libraries depend on each other (CMake repeats them on links).
if(TARGET x2w_vs_emit_lower)
    target_link_libraries(x2w_vs_emit_lower PUBLIC x2w_vs_emit_wave)
    target_link_libraries(x2w_vs_emit_wave PUBLIC x2w_vs_emit_lower)
endif()
