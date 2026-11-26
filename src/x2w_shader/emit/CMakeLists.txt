# DXIL writer for the shader translator.
# Usable on its own (cmake -S src/x2w_shader/emit -B out/build/vs-emit) or through
# add_subdirectory from a parent that defines the other x2w_shader packages.
cmake_minimum_required(VERSION 3.21)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(x2w_vs_emit LANGUAGES CXX)
    enable_testing()
endif()

get_filename_component(x2w_vs_src "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
get_filename_component(x2w_vs_root "${x2w_vs_src}/.." ABSOLUTE)
set(x2w_vs_emit_tests "${x2w_vs_root}/tests/x2w_shader/emit")

function(x2w_vs_emit_options target)
    target_include_directories(${target} PUBLIC "${x2w_vs_src}")
    target_compile_features(${target} PUBLIC cxx_std_20)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive- /utf-8 /EHsc)
        target_compile_definitions(${target} PUBLIC _CRT_SECURE_NO_WARNINGS NOMINMAX)
    endif()
endfunction()

# A test executable from tests/x2w_shader/emit/<name>.cpp linked to `libs`.
function(x2w_vs_emit_test name)
    add_executable(x2w_vs_${name} "${x2w_vs_emit_tests}/${name}.cpp")
    x2w_vs_emit_options(x2w_vs_${name})
    target_link_libraries(x2w_vs_${name} PRIVATE ${ARGN})
    target_include_directories(x2w_vs_${name} PRIVATE "${x2w_vs_emit_tests}" "${x2w_vs_emit_tests}/..")
    add_test(NAME x2w_vs_${name} COMMAND x2w_vs_${name})
endfunction()

# The other x2w_shader packages when built standalone (the root CMakeLists defines x2w_shader).
if(NOT TARGET x2w_shader)
    add_library(x2w_shader STATIC
        "${x2w_vs_src}/x2w_shader/lib.cpp"
        "${x2w_vs_src}/x2w_shader/dxbc.cpp"
        "${x2w_vs_src}/x2w_shader/dxil.cpp"
        "${x2w_vs_src}/x2w_shader/ir.cpp"
        "${x2w_vs_src}/x2w_shader/gfx10/mod.cpp"
        "${x2w_vs_src}/x2w_shader/gfx10/fields.cpp"
        "${x2w_vs_src}/x2w_shader/gfx10/opcodes.cpp"
        "${x2w_vs_src}/x2w_shader/gfx10/sp3.cpp")
    # vs-front's sources, as far as they are merged.
    foreach(f bvh native structure spill io resource)
        if(EXISTS "${x2w_vs_src}/x2w_shader/${f}.cpp")
            target_sources(x2w_shader PRIVATE "${x2w_vs_src}/x2w_shader/${f}.cpp")
        endif()
    endforeach()
    target_include_directories(x2w_shader PUBLIC "${x2w_vs_src}")
    target_compile_features(x2w_shader PUBLIC cxx_std_20)
endif()

# The parts that need no other x2w_shader package.
add_library(x2w_vs_emit_base STATIC
    bitcode.cpp
    census.cpp
    sampler.cpp
)
x2w_vs_emit_options(x2w_vs_emit_base)

x2w_vs_emit_test(bitcode_test x2w_vs_emit_base)
x2w_vs_emit_test(sampler_test x2w_vs_emit_base)

# Per-module fragments: opt.cmake, container.cmake,
# wave.cmake, validator.cmake, host.cmake, ...
# -DX2W_VS_EMIT_ONLY=opt;wave limits the build to those fragments (each helper builds its own
# while the others are in progress).
set(X2W_VS_EMIT_ONLY "" CACHE STRING "Fragments to include (empty: all)")
if(X2W_VS_EMIT_ONLY)
    set(x2w_vs_emit_fragments "")
    foreach(f IN LISTS X2W_VS_EMIT_ONLY)
        list(APPEND x2w_vs_emit_fragments "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${f}.cmake")
    endforeach()
else()
    file(GLOB x2w_vs_emit_fragments "${CMAKE_CURRENT_SOURCE_DIR}/cmake/*.cmake")
    list(SORT x2w_vs_emit_fragments)
endif()
foreach(fragment IN LISTS x2w_vs_emit_fragments)
    include("${fragment}")
endforeach()
