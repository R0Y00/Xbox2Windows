# Shader front-end modules (native, io, resource, spill, structure, bvh) and
# their tests. Usable on its own (cmake -S src/x2w_shader/front -B out/build/vs-front) or through
# add_subdirectory from a parent that defines the vs-core library.
#
# Standalone, the vs-core sources (lib, dxbc, dxil, ir, gfx10) are compiled from X2W_VS_CORE_DIR
# (default: this tree's src/x2w_shader); a parent can instead set X2W_VS_CORE_TARGET to its library.
cmake_minimum_required(VERSION 3.21)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(x2w_vs_front LANGUAGES CXX)
    enable_testing()
endif()

get_filename_component(x2w_vs_front_src "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
get_filename_component(x2w_vs_front_root "${x2w_vs_front_src}/.." ABSOLUTE)
set(x2w_vs_front_dir "${x2w_vs_front_src}/x2w_shader")
set(x2w_vs_front_tests "${x2w_vs_front_root}/tests/x2w_shader")

function(x2w_vs_front_options target)
    target_include_directories(${target} PUBLIC "${x2w_vs_front_src}")
    target_compile_features(${target} PUBLIC cxx_std_20)
    if(MSVC)
        # /fp:precise keeps multiply and add as separate operations.
        target_compile_options(${target} PRIVATE /W4 /permissive- /utf-8 /EHsc /fp:precise)
        target_compile_definitions(${target} PUBLIC _CRT_SECURE_NO_WARNINGS NOMINMAX)
    else()
        target_compile_options(${target} PRIVATE -ffp-contract=off)
    endif()
endfunction()

set(X2W_VS_CORE_TARGET "" CACHE STRING "Library target with the vs-core modules (empty: build them here)")
set(X2W_VS_CORE_DIR "${x2w_vs_front_dir}" CACHE PATH "Directory with the vs-core sources")
if(X2W_VS_CORE_TARGET)
    set(x2w_vs_front_core ${X2W_VS_CORE_TARGET})
else()
    add_library(x2w_vs_front_core STATIC
        "${X2W_VS_CORE_DIR}/lib.cpp"
        "${X2W_VS_CORE_DIR}/dxbc.cpp"
        "${X2W_VS_CORE_DIR}/dxil.cpp"
        "${X2W_VS_CORE_DIR}/ir.cpp"
        "${X2W_VS_CORE_DIR}/gfx10/fields.cpp"
        "${X2W_VS_CORE_DIR}/gfx10/mod.cpp"
        "${X2W_VS_CORE_DIR}/gfx10/opcodes.cpp"
        "${X2W_VS_CORE_DIR}/gfx10/sp3.cpp"
    )
    # The vs-core headers come from this tree; only the sources from X2W_VS_CORE_DIR.
    target_include_directories(x2w_vs_front_core BEFORE PUBLIC "${x2w_vs_front_src}")
    x2w_vs_front_options(x2w_vs_front_core)
    set(x2w_vs_front_core x2w_vs_front_core)
endif()

add_library(x2w_vs_front STATIC
    "${x2w_vs_front_dir}/bvh.cpp"
    "${x2w_vs_front_dir}/native.cpp"
    "${x2w_vs_front_dir}/structure.cpp"
    "${x2w_vs_front_dir}/spill.cpp"
    "${x2w_vs_front_dir}/io.cpp"
    "${x2w_vs_front_dir}/resource.cpp"
)
x2w_vs_front_options(x2w_vs_front)
target_link_libraries(x2w_vs_front PUBLIC ${x2w_vs_front_core})

# A test executable from tests/x2w_shader/<name>.cpp.
function(x2w_vs_front_test name)
    add_executable(x2w_${name} "${x2w_vs_front_tests}/${name}.cpp")
    x2w_vs_front_options(x2w_${name})
    target_link_libraries(x2w_${name} PRIVATE x2w_vs_front)
    target_include_directories(x2w_${name} PRIVATE "${x2w_vs_front_tests}")
    add_test(NAME x2w_${name} COMMAND x2w_${name})
endfunction()

x2w_vs_front_test(vs_bvh_test)
x2w_vs_front_test(vs_native_test)
x2w_vs_front_test(vs_structure_test)
x2w_vs_front_test(vs_spill_test)
x2w_vs_front_test(vs_io_test)
x2w_vs_front_test(vs_resource_test)
