# libzonos3_c: the zonos3.cpp core (backbone, waveform decoder, speaker encoder, LACI) behind a C interface.
#
# Copyright 2026 Subtitld contributors
# SPDX-License-Identifier: Apache-2.0
#
# Builds only what synthesis needs, from a ZONOS3-Mini checkout at the commit tools/build_native.py pins: no HTTP
# server, no cargo / zyphra-textnorm (the add-on normalizes text in Python with the zyphra-textnorm wheel). ggml is
# linked in statically, so the add-on ships one library, which exports only the zs_* functions of zonos3_c.h.
#
#     cmake -S native -B build/native -DZONOS3_DIR=<checkout>
#     cmake --build build/native --config Release
cmake_minimum_required(VERSION 3.18)
project(zonos3_c C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# The MSVC runtime linked in, so the add-on ships no msvcp140.dll of its own.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(ZONOS3_DIR "" CACHE PATH "ZONOS3-Mini checkout")
if (NOT EXISTS ${ZONOS3_DIR}/servers/ggml/src/zonos3.h)
    message(FATAL_ERROR "ZONOS3_DIR=${ZONOS3_DIR} is not a ZONOS3-Mini checkout")
endif()
set(G ${ZONOS3_DIR}/servers/ggml)
set(S ${G}/src)
set(REG ${ZONOS3_DIR}/model/src/zonos3/registries)

# ggml with Zyphra's patches, applied the way upstream's CMakeLists.txt does (idempotently).
find_package(Git REQUIRED)
file(GLOB GGML_PATCHES ${G}/patches/*.patch)
list(SORT GGML_PATCHES)
foreach(_patch ${GGML_PATCHES})
    execute_process(COMMAND ${GIT_EXECUTABLE} -C ${G}/ggml apply --reverse --check ${_patch}
                    RESULT_VARIABLE _applied OUTPUT_QUIET ERROR_QUIET)
    if (NOT _applied EQUAL 0)
        execute_process(COMMAND ${GIT_EXECUTABLE} -C ${G}/ggml apply ${_patch} RESULT_VARIABLE _rc)
        if (NOT _rc EQUAL 0)
            message(FATAL_ERROR "Failed to apply ${_patch}")
        endif()
    endif()
endforeach()

# CPU only, one static ggml. x86-64 builds target x86-64-v3 (AVX2, FMA, F16C: ggml's defaults without
# GGML_NATIVE); the core calls the CPU backend directly, so ggml's per-CPU variants (loaded as plugins) are no option.
# ggml's own thread pool instead of OpenMP: no libgomp or vcomp140 to ship.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${G}/ggml ggml)

function(zonos3_embed_json source symbol header)
    file(READ ${source} hex HEX)
    string(LENGTH "${hex}" hex_len)
    math(EXPR ZONOS3_EMBED_LEN "${hex_len} / 2")
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," ZONOS3_EMBED_BYTES "${hex}")
    file(RELATIVE_PATH ZONOS3_EMBED_SOURCE ${ZONOS3_DIR} ${source})
    set(ZONOS3_EMBED_SYMBOL ${symbol})
    configure_file(${S}/embed-json.h.in ${CMAKE_CURRENT_BINARY_DIR}/generated/${header} @ONLY)
endfunction()
zonos3_embed_json(${REG}/languages.json zonos3_languages_json zonos3-languages-json.h)
zonos3_embed_json(${REG}/generation.json zonos3_generation_json zonos3-generation-json.h)
zonos3_embed_json(${REG}/short_text.json zonos3_short_text_json zonos3-short-text-json.h)

# The pronunciation lexicon's LMDB (OpenLDAP Public License), which zonos3-phonemes.cpp opens only when asked to.
find_package(Threads REQUIRED)
add_library(zonos3-lmdb STATIC ${G}/vendor/lmdb/mdb.c ${G}/vendor/lmdb/midl.c)
target_include_directories(zonos3-lmdb PUBLIC ${G}/vendor/lmdb)
target_link_libraries(zonos3-lmdb PUBLIC Threads::Threads)
if (MSVC)
    target_compile_options(zonos3-lmdb PRIVATE /w)
else()
    target_compile_options(zonos3-lmdb PRIVATE -w)
endif()

add_library(zonos3_c SHARED
    zonos3_c.cpp
    ${S}/zonos3-text.cpp
    ${S}/zonos3.cpp ${S}/zonos3-graph.cpp ${S}/zonos3-sampler.cpp ${S}/zonos3-prompt.cpp ${S}/zonos3-tokenizer.cpp
    ${S}/zonos3-steer.cpp ${S}/zonos3-languages.cpp ${S}/zonos3-generation.cpp ${S}/zonos3-phonemes.cpp
    ${S}/zonos3-laci.cpp ${S}/zonos3-span-follow.cpp ${S}/zonos3-short-text.cpp ${S}/zonos3-accent.cpp
    ${S}/zonos3-edit.cpp
    ${S}/zac.cpp ${S}/spk-encoder.cpp)
target_include_directories(zonos3_c PRIVATE ${S} ${G}/vendor ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_link_libraries(zonos3_c PRIVATE ggml zonos3-lmdb ${CMAKE_DL_LIBS})
set(ZS_UPSTREAM_COMMIT "unknown" CACHE STRING "the ZONOS3-Mini commit built, for zs_version()")
target_compile_definitions(zonos3_c PRIVATE ZS_BUILDING "ZS_UPSTREAM_COMMIT=\"${ZS_UPSTREAM_COMMIT}\"")
set_target_properties(zonos3_c PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
if (APPLE)
    # The decoder's convolutions call Accelerate's sgemm (src/zac.cpp). Export only the C interface.
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    target_link_libraries(zonos3_c PRIVATE ${ACCELERATE_FRAMEWORK})
    target_link_options(zonos3_c PRIVATE "LINKER:-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/exports.macos")
elseif (UNIX)
    # ggml's archives were built with default visibility: keep their symbols out of the dynamic table, and the
    # GCC runtime (the toolset's newer libstdc++ parts) static.
    target_link_options(zonos3_c PRIVATE "LINKER:--exclude-libs,ALL" "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.linux"
                        -static-libgcc)
endif()
