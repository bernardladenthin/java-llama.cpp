# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# jllama-build-info.cmake -- what every natives directory says about the build it came from.
#
# Two files are written next to the library (or the module) of every build, both read by people and
# by scripts, neither extracted or loaded by LlamaLoader:
#
#   jllama-build.txt   key=value lines. `llama.cpp` (the b<nnnn> number, the loader's same-build
#                      check), `llama.cpp.commit` (the upstream commit actually fetched), `backend`,
#                      `kind`, `jllama.version` (the net.ladenthin:llama version from the pom),
#                      `jllama.commit` (this repository's commit, `-dirty` when the tree had local
#                      changes), `patches` (the ids of llama/patches/*.patch, in order),
#                      `patches.sha256` (one hash over the patch set: "<name> <sha256>\n" per patch,
#                      in name order -- the same lines the patch applier's stamp carries), `compiler`
#                      and `toolkit` (the vendor SDK a GPU module was built against), and `signed`
#                      (`no` until the signing step exists -- TODO.md, delivery roadmap, item 9).
#   jllama-sbom.json   a CycloneDX 1.5 bill of materials of the same facts: the natives jar as the
#                      component, llama.cpp with every carried patch as its pedigree, BoringSSL,
#                      CCCL and the vendor toolkit as components.
#
# Why: the `.asc` on a natives jar proves who signed it, not what is inside. A consumer who asks
# "which llama.cpp, with which of this project's patches, built by which compiler" must be able to
# read it off the artefact -- that is what makes "we ship more than upstream" verifiable instead of
# a claim (TODO.md, delivery roadmap, item 1). merge-native-artifacts.sh requires the keys and the
# BOM of every artifact; LlamaLoader.getBuildInfo() and DiagnosticReport expose the keys.
#
# Call jllama_write_build_info() after the backend directory (JLLAMA_DIR) and the backend name
# (JLLAMA_BACKEND) are known and after llama.cpp was fetched (llama.cpp_SOURCE_DIR).

# The classifier of a natives jar from its directory, the inverse of the natives.csv rows
# (`cpu-linux-x86-64` <- `Linux/x86_64/cpu`). Fails loud on an OS or arch the list does not know.
function(jllama_classifier_of out_var os_name os_arch backend)
    if(os_name STREQUAL "Linux")
        set(os linux)
    elseif(os_name STREQUAL "Linux-Android")
        set(os android)
    elseif(os_name STREQUAL "Windows")
        set(os windows)
    elseif(os_name STREQUAL "Mac")
        set(os macos)
    else()
        message(FATAL_ERROR "jllama-build-info: no classifier for OS_NAME '${os_name}' (natives.csv knows Linux, Linux-Android, Windows, Mac)")
    endif()
    string(REPLACE "_" "-" arch "${os_arch}")
    set(${out_var} "${backend}-${os}-${arch}" PARENT_SCOPE)
endfunction()

# The first line of `<program> --version`, or "" when it cannot be run.
function(jllama_first_version_line out_var program)
    set(${out_var} "" PARENT_SCOPE)
    if(NOT program)
        return()
    endif()
    execute_process(COMMAND "${program}" --version
                    OUTPUT_VARIABLE text ERROR_QUIET RESULT_VARIABLE rc
                    OUTPUT_STRIP_TRAILING_WHITESPACE TIMEOUT 20)
    if(NOT rc EQUAL 0)
        return()
    endif()
    string(REGEX REPLACE "\n.*" "" first "${text}")
    set(${out_var} "${first}" PARENT_SCOPE)
endfunction()

# The vendor toolkit a GPU module is built against, as "<name> <version>", or "none" for a CPU
# build and "<name> unknown" where the version cannot be read. Each probe is guarded: a missing
# answer is written as unknown, never as a configure error -- the build itself has already proven
# the toolkit is there.
function(jllama_toolkit_version out_var)
    set(result "none")
    if(GGML_CUDA)
        set(result "cuda unknown")
        set(nvcc "${CMAKE_CUDA_COMPILER}")
        if(NOT nvcc)
            find_program(_jllama_nvcc nvcc)
            set(nvcc "${_jllama_nvcc}")
        endif()
        if(nvcc)
            execute_process(COMMAND "${nvcc}" --version OUTPUT_VARIABLE text ERROR_QUIET
                            RESULT_VARIABLE rc OUTPUT_STRIP_TRAILING_WHITESPACE TIMEOUT 20)
            if(rc EQUAL 0 AND text MATCHES "release ([0-9]+\\.[0-9]+)")
                set(result "cuda ${CMAKE_MATCH_1}")
            endif()
        endif()
    elseif(GGML_HIP)
        set(result "rocm unknown")
        # Linux compiles HIP with CMAKE_HIP_COMPILER, Windows as C++ with TheRock's clang.
        if(CMAKE_HIP_COMPILER)
            jllama_first_version_line(line "${CMAKE_HIP_COMPILER}")
        else()
            jllama_first_version_line(line "${CMAKE_CXX_COMPILER}")
        endif()
        if(line MATCHES "roc-([0-9]+\\.[0-9]+(\\.[0-9]+)?)")
            set(result "rocm ${CMAKE_MATCH_1}")
        elseif(line)
            set(result "rocm ${line}")
        endif()
    elseif(GGML_SYCL)
        set(result "oneapi ${CMAKE_CXX_COMPILER_VERSION}")
    elseif(GGML_VULKAN)
        set(result "vulkan unknown")
        find_package(Vulkan QUIET)
        if(Vulkan_VERSION)
            set(result "vulkan ${Vulkan_VERSION}")
        endif()
    elseif(GGML_OPENCL)
        set(result "opencl unknown")
        find_package(OpenCL QUIET)
        if(OpenCL_VERSION_STRING)
            set(result "opencl ${OpenCL_VERSION_STRING}")
        endif()
    elseif(GGML_OPENVINO)
        set(result "openvino unknown")
        find_package(OpenVINO QUIET)
        if(OpenVINO_VERSION)
            set(result "openvino ${OpenVINO_VERSION}")
        endif()
    elseif(APPLE AND GGML_METAL)
        set(result "metal ${CMAKE_OSX_DEPLOYMENT_TARGET}")
    endif()
    set(${out_var} "${result}" PARENT_SCOPE)
endfunction()

# The commit of a git work tree ("unknown" without git or outside a checkout), with "-dirty"
# appended when tracked files are modified. `safe.directory=*` because CI mounts the checkout into
# containers under another uid, where git refuses to read it otherwise.
function(jllama_git_commit out_var dir)
    set(${out_var} "unknown" PARENT_SCOPE)
    if(NOT GIT_EXECUTABLE)
        return()
    endif()
    execute_process(COMMAND "${GIT_EXECUTABLE}" -c safe.directory=* -C "${dir}" rev-parse HEAD
                    OUTPUT_VARIABLE head ERROR_QUIET RESULT_VARIABLE rc OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT rc EQUAL 0 OR NOT head MATCHES "^[0-9a-f]+$")
        return()
    endif()
    execute_process(COMMAND "${GIT_EXECUTABLE}" -c safe.directory=* -C "${dir}" status --porcelain --untracked-files=no
                    OUTPUT_VARIABLE status ERROR_QUIET RESULT_VARIABLE status_rc OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(status_rc EQUAL 0 AND NOT status STREQUAL "")
        set(head "${head}-dirty")
    endif()
    set(${out_var} "${head}" PARENT_SCOPE)
endfunction()

function(jllama_write_build_info)
    # --- llama.cpp: the number (the pin) and the commit that was fetched for it
    set(llama_tag "${JLLAMA_GIT_TAG}")
    set(llama_commit "unknown")
    if(GIT_EXECUTABLE AND EXISTS "${llama.cpp_SOURCE_DIR}/.git")
        execute_process(COMMAND "${GIT_EXECUTABLE}" -c safe.directory=* -C "${llama.cpp_SOURCE_DIR}" rev-parse HEAD
                        OUTPUT_VARIABLE llama_commit ERROR_QUIET RESULT_VARIABLE rc OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(NOT rc EQUAL 0)
            set(llama_commit "unknown")
        endif()
    endif()
    if(llama_commit STREQUAL "unknown" AND JLLAMA_GIT_COMMIT)
        set(llama_commit "${JLLAMA_GIT_COMMIT}")
    endif()

    # --- this project: version from the pom, commit from git
    set(JLLAMA_VERSION "" CACHE STRING "The net.ladenthin:llama version this build belongs to (default: the <parent> version of llama/pom.xml)")
    set(jllama_version "${JLLAMA_VERSION}")
    if(NOT jllama_version)
        # The module declares no version of its own: the first <version> of llama/pom.xml is the
        # <parent> pointer, which `mvn versions:set` keeps equal to the reactor's version.
        file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/pom.xml" pom_versions REGEX "<version>[^<]+</version>")
        list(LENGTH pom_versions n)
        if(n EQUAL 0)
            message(FATAL_ERROR "jllama-build-info: no <version> in ${CMAKE_CURRENT_SOURCE_DIR}/pom.xml")
        endif()
        list(GET pom_versions 0 first)
        string(REGEX REPLACE ".*<version>([^<]+)</version>.*" "\\1" jllama_version "${first}")
    endif()
    if(NOT jllama_version MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+(-SNAPSHOT)?$")
        message(FATAL_ERROR "jllama-build-info: '${jllama_version}' is not a net.ladenthin:llama version (x.y.z or x.y.z-SNAPSHOT); set -DJLLAMA_VERSION=")
    endif()
    jllama_git_commit(jllama_commit "${CMAKE_CURRENT_SOURCE_DIR}")
    if(jllama_commit STREQUAL "unknown" AND DEFINED ENV{GITHUB_SHA})
        set(jllama_commit "$ENV{GITHUB_SHA}")
    endif()

    # --- the patch set: ids in order, one hash over names and contents
    file(GLOB patch_files "${CMAKE_CURRENT_SOURCE_DIR}/patches/*.patch" "${CMAKE_CURRENT_SOURCE_DIR}/patches/*.diff")
    list(SORT patch_files)
    set(patch_ids "")
    set(patch_manifest "")
    set(patch_pedigree "")
    foreach(patch IN LISTS patch_files)
        get_filename_component(name "${patch}" NAME)
        string(REGEX REPLACE "^([0-9]+)-.*" "\\1" id "${name}")
        list(APPEND patch_ids "${id}")
        file(SHA256 "${patch}" patch_hash)
        string(APPEND patch_manifest "${name} ${patch_hash}\n")
        string(REGEX REPLACE "-dirty$" "" pedigree_ref "${jllama_commit}")
        string(APPEND patch_pedigree
               "      {\"type\": \"unofficial\", \"diff\": {\"url\": \"https://github.com/bernardladenthin/java-llama.cpp/blob/${pedigree_ref}/llama/patches/${name}\"}, \"resolves\": [{\"type\": \"defect\", \"name\": \"${name}\", \"description\": \"sha256 ${patch_hash}\"}]},\n")
    endforeach()
    string(JOIN "," patch_ids_csv ${patch_ids})
    string(SHA256 patches_sha256 "${patch_manifest}")
    string(REGEX REPLACE ",\n$" "\n" patch_pedigree "${patch_pedigree}")

    # --- toolchain
    set(compiler "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
    jllama_toolkit_version(toolkit)
    if(JLLAMA_MODULE_ONLY)
        set(kind module)
    else()
        set(kind library)
    endif()
    jllama_classifier_of(classifier "${OS_NAME}" "${OS_ARCH}" "${JLLAMA_BACKEND}")

    file(GENERATE OUTPUT ${JLLAMA_DIR}/jllama-build.txt CONTENT
         "# Written by llama/cmake/jllama-build-info.cmake: the build this directory came from (read by LlamaLoader)
llama.cpp=${llama_tag}
llama.cpp.commit=${llama_commit}
backend=${JLLAMA_BACKEND}
kind=${kind}
classifier=${classifier}
jllama.version=${jllama_version}
jllama.commit=${jllama_commit}
patches=${patch_ids_csv}
patches.sha256=${patches_sha256}
compiler=${compiler}
toolkit=${toolkit}
signed=no
"
         NEWLINE_STYLE LF)

    # --- the BOM: the same facts as CycloneDX 1.5 JSON
    set(components "")
    string(APPEND components
           "    {\"type\": \"library\", \"name\": \"llama.cpp\", \"version\": \"${llama_tag}\", \"purl\": \"pkg:github/ggml-org/llama.cpp@${llama_commit}\", \"pedigree\": {\"patches\": [\n${patch_pedigree}    ]}}")
    if(DEFINED BORINGSSL_VERSION AND LLAMA_BUILD_BORINGSSL)
        string(APPEND components ",\n    {\"type\": \"library\", \"name\": \"boringssl\", \"version\": \"${BORINGSSL_VERSION}\", \"purl\": \"pkg:generic/boringssl@${BORINGSSL_VERSION}\"}")
    endif()
    if(GGML_CUDA AND GGML_CUDA_CCCL_VERSION)
        string(APPEND components ",\n    {\"type\": \"library\", \"name\": \"cccl\", \"version\": \"${GGML_CUDA_CCCL_VERSION}\", \"purl\": \"pkg:github/NVIDIA/cccl@${GGML_CUDA_CCCL_VERSION}\"}")
    endif()
    if(NOT toolkit STREQUAL "none")
        string(REGEX REPLACE " .*" "" toolkit_name "${toolkit}")
        string(REGEX REPLACE "^[^ ]+ ?" "" toolkit_version "${toolkit}")
        string(REPLACE "\"" "'" toolkit_version "${toolkit_version}")
        string(REPLACE "\\" "/" toolkit_version "${toolkit_version}")
        string(APPEND components ",\n    {\"type\": \"library\", \"name\": \"${toolkit_name}\", \"version\": \"${toolkit_version}\", \"scope\": \"required\"}")
    endif()
    string(REPLACE "\\" "/" compiler_json "${compiler}")
    file(GENERATE OUTPUT ${JLLAMA_DIR}/jllama-sbom.json CONTENT
         "{
  \"bomFormat\": \"CycloneDX\",
  \"specVersion\": \"1.5\",
  \"version\": 1,
  \"metadata\": {
    \"component\": {\"type\": \"library\", \"group\": \"net.ladenthin\", \"name\": \"llama\", \"version\": \"${jllama_version}\", \"purl\": \"pkg:maven/net.ladenthin/llama@${jllama_version}?classifier=${classifier}\"},
    \"properties\": [
      {\"name\": \"jllama.commit\", \"value\": \"${jllama_commit}\"},
      {\"name\": \"jllama.backend\", \"value\": \"${JLLAMA_BACKEND}\"},
      {\"name\": \"jllama.kind\", \"value\": \"${kind}\"},
      {\"name\": \"jllama.compiler\", \"value\": \"${compiler_json}\"},
      {\"name\": \"jllama.patches.sha256\", \"value\": \"${patches_sha256}\"}
    ]
  },
  \"components\": [
${components}
  ]
}
"
         NEWLINE_STYLE LF)
    message(STATUS "jllama-build-info: ${classifier} = llama.cpp ${llama_tag} (${llama_commit}), jllama ${jllama_version} (${jllama_commit}), patches ${patch_ids_csv}, ${compiler}, toolkit ${toolkit}")
endfunction()
