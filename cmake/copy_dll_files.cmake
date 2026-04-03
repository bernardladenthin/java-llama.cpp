# Script to find and copy DLL/SO/DYLIB files from llama.cpp build directory
# Run as: cmake -P copy_dll_files.cmake
# Requires CMake variables to be passed via -D flags:
#   - SOURCE_DIR: project root
#   - OS_NAME: operating system name (Windows, Linux, Mac)
#   - OS_ARCH: architecture (x86_64, aarch64, x86, etc.)
#   - BINARY_DIR: llama.cpp build directory

# Validate required variables
if(NOT DEFINED SOURCE_DIR)
    message(FATAL_ERROR "SOURCE_DIR not defined")
endif()
if(NOT DEFINED OS_NAME)
    message(FATAL_ERROR "OS_NAME not defined")
endif()
if(NOT DEFINED OS_ARCH)
    message(FATAL_ERROR "OS_ARCH not defined")
endif()
if(NOT DEFINED BINARY_DIR)
    message(FATAL_ERROR "BINARY_DIR not defined")
endif()

# Determine output directory
if(DEFINED GGML_CUDA AND GGML_CUDA)
    set(OUTPUT_DIR "${SOURCE_DIR}/src/main/resources_linux_cuda/de/kherud/llama/${OS_NAME}/${OS_ARCH}")
else()
    set(OUTPUT_DIR "${SOURCE_DIR}/src/main/resources/de/kherud/llama/${OS_NAME}/${OS_ARCH}")
endif()

# Find DLL/SO/DYLIB files in common build locations
set(SEARCH_DIRS
    "${BINARY_DIR}/bin"
    "${BINARY_DIR}/lib"
    "${BINARY_DIR}/Release"
    "${BINARY_DIR}/Debug"
    "${BINARY_DIR}/_deps"
)

message(STATUS "[DLL Discovery] Searching for DLL/SO/DYLIB files")
message(STATUS "[DLL Discovery] OS: ${OS_NAME}/${OS_ARCH}")
message(STATUS "[DLL Discovery] Output: ${OUTPUT_DIR}")

# Platform-specific library patterns
if(OS_NAME STREQUAL "Windows")
    set(LIB_PATTERNS "*.dll" "*.lib")
elseif(OS_NAME STREQUAL "Mac")
    set(LIB_PATTERNS "*.dylib" "*.a")
else()
    # Linux
    set(LIB_PATTERNS "*.so*" "*.a")
endif()

# Find matching files
set(FOUND_FILES)
foreach(SEARCH_DIR ${SEARCH_DIRS})
    if(EXISTS ${SEARCH_DIR})
        foreach(PATTERN ${LIB_PATTERNS})
            file(GLOB_RECURSE MATCHES "${SEARCH_DIR}/${PATTERN}")
            list(APPEND FOUND_FILES ${MATCHES})
        endforeach()
    endif()
endforeach()

# Filter out CMake-generated files and duplicates
list(REMOVE_DUPLICATES FOUND_FILES)

if(FOUND_FILES)
    message(STATUS "[DLL Discovery] Found ${FOUND_FILES} files:")
    foreach(FILE ${FOUND_FILES})
        get_filename_component(FILENAME "${FILE}" NAME)
        set(DEST "${OUTPUT_DIR}/${FILENAME}")

        # Copy file
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${FILE}" "${DEST}"
            RESULT_VARIABLE COPY_RESULT
        )

        if(COPY_RESULT EQUAL 0)
            message(STATUS "[DLL Discovery] ✓ ${FILENAME}")
        else()
            message(WARNING "[DLL Discovery] ✗ Failed to copy ${FILENAME}")
        endif()
    endforeach()
else()
    message(STATUS "[DLL Discovery] No DLL/SO/DYLIB files found (normal for static builds)")
endif()

