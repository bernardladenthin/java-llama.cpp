# Script to find and copy DLL/SO/DYLIB files from llama.cpp build directory
# Run as: cmake -P copy_dll_files.cmake
# Requires CMake variables:
#   - CMAKE_SOURCE_DIR: project root
#   - OS_NAME: operating system name (Windows, Linux, Mac)
#   - OS_ARCH: architecture (x86_64, aarch64, x86, etc.)
#   - llama.cpp_BINARY_DIR: llama.cpp build directory

# Get variables from parent CMake
get_filename_component(CMAKE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

# Read OS_NAME and OS_ARCH from file if not set in environment
if(NOT DEFINED OS_NAME)
    find_package(Java REQUIRED)
    find_program(JAVA_EXECUTABLE NAMES java)
    execute_process(
        COMMAND ${JAVA_EXECUTABLE} -cp ${CMAKE_SOURCE_DIR}/target/classes de.kherud.llama.OSInfo --os
        OUTPUT_VARIABLE OS_NAME
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

if(NOT DEFINED OS_ARCH)
    find_package(Java REQUIRED)
    find_program(JAVA_EXECUTABLE NAMES java)
    execute_process(
        COMMAND ${JAVA_EXECUTABLE} -cp ${CMAKE_SOURCE_DIR}/target/classes de.kherud.llama.OSInfo --arch
        OUTPUT_VARIABLE OS_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

# Determine output directory
if(GGML_CUDA)
    set(OUTPUT_DIR "${CMAKE_SOURCE_DIR}/src/main/resources_linux_cuda/de/kherud/llama/${OS_NAME}/${OS_ARCH}")
else()
    set(OUTPUT_DIR "${CMAKE_SOURCE_DIR}/src/main/resources/de/kherud/llama/${OS_NAME}/${OS_ARCH}")
endif()

# Find DLL/SO/DYLIB files in common build locations
set(SEARCH_DIRS
    "${CMAKE_BINARY_DIR}/bin"
    "${CMAKE_BINARY_DIR}/lib"
    "${CMAKE_BINARY_DIR}/Release"
    "${CMAKE_BINARY_DIR}/Debug"
    "${CMAKE_BINARY_DIR}/_deps"
)

message(STATUS "Searching for DLL/SO/DYLIB files in llama.cpp build directory")
message(STATUS "Output directory: ${OUTPUT_DIR}")

# Platform-specific library patterns
if(OS_NAME STREQUAL "Windows")
    set(LIB_PATTERNS "*.dll" "*.lib")
    set(SEARCH_EXCLUDE_PATTERNS ".*\\.exp$")
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
    message(STATUS "Found DLL/SO/DYLIB files:")
    foreach(FILE ${FOUND_FILES})
        message(STATUS "  - ${FILE}")

        # Get filename
        get_filename_component(FILENAME "${FILE}" NAME)
        set(DEST "${OUTPUT_DIR}/${FILENAME}")

        # Copy file
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${FILE}" "${DEST}"
            RESULT_VARIABLE COPY_RESULT
        )

        if(COPY_RESULT EQUAL 0)
            message(STATUS "    ✓ Copied to ${DEST}")
        else()
            message(WARNING "    ✗ Failed to copy ${FILE}")
        endif()
    endforeach()
else()
    message(STATUS "No DLL/SO/DYLIB files found in build directory (this is normal for static builds)")
endif()
