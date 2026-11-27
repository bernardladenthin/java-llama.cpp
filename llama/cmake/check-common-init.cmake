# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# common_init() (common/common.cpp) does exactly what jllama_engine.cpp's init_common_for_load() repeats on
# every load -- checked at configure time, on every platform, in every build.
#
# Why: the engine calls common_init() once per process and afterwards sets only common_log's prefix and
# timestamps itself, because common_init() also points the process-wide llama logger at
# common_log_default_callback, which took the logger that keeps a load's reason away from every load on
# another thread (a failed load named no reason in 769 of 1000 tries, measured 2026-11-27). A statement
# upstream adds to common_init() would run once instead of on every load, unseen; this fails the configure
# instead, so the engine is changed with it.

function(jllama_check_common_init root)
    set(source "${root}/common/common.cpp")
    if(NOT EXISTS "${source}")
        message(STATUS "common_init: ${source} not found, nothing to check")
        return()
    endif()
    file(READ "${source}" text)
    string(FIND "${text}" "\nvoid common_init() {\n" start)
    if(start EQUAL -1)
        message(FATAL_ERROR "common_init: no 'void common_init() {' in common/common.cpp. "
            "Read what replaced it and update init_common_for_load() in src/main/cpp/jllama_engine.cpp and "
            "llama/cmake/check-common-init.cmake.")
    endif()
    string(SUBSTRING "${text}" ${start} -1 rest)
    string(FIND "${rest}" "\n}\n" end)
    string(SUBSTRING "${rest}" 0 ${end} body)
    # compare without whitespace, so that a reformat alone does not fail the configure
    string(REGEX REPLACE "[ \t\r\n]+" "" body "${body}")
    # one bracket argument: its semicolons would split a quoted list into elements
    set(expected [[voidcommon_init(){#ifdefined(_WIN32)SetConsoleOutputCP(CP_UTF8);SetConsoleCP(CP_UTF8);#endifcommon_log_set_prefix(common_log_main(),true);common_log_set_timestamps(common_log_main(),true);llama_log_set(common_log_default_callback,NULL);]])
    if(NOT body STREQUAL expected)
        message(FATAL_ERROR
            "common_init() in common/common.cpp is not what jllama_engine.cpp's init_common_for_load() "
            "repeats on every load (it calls common_init() once per process, then sets common_log's prefix "
            "and timestamps itself and keeps the llama logger that remembers a load's reason).\n"
            "  found:    ${body}\n  expected: ${expected}\n"
            "Repeat what upstream added in init_common_for_load(), then update the expected body here. "
            "See llama/cmake/check-common-init.cmake.")
    endif()
    message(STATUS "common_init: its body is what init_common_for_load() repeats")
endfunction()
