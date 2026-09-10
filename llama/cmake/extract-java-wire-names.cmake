# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT

# Generates a C++ header listing the wire names one Java enum registry declares, together with
# the contract each one states it satisfies.
#
# WHY THIS EXISTS
# ---------------
# Three surfaces leave this library as names on a wire, and each has a receiver that can be
# asked what it accepts:
#
#   * CLI options (ModelFlag + ModelOption) -> llama.cpp's
#     common_params_parser_init(params, LLAMA_EXAMPLE_SERVER).options
#   * request keys (RequestField)           -> server_schema::make_llama_cmpl_schema(...)
#   * trainer keys (TrainingField)          -> jllama_train::config_keys() in train_engine.cpp
#
# Both failure modes are silent on the Java side. An unregistered CLI option is a hard parse
# error, so the model simply never loads; an unknown request key is discarded without a word, so
# the parameter simply never takes effect. Either way a Java test that asserts the string mapping
# ("does the map contain top_k") passes forever while the name is dead. The C++ tests close that
# by feeding this generated list to the real receiver, and they run on every platform.
#
# EXTRACTION
# ----------
# Only enum constant declarations are matched -- `NAME("wire-name")` or
# `NAME("wire-name", XxxContract.KIND)` -- so prose, javadoc and helper code cannot contribute a
# name by accident. That precision is why the registries were made enums in the first place: the
# previous version of this script scanned any string literal in code position across a 1900-line
# builder and needed a comment-stripping heuristic to do it.
#
# THE EXEMPTION HOLE, AND WHY THERE IS A SECOND SCAN
# --------------------------------------------------
# A name that declares a contract the receiver above cannot answer for -- OAI_LAYER, consumed by
# oaicompat_*_params_parse and the task layer before the schema ever sees the body -- was checked
# only for *absence* from the schema. Absence is satisfied just as well by a name nothing reads at
# all, so the exemption was a hole exactly the size of the problem the registry was built to close:
# `chat_template` sat in it, written by a public builder method, read by nobody, discarded silently.
#
# There is no callable table to ask "which keys does this parser read", so the oracle here is a
# *reader-shaped* sweep of the receiver's own source -- `json_value(x, "k", ...)`, `.contains("k")`,
# `.at("k")` -- rather than a bare token grep. The shape is what makes it useful: `chat_template`
# does occur as a literal upstream, in the `/props` payload the server *emits*, and a token grep
# would have called it live. This is weaker evidence than driving the real receiver, so it proves
# only "something reads this key from a body"; the C++ test says so where it asserts on it.
#
# Inputs : JAVA_SOURCES     - the registry .java files to scan
#          ARRAY_PREFIX     - C identifier prefix; emits <P>_NAMES[], <P>_CONTRACTS[],
#                             <P>_READERS[], <P>_COUNT
#          DEFAULT_CONTRACT - contract for a constant that does not name one
#          MIN_COUNT        - floor below which extraction is treated as broken
#          OUTPUT_HEADER    - path of the header to write
#          READER_CONTRACT  - optional: contract whose names get the reader sweep
#          READER_SOURCES   - optional: receiver sources to sweep for those names

function(jllama_extract_java_wire_names)
    cmake_parse_arguments(ARG ""
        "ARRAY_PREFIX;DEFAULT_CONTRACT;MIN_COUNT;OUTPUT_HEADER;READER_CONTRACT"
        "JAVA_SOURCES;READER_SOURCES" ${ARGN})

    foreach(_required ARRAY_PREFIX DEFAULT_CONTRACT MIN_COUNT OUTPUT_HEADER JAVA_SOURCES)
        if(NOT ARG_${_required})
            message(FATAL_ERROR "jllama_extract_java_wire_names: ${_required} is required")
        endif()
    endforeach()

    set(_names "")
    set(_contracts "")
    foreach(_src IN LISTS ARG_JAVA_SOURCES)
        if(NOT EXISTS "${_src}")
            message(FATAL_ERROR "jllama_extract_java_wire_names: missing source ${_src}")
        endif()
        file(STRINGS "${_src}" _lines)
        foreach(_line IN LISTS _lines)
            string(STRIP "${_line}" _trimmed)
            # Enum constant declarations only. A javadoc line mentioning {@code --mlock} cannot
            # match this, so no comment stripping is needed to keep prose out.
            if(NOT _trimmed MATCHES "^[A-Z][A-Z0-9_]*\\(\"([^\"]+)\"(.*)\\)[,;]$")
                continue()
            endif()
            set(_name "${CMAKE_MATCH_1}")
            set(_rest "${CMAKE_MATCH_2}")
            set(_contract "${ARG_DEFAULT_CONTRACT}")
            if(_rest MATCHES "Contract\\.([A-Z][A-Z0-9_]*)")
                set(_contract "${CMAKE_MATCH_1}")
            endif()
            list(APPEND _names "${_name}")
            list(APPEND _contracts "${_name}=${_contract}")
        endforeach()
    endforeach()

    list(LENGTH _names _count)
    list(REMOVE_DUPLICATES _names)
    list(LENGTH _names _unique_count)
    if(NOT _count EQUAL _unique_count)
        message(FATAL_ERROR
            "jllama_extract_java_wire_names: ${ARG_ARRAY_PREFIX} declares a wire name twice -- "
            "two builder methods would write the same key and the last call would win")
    endif()
    list(SORT _names)

    # A silently empty list would make the contract test vacuously pass -- the same
    # "nothing to scan reported as a clean pass" trap verify-bytecode-version.sh exits 2 for.
    if(_count LESS ${ARG_MIN_COUNT})
        message(FATAL_ERROR
            "jllama_extract_java_wire_names: only ${_count} names extracted for "
            "${ARG_ARRAY_PREFIX} from ${ARG_JAVA_SOURCES} -- the extractor is broken or the "
            "sources moved")
    endif()

    # Reader sweep. Concatenated once, then matched per name -- the receiver sources are read
    # here and nowhere else, so a rename upstream shows up as an empty corpus, not as silence.
    set(_reader_corpus "")
    if(ARG_READER_CONTRACT)
        if(NOT ARG_READER_SOURCES)
            message(FATAL_ERROR
                "jllama_extract_java_wire_names: READER_CONTRACT without READER_SOURCES")
        endif()
        foreach(_src IN LISTS ARG_READER_SOURCES)
            if(NOT EXISTS "${_src}")
                continue()
            endif()
            file(READ "${_src}" _content)
            string(APPEND _reader_corpus "${_content}")
            set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_src}")
        endforeach()
        string(LENGTH "${_reader_corpus}" _corpus_length)
        if(_corpus_length EQUAL 0)
            message(FATAL_ERROR
                "jllama_extract_java_wire_names: ${ARG_ARRAY_PREFIX} reader sweep read nothing "
                "from READER_SOURCES -- the receiver sources moved, and every scanned name would "
                "otherwise report as unread")
        endif()
    endif()

    set(_name_body "")
    set(_contract_body "")
    set(_reader_body "")
    set(_swept 0)
    foreach(_name IN LISTS _names)
        string(APPEND _name_body "    \"${_name}\",\n")
        set(_this_contract "")
        foreach(_pair IN LISTS _contracts)
            if(_pair MATCHES "^${_name}=(.*)$")
                set(_this_contract "${CMAKE_MATCH_1}")
                string(APPEND _contract_body "    \"${_this_contract}\",\n")
                break()
            endif()
        endforeach()
        # -1 means "not swept", which is not the same as "swept and found nothing" (0).
        set(_readers -1)
        if(ARG_READER_CONTRACT AND _this_contract STREQUAL "${ARG_READER_CONTRACT}")
            string(REGEX MATCHALL
                "json_value\\([A-Za-z_.]+, *\"${_name}\"|\\.contains\\(\"${_name}\"\\)|\\.at\\(\"${_name}\"\\)"
                _hits "${_reader_corpus}")
            list(LENGTH _hits _readers)
            math(EXPR _swept "${_swept} + 1")
        endif()
        string(APPEND _reader_body "    ${_readers},\n")
    endforeach()

    set(_header "// Generated by cmake/extract-java-wire-names.cmake -- DO NOT EDIT.\n")
    string(APPEND _header "// Source of truth: the Java registry files listed in llama/CMakeLists.txt.\n")
    string(APPEND _header "#pragma once\n\n")
    string(APPEND _header "static const char * const ${ARG_ARRAY_PREFIX}_NAMES[] = {\n${_name_body}};\n\n")
    string(APPEND _header "static const char * const ${ARG_ARRAY_PREFIX}_CONTRACTS[] = {\n${_contract_body}};\n\n")
    string(APPEND _header "static const int ${ARG_ARRAY_PREFIX}_READERS[] = {\n${_reader_body}};\n\n")
    string(APPEND _header "static const int ${ARG_ARRAY_PREFIX}_COUNT = ${_count};\n")

    # Only rewrite when the content actually changed, so an unrelated re-configure does not
    # touch the header and force a needless rebuild of the test.
    set(_existing "")
    if(EXISTS "${ARG_OUTPUT_HEADER}")
        file(READ "${ARG_OUTPUT_HEADER}" _existing)
    endif()
    if(NOT _existing STREQUAL _header)
        file(WRITE "${ARG_OUTPUT_HEADER}" "${_header}")
    endif()

    # Re-run configure (and therefore this extractor) whenever a scanned Java file changes.
    foreach(_src IN LISTS ARG_JAVA_SOURCES)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_src}")
    endforeach()

    set(_swept_note "")
    if(ARG_READER_CONTRACT)
        set(_swept_note " (${_swept} swept for ${ARG_READER_CONTRACT} readers)")
    endif()
    message(STATUS
        "jllama: extracted ${_count} ${ARG_ARRAY_PREFIX} names${_swept_note} -> ${ARG_OUTPUT_HEADER}")
endfunction()
