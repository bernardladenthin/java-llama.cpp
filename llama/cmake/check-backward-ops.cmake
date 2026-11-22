# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# The ops patches/0044 lets a fine-tuning run differentiate must be the ops ggml_compute_backward
# (ggml/src/ggml.c) has a case for -- checked at configure time, on every platform, in every build.
#
# Why. ggml_build_backward_expand ends the process (GGML_ABORT) on a graph op it has no backward pass
# for, and llama_opt_epoch has no way to report anything, so fine-tuning a mixture of experts, a
# recurrent model, Gemma 4 or Cohere2 ended the JVM that ran it. patches/0044 walks the forward graph
# with the backward pass's rules first and refuses such a model by its op; its op lists
# (llama_opt_backward_op_supported() in src/llama-context.cpp, between the "jllama-backward-*-ops-begin"
# and "-end" markers) repeat the case labels of ggml_compute_backward. Two lists of the same thing drift
# apart one bump at a time: an op upstream gives a backward pass (b12635 added the unary SIGMOID's)
# would be refused here although ggml could train it, and a case upstream removes would let the abort
# through again. Neither shows in a test unless some test happens to train a model with that op.
#
# The comparison is of case labels, with what the patch deliberately answers differently named here:
# WIN_PART and WIN_UNPART share UNARY's case upstream, whose ggml_get_unary_op asserts that the op is
# UNARY, so the patch refuses them; UNARY and GLU are checked by their sub-op lists; COUNT is no op.
#
# A failure means upstream changed ggml_compute_backward: read its switch at the new tag and change the
# lists in the patch to match it (and the patch's tests, if an op of the generator's architectures moved).

set(JLLAMA_BACKWARD_OPS_HANDLED_ELSEWHERE GGML_OP_WIN_PART GGML_OP_WIN_UNPART GGML_OP_UNARY GGML_OP_GLU GGML_OP_COUNT)

# the labels "case <prefix>X:" in text, sorted, as a list of the X with the prefix
function(_jllama_case_labels text prefix out)
    string(REGEX MATCHALL "case ${prefix}[A-Z0-9_]+:" found "${text}")
    set(labels "")
    foreach(label IN LISTS found)
        string(REGEX REPLACE "^case (${prefix}[A-Z0-9_]+):$" "\\1" label "${label}")
        list(APPEND labels "${label}")
    endforeach()
    list(REMOVE_DUPLICATES labels)
    list(SORT labels)
    set(${out} "${labels}" PARENT_SCOPE)
endfunction()

# the text of src between "// <marker>-begin" and "// <marker>-end"
function(_jllama_marked_region src marker shown out)
    string(FIND "${src}" "// ${marker}-begin" begin)
    string(FIND "${src}" "// ${marker}-end" end)
    if(begin EQUAL -1 OR end EQUAL -1 OR end LESS begin)
        message(FATAL_ERROR
            "backward ops: ${shown} has no \"// ${marker}-begin\" ... \"-end\" region. patches/0044 puts "
            "its op lists there; was the patch dropped or changed? See llama/cmake/check-backward-ops.cmake.")
    endif()
    math(EXPR length "${end} - ${begin}")
    string(SUBSTRING "${src}" ${begin} ${length} region)
    set(${out} "${region}" PARENT_SCOPE)
endfunction()

function(jllama_check_backward_ops root)
    set(ggml_c "${root}/ggml/src/ggml.c")
    set(context "${root}/src/llama-context.cpp")
    file(READ "${ggml_c}" ggml_text)
    string(FIND "${ggml_text}" "static void ggml_compute_backward(" start)
    if(start EQUAL -1)
        message(FATAL_ERROR "backward ops: ggml_compute_backward is not in ${ggml_c}; read how upstream builds "
                            "the backward pass now. See llama/cmake/check-backward-ops.cmake.")
    endif()
    string(SUBSTRING "${ggml_text}" ${start} -1 rest)
    string(FIND "${rest}" "\n}\n" end)
    string(SUBSTRING "${rest}" 0 ${end} backward)

    _jllama_case_labels("${backward}" "GGML_OP_" ggml_ops)
    list(REMOVE_ITEM ggml_ops ${JLLAMA_BACKWARD_OPS_HANDLED_ELSEWHERE})
    _jllama_case_labels("${backward}" "GGML_UNARY_OP_" ggml_unary)
    _jllama_case_labels("${backward}" "GGML_GLU_OP_" ggml_glu)
    list(LENGTH ggml_ops n_ops)
    if(n_ops LESS 20)
        message(FATAL_ERROR "backward ops: only ${n_ops} op cases read from ggml_compute_backward in ${ggml_c}; "
                            "the reader no longer matches its switch. See llama/cmake/check-backward-ops.cmake.")
    endif()

    file(READ "${context}" context_text)
    _jllama_marked_region("${context_text}" "jllama-backward-ops" "src/llama-context.cpp" region)
    _jllama_case_labels("${region}" "GGML_OP_" patch_ops)
    _jllama_marked_region("${context_text}" "jllama-backward-unary-ops" "src/llama-context.cpp" region)
    _jllama_case_labels("${region}" "GGML_UNARY_OP_" patch_unary)
    _jllama_marked_region("${context_text}" "jllama-backward-glu-ops" "src/llama-context.cpp" region)
    _jllama_case_labels("${region}" "GGML_GLU_OP_" patch_glu)

    set(problems "")
    foreach(kind ops unary glu)
        set(refused ${ggml_${kind}})
        if(patch_${kind})
            list(REMOVE_ITEM refused ${patch_${kind}})
        endif()
        set(accepted ${patch_${kind}})
        if(ggml_${kind})
            list(REMOVE_ITEM accepted ${ggml_${kind}})
        endif()
        foreach(op IN LISTS refused)
            list(APPEND problems "${op}: ggml_compute_backward has a case for it, patches/0044 refuses it")
        endforeach()
        foreach(op IN LISTS accepted)
            list(APPEND problems "${op}: patches/0044 accepts it, ggml_compute_backward has no case for it")
        endforeach()
    endforeach()
    if(problems)
        string(REPLACE ";" "\n  - " shown "${problems}")
        message(FATAL_ERROR
            "backward ops: the op lists of patches/0044 (llama_opt_backward_op_supported in "
            "src/llama-context.cpp) differ from ggml_compute_backward's switch in ggml/src/ggml.c:\n  - ${shown}\n"
            "A refused op that ggml can differentiate refuses a trainable model; an accepted op that it cannot "
            "lets ggml_build_backward_expand end the process again. Change the lists in the patch to match the "
            "switch. See llama/cmake/check-backward-ops.cmake.")
    endif()
    list(LENGTH patch_unary n_unary)
    list(LENGTH patch_glu n_glu)
    message(STATUS "backward ops: patches/0044 accepts the ${n_ops} ops, ${n_unary} unary and ${n_glu} glu ops "
                   "ggml_compute_backward has a case for")
endfunction()
