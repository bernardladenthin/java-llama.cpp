// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
//
// The one place that resolves a hand-built common_params' CPU fields.
//
// Every common_params this project assembles by hand -- the TTS backbone (tts_params.hpp) and the
// trainer (train_engine.cpp) -- has to go through here before it reaches common_init_from_params.
// It is a header for the same reason tts_params.hpp is one: jllama_test does not compile
// tts_engine.cpp or train_engine.cpp (both need the mtmd/llama runtime), and a header lets the
// tests exercise the real resolver those TUs call rather than a copy that could drift from it.

#ifndef JLLAMA_CPU_PARAMS_HPP
#define JLLAMA_CPU_PARAMS_HPP

#include "common.h"

namespace jllama {

// ---------------------------------------------------------------------------
// Resolves params.cpuparams / params.cpuparams_batch the way common/arg.cpp does.
//
// This is load-bearing, not tidiness. common/arg.cpp is upstream's ONLY caller of
// postprocess_cpu_params, so a common_params assembled by hand never gets its CPU fields
// resolved: common_cpu_params::n_threads keeps its -1 default, and common_init_from_params does
// not fix it up. common_threadpools::init then finds tpp and tpp_batch mismatched, builds a
// separate batch pool, and ggml_threadpool_new computes
//   workers_size = sizeof(struct ggml_compute_state) * tpp->n_threads
// with n_threads == -1 -- a huge size_t. ggml_aligned_malloc returns NULL and the very next line
// memsets it unchecked, so the process dies on memset(NULL, 0, huge): SIGSEGV at address 0,
// reported as __bzero on macOS and as a bare libc frame on Linux. Because the abort bypasses the
// JVM error handler it writes no hs_err_pid log, which is what made it expensive to find.
//
// The role_model argument on the second call is what makes the batch pool INHERIT the main pool's
// count instead of resolving independently; passing nullptr for both would let them diverge.
// This mirrors common/arg.cpp's own pair exactly.
// ---------------------------------------------------------------------------
inline void resolve_cpu_params(common_params &params) {
    postprocess_cpu_params(params.cpuparams, nullptr);
    postprocess_cpu_params(params.cpuparams_batch, &params.cpuparams);
}

} // namespace jllama

#endif // JLLAMA_CPU_PARAMS_HPP
