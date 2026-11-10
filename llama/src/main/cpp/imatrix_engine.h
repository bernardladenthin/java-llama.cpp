// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

// Computing an importance matrix: what upstream's llama-imatrix tool does, as a library function.
// The model reads a calibration text chunk by chunk, and for every weight matrix of its layers the
// squared activations that enter it are summed per column. The quantizer (jllama::engine::quantize
// with an imatrix path) weighs its rounding error with those sums; the IQ1, IQ2, IQ3_XS, IQ3_XXS and
// Q2_K_S file types cannot be written without them.
namespace jllama::imatrix {

struct options {
    int chunk_size = 512; // tokens per chunk, the context the model is run with; at least 16
    int max_chunks = 0;   // 0: every whole chunk of the text
    int n_threads = 0;    // 0: llama.cpp's default
    int n_gpu_layers = 0; // layers to offload
};

// Writes the importance matrix of the model at model_path over the text in text_path to output_path
// (GGUF, the format llama-imatrix writes and llama-quantize reads) and returns the number of chunks
// it was computed over. Throws std::runtime_error naming what failed.
int compute(const std::string &model_path, const std::string &text_path, const std::string &output_path,
            const options &opts);

} // namespace jllama::imatrix
