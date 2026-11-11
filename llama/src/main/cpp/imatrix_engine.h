// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

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
    // Called on the calling thread after every chunk with the fraction done; false stops the run,
    // which then throws and writes nothing. Null: no report.
    bool (*progress)(float fraction, void *user_data) = nullptr;
    void *progress_user_data = nullptr;
};

// Writes the importance matrix of the model at model_path over the text in text_path to output_path
// (GGUF, the format llama-imatrix writes and llama-quantize reads) and returns the number of chunks
// it was computed over. Throws std::runtime_error naming what failed.
int compute(const std::string &model_path, const std::string &text_path, const std::string &output_path,
            const options &opts);

// Adds up importance matrices of one model, computed over different texts, into the one at
// output_path (GGUF), as llama-imatrix does with several --in-file and no text. The inputs are GGUF
// or the legacy binary format (which is thereby converted); they must have been computed with the
// same chunk size. With legacy_format the result is written in the legacy binary format instead,
// for a tool that reads only that. Returns the number of chunks the result stands for. Throws
// std::runtime_error naming the file that could not be read or does not fit.
int merge(const std::vector<std::string> &input_paths, const std::string &output_path, bool legacy_format);

} // namespace jllama::imatrix
