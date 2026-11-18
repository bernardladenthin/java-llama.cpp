// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

// Writes the model of the c_abi test's model-backed half: the C++ suite's tiny random llama GGUF
// (tiny_llama_gguf.hpp, the GPT-2 byte tokenizer), through the public gguf API of the shipped
// ggml-base -- so a modular build job tests a model load without downloading one. Registered as the
// ctest fixture `c_abi_model` (llama/CMakeLists.txt).
//
// Usage: jllama_c_abi_model <output.gguf>

#include "tiny_llama_gguf.hpp"

#include <cstdio>
#include <filesystem>
#include <string>
#include <system_error>

int main(int argc, char **argv) {
    if (argc < 2 || argv[1][0] == '\0') {
        std::fprintf(stderr, "usage: jllama_c_abi_model <output.gguf>\n");
        return 1;
    }
    jllama_test::tiny_model_spec spec;
    spec.prefix = "jllama-c-abi-model";
    const std::string written = jllama_test::write_tiny_llama_gguf(spec);
    if (written.empty()) {
        std::fprintf(stderr, "jllama_c_abi_model: the model could not be written\n");
        return 1;
    }
    std::error_code ec;
    std::filesystem::copy_file(written, argv[1], std::filesystem::copy_options::overwrite_existing, ec);
    std::filesystem::remove(written);
    if (ec) {
        std::fprintf(stderr, "jllama_c_abi_model: %s: %s\n", argv[1], ec.message().c_str());
        return 1;
    }
    std::printf("jllama_c_abi_model: wrote %s\n", argv[1]);
    return 0;
}
