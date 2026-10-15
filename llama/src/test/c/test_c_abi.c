// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
//
// The plain-C consumer of layer A (TODO.md, delivery roadmap, item 5): a C11 program -- no C++, no
// JVM -- linking the shared llama and ggml-base libraries a modular build puts next to libjllama,
// the way a consumer of upstream's own release zip does. It proves three things ctest cannot see
// in the static test build: that llama.h and ggml-backend.h are consumable as C from the shipped
// shared libraries (the symbols are exported, the headers are C), that
// ggml_backend_load_all_from_path() finds the CPU modules in the natives directory it is given,
// and -- when JLLAMA_C_ABI_MODEL names a GGUF -- that a model loads through that API. Registered
// as the ctest `c_abi` of every modular build (llama/CMakeLists.txt), run by build.sh/build.bat
// right after the build, so the four modular CPU jobs run a test again (BUILD_TESTING is refused
// there: jllama_test needs the CPU backend statically).
//
// Usage: jllama_c_abi <natives directory>   (JLLAMA_C_ABI_MODEL=<gguf> loads a model as well)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml-backend.h"
#include "llama.h"

static int fail(const char * what) {
    fprintf(stderr, "c_abi: FAILED: %s\n", what);
    return 1;
}

int main(int argc, char ** argv) {
    if (argc < 2 || argv[1][0] == '\0') {
        return fail("usage: jllama_c_abi <natives directory>");
    }
    const char * dir = argv[1];
    printf("c_abi: loading the backend modules from %s\n", dir);
    ggml_backend_load_all_from_path(dir);
    llama_backend_init();

    const size_t devices = ggml_backend_dev_count();
    const size_t backends = ggml_backend_reg_count();
    printf("c_abi: %zu backend(s) registered, %zu device(s)\n", backends, devices);
    for (size_t i = 0; i < backends; i++) {
        printf("c_abi:   backend %s\n", ggml_backend_reg_name(ggml_backend_reg_get(i)));
    }
    if (devices == 0) {
        return fail("no device: the CPU modules next to the library did not load");
    }
    int cpu = 0;
    for (size_t i = 0; i < devices; i++) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            cpu = 1;
        }
    }
    if (!cpu) {
        return fail("no CPU device among the registered devices");
    }
    printf("c_abi: system info: %s\n", llama_print_system_info());

    const char * model_path = getenv("JLLAMA_C_ABI_MODEL");
    if (model_path != NULL && model_path[0] != '\0') {
        struct llama_model_params params = llama_model_default_params();
        params.n_gpu_layers = 0;
        struct llama_model * model = llama_model_load_from_file(model_path, params);
        if (model == NULL) {
            return fail("llama_model_load_from_file returned NULL");
        }
        char description[256];
        llama_model_desc(model, description, sizeof description);
        printf("c_abi: loaded %s: %s, %lld parameters\n", model_path, description,
               (long long) llama_model_n_params(model));
        llama_model_free(model);
    } else {
        printf("c_abi: JLLAMA_C_ABI_MODEL not set, no model loaded\n");
    }
    llama_backend_free();
    printf("c_abi: OK\n");
    return 0;
}
