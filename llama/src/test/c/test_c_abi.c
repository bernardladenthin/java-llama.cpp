// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
//
// The plain-C consumer of both layers (TODO.md, delivery roadmap, items 5 and 7): a C11 program --
// no C++, no JVM -- linking the shared llama and ggml-base libraries a modular build puts next to
// libjllama, the way a consumer of upstream's own release zip does, and libjllama itself for its
// C API (jllama_api.h). It proves what ctest cannot see in the static test build: that llama.h and
// ggml-backend.h are consumable as C from the shipped shared libraries (the symbols are exported,
// the headers are C), that jllama_api.h compiles as C and its jllama_* symbols are exported from
// the shipped libjllama, that ggml_backend_load_all_from_path() finds the CPU modules in the
// natives directory it is given, and -- when JLLAMA_C_ABI_MODEL names a GGUF -- that a model loads
// through upstream's API. Registered
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
#include "jllama_api.h"

static int fail(const char * what) {
    fprintf(stderr, "c_abi: FAILED: %s\n", what);
    return 1;
}

/* The `llama.cpp=b<nnnn>` line of the directory's jllama-build.txt, the pin the loader checks a
 * module jar against; "" when the file or the line is missing. */
static const char * pinned_tag(const char * dir) {
    static char tag[64];
    char path[4096];
    char line[512];
    tag[0] = '\0';
    snprintf(path, sizeof(path), "%s/jllama-build.txt", dir);
    FILE * f = fopen(path, "r");
    if (f == NULL) {
        return tag;
    }
    while (fgets(line, sizeof(line), f) != NULL) {
        if (strncmp(line, "llama.cpp=", 10) == 0) {
            size_t n = strcspn(line + 10, "\r\n");
            if (n >= sizeof(tag)) {
                n = sizeof(tag) - 1;
            }
            memcpy(tag, line + 10, n);
            tag[n] = '\0';
            break;
        }
    }
    fclose(f);
    return tag;
}

int main(int argc, char ** argv) {
    if (argc < 2 || argv[1][0] == '\0') {
        return fail("usage: jllama_c_abi <natives directory>");
    }
    /* Layer B: the C API of libjllama, reached through the shipped library's exports. */
    const char * api_version = jllama_api_version();
    const char * build_info = jllama_build_info();
    if (api_version == NULL || api_version[0] == '\0' || build_info == NULL || build_info[0] != 'b') {
        return fail("jllama_api_version() / jllama_build_info() returned nothing usable");
    }
    const char * dir = argv[1];
    /* build_info is llama.cpp's own `b<nnnn>-<commit>`; its number must be the tag the directory's
     * jllama-build.txt pins (what LlamaLoader holds every module jar to). */
    const char * tag = pinned_tag(dir);
    const size_t tag_len = strlen(tag);
    if (tag_len == 0 || strncmp(build_info, tag, tag_len) != 0 || build_info[tag_len] != '-') {
        fprintf(stderr, "c_abi: jllama_build_info() = '%s', jllama-build.txt pins '%s'\n", build_info, tag);
        return fail("jllama_build_info() does not name the pinned llama.cpp tag of its directory");
    }
    printf("c_abi: jllama C API %s over llama.cpp %s\n", api_version, build_info);
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
