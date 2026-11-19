// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

#ifndef JLLAMA_BACKEND_MODULES_HPP
#define JLLAMA_BACKEND_MODULES_HPP

namespace jllama {

// Loads the ggml backend modules that lie next to this library (a JLLAMA_BACKEND_DL build: the CPU
// variants, ggml-rpc and every GPU module jar the loader extracted there), once per process; a no-op
// in a static build. JNI_OnLoad calls it, and so does every C API function that needs a backend before
// it touches one: a C or FFM host has no JNI_OnLoad, and ggml's own ggml_backend_load_all() looks next
// to the executable and in the working directory, not next to libjllama -- a C program linked against
// the shipped library found no backend at all ("no backends are loaded").
void load_backend_modules();

} // namespace jllama

#endif // JLLAMA_BACKEND_MODULES_HPP
