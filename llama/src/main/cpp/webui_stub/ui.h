// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

#pragma once

// ui.h — minimal stand-in for the WebUI asset interface that llama.cpp's
// tools/ui (CMake target "llama-ui") normally GENERATES into ui.h / ui.cpp at
// build time via scripts/ui-assets.cmake (before upstream #28445: the embed.cpp host tool).
//
// The upstream HTTP transport (tools/server/server-http.cpp) does
//     #include "ui.h"
// and references llama_ui_get_assets() / llama_ui_find_asset() /
// llama_ui_use_gzip().  We compile server-http.cpp directly into libjllama but do
// NOT ship the Svelte WebUI assets (building them needs npm, or a prebuilt-asset
// download from Hugging Face) — so we provide the exact "empty asset table"
// interface upstream's ui.h.in emits for its n_assets == 0 branch: the struct plus
// the three functions, returning nothing.
//
// LLAMA_UI_HAS_ASSETS is intentionally left UNDEFINED.  Every static-asset-serving
// block in server-http.cpp is guarded by `#if defined(LLAMA_UI_HAS_ASSETS)`, so
// all of them compile out; the single unguarded use — iterating the asset list to
// collect public endpoint paths — simply iterates this empty array.
//
// To actually ship the WebUI later: remove this stub directory from jllama's
// include path, build the real llama-ui target (assets on), and add its
// generated-header directory instead.

#include <array>
#include <cstddef>
#include <string>

struct llama_ui_asset {
    std::string name;
    const unsigned char * data;
    std::size_t size;
    std::string etag;
    std::string type;
};

inline const llama_ui_asset * llama_ui_find_asset(const std::string & /*name*/) {
    return nullptr;
}

inline bool llama_ui_use_gzip() {
    return false;
}

inline const std::array<llama_ui_asset, 0> & llama_ui_get_assets() {
    static const std::array<llama_ui_asset, 0> empty{};
    return empty;
}
