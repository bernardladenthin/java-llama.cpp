// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
//
// Runnable guard for patches/0020 (ggml-alloc: validate the leafs before reusing an allocation).
//
// ggml_gallocr_alloc_graph() hands every tensor the slot of the same position in the graph the
// allocator was reserved with. Before the patch, ggml_gallocr_needs_realloc() validated the nodes
// and their sources but compared only the NUMBER of leafs, so a graph with the same leaf count in
// another order reused the old leaf slots: an input leaf landed on the slot of a weight (no
// allocation, buffer_id -1) and ggml_vbuffer_tensor_alloc() indexed galloc->buffers[-1]. Since
// llama.cpp #30445 that is exactly what an --embedding context does when the server samples on it
// (t_embd + t_logits expanded at reserve, only t_logits at decode); it killed the JVM in
// ErrorHandlingTest on every platform.
//
// The two graphs below have the same two nodes in the same order -- n1 = add(x, x), then
// n2 = mul(w, n1) -- so every node and every node source matches its slot, which the old check did
// verify. Only the leafs move: the first graph expands n1 before n2 and meets the input x first, the
// second expands only n2, whose first source is the weight w. That is the shape of the llama.cpp
// case (t_embd then t_logits at reserve, only t_logits at decode, output.weight first). Reserved
// with the first, the second must still get an allocation for x inside the allocator's buffer and
// compute the right product on it. Without the patch x takes the slot of w (no allocation) and the
// allocation step reads galloc->buffers[-1]; the test dies or fails here, on every platform, without
// a model.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace {

constexpr int64_t N = 16;

struct ctx_deleter {
    void operator()(ggml_context *ctx) const { ggml_free(ctx); }
};
using ctx_ptr = std::unique_ptr<ggml_context, ctx_deleter>;

ctx_ptr new_no_alloc_context(size_t n_tensors, size_t n_graphs) {
    ggml_init_params params = {
        /*.mem_size   =*/ggml_tensor_overhead() * n_tensors + ggml_graph_overhead() * n_graphs,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    return ctx_ptr(ggml_init(params));
}

} // namespace

TEST(GallocrLeafOrder, AGraphWithItsLeafsInAnotherOrderIsAllocatedAgain) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    ASSERT_NE(backend, nullptr);

    // the "weight": a leaf that already has data, so the reserve gives it no slot of its own
    ctx_ptr wctx = new_no_alloc_context(1, 0);
    ggml_tensor *w = ggml_new_tensor_1d(wctx.get(), GGML_TYPE_F32, N);
    ggml_set_name(w, "w");
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx.get(), backend);
    ASSERT_NE(wbuf, nullptr);

    // the "input": a leaf without data, which the allocator has to place
    ctx_ptr gctx = new_no_alloc_context(8, 2);
    ggml_tensor *x = ggml_new_tensor_1d(gctx.get(), GGML_TYPE_F32, N);
    ggml_set_name(x, "x");
    ggml_set_input(x);

    ggml_tensor *n1 = ggml_add(gctx.get(), x, x);
    ggml_tensor *out = ggml_mul(gctx.get(), w, n1);
    ggml_set_output(out);

    ggml_cgraph *reserved = ggml_new_graph(gctx.get());
    ggml_build_forward_expand(reserved, n1);  // leafs: x, w
    ggml_build_forward_expand(reserved, out); // nodes: n1, out
    ggml_cgraph *decoded = ggml_new_graph(gctx.get());
    ggml_build_forward_expand(decoded, out); // leafs: w, x -- nodes: n1, out
    ASSERT_EQ(ggml_graph_n_nodes(reserved), ggml_graph_n_nodes(decoded));

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ASSERT_NE(galloc, nullptr);
    ASSERT_TRUE(ggml_gallocr_reserve(galloc, reserved));
    ASSERT_TRUE(ggml_gallocr_alloc_graph(galloc, decoded));

    ASSERT_NE(x->data, nullptr) << "the input leaf got no allocation";
    ASSERT_NE(x->buffer, nullptr) << "the input leaf is not in the allocator's buffer";
    const auto *base = static_cast<const uint8_t *>(ggml_backend_buffer_get_base(x->buffer));
    const auto *data = static_cast<const uint8_t *>(x->data);
    ASSERT_GE(data, base);
    ASSERT_LE(data + ggml_nbytes(x), base + ggml_backend_buffer_get_size(x->buffer));

    std::vector<float> wv(N), xv(N), ov(N);
    for (int64_t i = 0; i < N; i++) {
        wv[i] = static_cast<float>(i);
        xv[i] = 100.0f;
    }
    ggml_backend_tensor_set(w, wv.data(), 0, ggml_nbytes(w));
    ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
    ASSERT_EQ(ggml_backend_graph_compute(backend, decoded), GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(out, ov.data(), 0, ggml_nbytes(out));
    for (int64_t i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(ov[i], 200.0f * static_cast<float>(i)) << "at " << i;
    }

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(wbuf);
    ggml_backend_free(backend);
}
