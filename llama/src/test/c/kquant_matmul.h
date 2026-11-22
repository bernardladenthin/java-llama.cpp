// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
//
// MUL_MAT over K-quant weights (Q2_K to Q6_K) through the CPU backend ggml loaded -- in a modular
// build the CPU variant module ggml_backend_load_best() chose for the running CPU -- held to a
// reference computed here, from ggml-base's dequantization of the weights and the activations
// quantized to Q8_K the way ggml's own reference quantizer does (quantize_row_q8_K_ref, which
// ggml-base does not export: its type traits carry no conversion for Q8_K). The CPU's K-quant dot
// products work on Q8_K activations, so the two differ only by the order of the float sums and by
// a rounding a SIMD quantizer may take differently from the reference.
//
// Why here and not in jllama_test: since llama.cpp b12567 (#32600) the s390x CPU modules built with
// VXE (z15, z16) take these products through llamafile's tinyBLAS_QK_VXE kernel, and the only place
// CI runs such a module is the c_abi test of build-linux-s390x, under qemu-s390x (its CPU model has
// VXE2, so ggml loads the z15 module). The static s390x test build is scalar and never reaches the
// kernel. On the other platforms the check covers whatever path their CPU module takes.
//
// Header-only and plain C11, like test_c_abi.c; no VLAs (MSVC has none).

#ifndef JLLAMA_KQUANT_MATMUL_H
#define JLLAMA_KQUANT_MATMUL_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#define JLLAMA_KQ_BLOCK 256 /* QK_K */

/* A deterministic value in [-1, 1). */
static float jllama_kq_random(uint32_t * state) {
    *state = *state * 1664525u + 1013904223u;
    return (float) ((*state >> 8) & 0xffffff) / 8388608.0f - 1.0f;
}

/* quantize_row_q8_K_ref followed by the dequantization d * q: what a K-quant dot product sees of an
 * activation row of k values. */
static void jllama_kq_through_q8_k(const float * x, float * out, int64_t k) {
    for (int64_t b = 0; b < k / JLLAMA_KQ_BLOCK; b++) {
        const float * xb = x + b * JLLAMA_KQ_BLOCK;
        float * ob = out + b * JLLAMA_KQ_BLOCK;
        float max = 0.0f;
        float amax = 0.0f;
        for (int j = 0; j < JLLAMA_KQ_BLOCK; j++) {
            const float ax = fabsf(xb[j]);
            if (ax > amax) {
                amax = ax;
                max = xb[j];
            }
        }
        if (amax == 0.0f) {
            memset(ob, 0, sizeof(float) * JLLAMA_KQ_BLOCK);
            continue;
        }
        const float iscale = -127.0f / max;
        const float d = 1.0f / iscale;
        for (int j = 0; j < JLLAMA_KQ_BLOCK; j++) {
            long q = lrintf(iscale * xb[j]);
            if (q > 127) {
                q = 127;
            }
            ob[j] = d * (float) q;
        }
    }
}

/* One type: an m x k weight matrix of `type` times n activation columns of k values, on `backend`.
 * Returns 0 when every product is within tolerance of the reference, else 1 (after printing it). */
static int jllama_kq_check_type(ggml_backend_t backend, enum ggml_type type, int64_t m, int64_t n, int64_t k,
                                uint32_t seed) {
    const struct ggml_init_params ip = {ggml_tensor_overhead() * 8 + ggml_graph_overhead(), NULL, true};
    struct ggml_context * ctx = ggml_init(ip);
    if (ctx == NULL) {
        fprintf(stderr, "c_abi: kquant: ggml_init failed\n");
        return 1;
    }
    struct ggml_tensor * w = ggml_new_tensor_2d(ctx, type, k, m);
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    struct ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == NULL) {
        fprintf(stderr, "c_abi: kquant: no buffer for %s\n", ggml_type_name(type));
        ggml_free(ctx);
        return 1;
    }

    uint32_t state = seed;
    float * wf = (float *) malloc(sizeof(float) * (size_t) (m * k));
    float * xf = (float *) malloc(sizeof(float) * (size_t) (n * k));
    float * xq = (float *) malloc(sizeof(float) * (size_t) (n * k));
    float * yf = (float *) malloc(sizeof(float) * (size_t) (m * n));
    const size_t wbytes = ggml_row_size(type, k) * (size_t) m;
    void * wq = malloc(wbytes);
    int failed = 0;
    if (wf == NULL || xf == NULL || xq == NULL || yf == NULL || wq == NULL) {
        fprintf(stderr, "c_abi: kquant: out of memory\n");
        failed = 1;
    } else {
        /* Every row gets its own magnitude, so the per-block scales differ between the rows. */
        for (int64_t i = 0; i < m; i++) {
            const float row_scale = 0.25f + 0.5f * (float) (i % 5);
            for (int64_t j = 0; j < k; j++) {
                wf[i * k + j] = row_scale * jllama_kq_random(&state);
            }
        }
        for (int64_t i = 0; i < n * k; i++) {
            xf[i] = jllama_kq_random(&state);
        }
        ggml_quantize_chunk(type, wf, wq, 0, m, k, NULL);
        ggml_backend_tensor_set(w, wq, 0, wbytes);
        ggml_backend_tensor_set(x, xf, 0, sizeof(float) * (size_t) (n * k));
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "c_abi: kquant: the graph did not compute for %s\n", ggml_type_name(type));
            failed = 1;
        } else {
            ggml_backend_tensor_get(y, yf, 0, sizeof(float) * (size_t) (m * n));
            /* The reference: the weights as ggml-base dequantizes them, the activations through Q8_K. */
            ggml_get_type_traits(type)->to_float(wq, wf, m * k);
            for (int64_t c = 0; c < n; c++) {
                jllama_kq_through_q8_k(xf + c * k, xq + c * k, k);
            }
            double worst = 0.0;
            for (int64_t c = 0; c < n && !failed; c++) {
                for (int64_t r = 0; r < m; r++) {
                    double sum = 0.0;
                    double size = 0.0;
                    for (int64_t j = 0; j < k; j++) {
                        const double t = (double) wf[r * k + j] * (double) xq[c * k + j];
                        sum += t;
                        size += fabs(t);
                    }
                    const double got = (double) yf[c * m + r];
                    const double err = fabs(got - sum) / (size > 0.0 ? size : 1.0);
                    if (err > worst) {
                        worst = err;
                    }
                    if (!(err <= 1e-6)) {
                        fprintf(stderr,
                                "c_abi: kquant: %s row %lld column %lld: %.6f, the reference %.6f (relative %.3g)\n",
                                ggml_type_name(type), (long long) r, (long long) c, got, sum, err);
                        failed = 1;
                        break;
                    }
                }
            }
            if (!failed) {
                printf("c_abi: kquant: %s %lldx%lld times %lld columns within %.2g of the reference\n",
                       ggml_type_name(type), (long long) m, (long long) k, (long long) n, worst);
            }
        }
    }
    free(wq);
    free(yf);
    free(xq);
    free(xf);
    free(wf);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return failed;
}

/* Every K-quant type through the CPU backend, on three threads, for column counts that take each of
 * tinyBLAS_QK_VXE's tile widths (4, 2, 1) and the one-column path ggml keeps for itself. Returns 0 on
 * success. */
static int jllama_check_kquant_matmul(void) {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (backend == NULL) {
        fprintf(stderr, "c_abi: kquant: no CPU backend\n");
        return 1;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    /* ISO C has no cast from an object pointer to a function pointer; memcpy is the conversion it allows. */
    void * proc = ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    ggml_backend_set_n_threads_t set_threads = NULL;
    if (proc != NULL) {
        memcpy(&set_threads, &proc, sizeof(set_threads));
    }
    if (set_threads != NULL) {
        set_threads(backend, 3);
    }
    const enum ggml_type types[] = {GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,
                                    GGML_TYPE_Q6_K};
    const int64_t columns[] = {1, 2, 7};
    int failed = 0;
    for (size_t t = 0; t < sizeof(types) / sizeof(types[0]); t++) {
        for (size_t c = 0; c < sizeof(columns) / sizeof(columns[0]); c++) {
            failed |= jllama_kq_check_type(backend, types[t], 5, columns[c], 2 * JLLAMA_KQ_BLOCK,
                                           (uint32_t) (17 + 31 * t + c));
        }
    }
    ggml_backend_free(backend);
    return failed;
}

#endif
