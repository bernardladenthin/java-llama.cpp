// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
//
// MUL_MAT over quantized weights through the CPU backend ggml loaded -- in a modular build the CPU
// variant module ggml_backend_load_best() chose for the running CPU -- held to a reference computed
// here from ggml-base's dequantization of the weights. The CPU's dot products work on activations
// quantized to the type's vec_dot_type (Q8_K for the K-quants and IQ4_XS, Q8_0 or Q8_1 for the 32-value
// types), which ggml-base does not export a conversion for. Three kinds of activations:
//
// - random (K-quants only): the activations quantized to Q8_K the way ggml's own reference quantizer
//   does (quantize_row_q8_K_ref); the result then differs from the reference only by the order of the
//   float sums and by a rounding a SIMD quantizer may take differently from it.
// - grid (every type): per block a power-of-two step s, one value at +-127 s (the block's maximum) and
//   the others integer multiples of s in [-31, 31]. Every quantizer of Q8_0, Q8_1 and Q8_K reproduces
//   such a block exactly -- its scale is s, every quotient an integer, a Q8_1 block sum s times at most
//   1088, exact in its fp16 -- so the reference takes the activations as they are, and the kernel's
//   integer sums can be held to it without a rounding that is not its own.
// - extreme (every type): weights and activations of one magnitude with alternating, equal signs, so
//   every product is the largest the two quantized blocks allow and of one sign: the case in which a
//   kernel that adds two products into one 16-bit lane comes closest to its limit (2 * 127 * 127 =
//   32258 for Q8_0 weights).
//
// Why here and not in jllama_test: since llama.cpp b12567 (#32600) the s390x CPU modules built with
// VXE (z15, z16) take K-quant products through llamafile's tinyBLAS_QK_VXE kernel, and since b12831
// (#32253) their dot products of Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, MXFP4, IQ4_NL and IQ4_XS add pairs of
// products in 16-bit lanes, their Q8_0 quantizer saturates and tinyBLAS_Q0_VXE sums two Q8_0 products
// in 16 bits. The only place CI runs such a module is the c_abi test of build-linux-s390x, under
// qemu-s390x (its CPU model has VXE2, so ggml loads the z15 module). The static s390x test build is
// scalar and never reaches the kernels. On the other platforms the check covers whatever path their
// CPU module takes. The weights also go through the CPU_REPACK buffer where the CPU repacks their type
// (on s390x Q4_0's 4x4 gemv and gemm; on x86 and ARM several types).
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

enum jllama_kq_mode { JLLAMA_KQ_RANDOM, JLLAMA_KQ_GRID, JLLAMA_KQ_EXTREME };

static const char * jllama_kq_mode_name(enum jllama_kq_mode mode) {
    return mode == JLLAMA_KQ_RANDOM ? "random" : mode == JLLAMA_KQ_GRID ? "grid" : "extreme";
}

/* A deterministic value in [-1, 1). */
static float jllama_kq_random(uint32_t * state) {
    *state = *state * 1664525u + 1013904223u;
    return (float) ((*state >> 8) & 0xffffff) / 8388608.0f - 1.0f;
}

/* The block of the activations a type's dot product reads: Q8_K for the K-quants and IQ4_XS, else Q8_0
 * or Q8_1, both of 32 values. */
static int64_t jllama_kq_activation_block(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ4_XS:
            return JLLAMA_KQ_BLOCK;
        default:
            return 32;
    }
}

/* The sign of value j in the extreme pattern, the same for weights and activations. */
static float jllama_kq_extreme_sign(int64_t j) {
    return (j & 1) ? 1.0f : -1.0f;
}

/* Activations on the grid of the header comment (or the extreme pattern), blocks of `block` values. */
static void jllama_kq_grid_activations(float * x, int64_t k, int64_t block, enum jllama_kq_mode mode,
                                       uint32_t * state) {
    for (int64_t b = 0; b < k / block; b++) {
        const float s = ldexpf(1.0f, -(5 + (int) (b % 3)));
        const int64_t peak = (b * 7) % block;
        for (int64_t j = 0; j < block; j++) {
            float q;
            if (mode == JLLAMA_KQ_EXTREME) {
                q = 127.0f * jllama_kq_extreme_sign(b * block + j);
            } else if (j == peak) {
                q = jllama_kq_random(state) >= 0.0f ? 127.0f : -127.0f;
            } else {
                q = (float) lrintf(31.0f * jllama_kq_random(state));
            }
            x[b * block + j] = q * s;
        }
    }
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

/* One type: an m x k weight matrix of `type` times n activation columns of k values, on `backend`; the
 * weights in a buffer of `wbuft` (NULL: the backend's own). Returns 0 when every product is within
 * tolerance of the reference, or when `wbuft` does not repack the type on this CPU (said so), else 1
 * (after printing it). */
static int jllama_kq_check_type(ggml_backend_t backend, ggml_backend_buffer_type_t wbuft, enum ggml_type type,
                                enum jllama_kq_mode mode, int64_t m, int64_t n, int64_t k, uint32_t seed) {
    const struct ggml_init_params ip = {ggml_tensor_overhead() * 8 + ggml_graph_overhead(), NULL, true};
    struct ggml_context * wctx = ggml_init(ip);
    struct ggml_context * ctx = ggml_init(ip);
    if (wctx == NULL || ctx == NULL) {
        fprintf(stderr, "c_abi: matmul: ggml_init failed\n");
        if (wctx != NULL) {
            ggml_free(wctx);
        }
        if (ctx != NULL) {
            ggml_free(ctx);
        }
        return 1;
    }
    struct ggml_tensor * w = ggml_new_tensor_2d(wctx, type, k, m);
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    struct ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t wbuf = wbuft != NULL ? ggml_backend_alloc_ctx_tensors_from_buft(wctx, wbuft)
                                               : ggml_backend_alloc_ctx_tensors(wctx, backend);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (wbuf == NULL || buf == NULL) {
        fprintf(stderr, "c_abi: matmul: no buffer for %s\n", ggml_type_name(type));
        if (wbuf != NULL) {
            ggml_backend_buffer_free(wbuf);
        }
        if (buf != NULL) {
            ggml_backend_buffer_free(buf);
        }
        ggml_free(ctx);
        ggml_free(wctx);
        return 1;
    }
    const char * where = wbuft != NULL ? ggml_backend_buft_name(wbuft) : "plain";
    if (wbuft != NULL && w->extra == NULL) {
        /* The buffer keeps no repacked layout for this type and shape on this CPU: nothing to check. */
        printf("c_abi: matmul: %s is not repacked on this CPU\n", ggml_type_name(type));
        ggml_backend_buffer_free(buf);
        ggml_backend_buffer_free(wbuf);
        ggml_free(ctx);
        ggml_free(wctx);
        return 0;
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
        fprintf(stderr, "c_abi: matmul: out of memory\n");
        failed = 1;
    } else {
        /* Every row gets its own magnitude, so the per-block scales differ between the rows. */
        for (int64_t i = 0; i < m; i++) {
            const float row_scale = 0.25f + 0.5f * (float) (i % 5);
            for (int64_t j = 0; j < k; j++) {
                wf[i * k + j] = row_scale * (mode == JLLAMA_KQ_EXTREME ? jllama_kq_extreme_sign(j)
                                                                       : jllama_kq_random(&state));
            }
        }
        if (mode == JLLAMA_KQ_RANDOM) {
            for (int64_t i = 0; i < n * k; i++) {
                xf[i] = jllama_kq_random(&state);
            }
        } else {
            for (int64_t c = 0; c < n; c++) {
                jllama_kq_grid_activations(xf + c * k, k, jllama_kq_activation_block(type), mode, &state);
            }
        }
        ggml_quantize_chunk(type, wf, wq, 0, m, k, NULL);
        ggml_backend_tensor_set(w, wq, 0, wbytes);
        ggml_backend_tensor_set(x, xf, 0, sizeof(float) * (size_t) (n * k));
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "c_abi: matmul: the graph did not compute for %s\n", ggml_type_name(type));
            failed = 1;
        } else {
            ggml_backend_tensor_get(y, yf, 0, sizeof(float) * (size_t) (m * n));
            /* The reference: the weights as ggml-base dequantizes them, the activations as the dot
             * product sees them -- through Q8_K for random ones, as they are on the grid. */
            ggml_get_type_traits(type)->to_float(wq, wf, m * k);
            for (int64_t c = 0; c < n; c++) {
                if (mode == JLLAMA_KQ_RANDOM) {
                    jllama_kq_through_q8_k(xf + c * k, xq + c * k, k);
                } else {
                    memcpy(xq + c * k, xf + c * k, sizeof(float) * (size_t) k);
                }
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
                                "c_abi: matmul: %s (%s, %s) row %lld column %lld: %.6f, the reference %.6f "
                                "(relative %.3g)\n",
                                ggml_type_name(type), where, jllama_kq_mode_name(mode), (long long) r,
                                (long long) c, got, sum, err);
                        failed = 1;
                        break;
                    }
                }
            }
            if (!failed) {
                printf("c_abi: matmul: %s (%s, %s) %lldx%lld times %lld columns within %.2g of the reference\n",
                       ggml_type_name(type), where, jllama_kq_mode_name(mode), (long long) m, (long long) k,
                       (long long) n, worst);
            }
        }
    }
    free(wq);
    free(yf);
    free(xq);
    free(xf);
    free(wf);
    ggml_backend_buffer_free(buf);
    ggml_backend_buffer_free(wbuf);
    ggml_free(ctx);
    ggml_free(wctx);
    return failed;
}

/* The CPU device's CPU_REPACK buffer type, or NULL when the build has none. */
static ggml_backend_buffer_type_t jllama_kq_repack_buffer_type(ggml_backend_t backend) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    void * proc = ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    ggml_backend_dev_get_extra_bufts_t get_extra = NULL;
    if (proc == NULL) {
        return NULL;
    }
    memcpy(&get_extra, &proc, sizeof(get_extra));
    for (ggml_backend_buffer_type_t * buft = get_extra(dev); buft != NULL && *buft != NULL; ++buft) {
        if (strcmp(ggml_backend_buft_name(*buft), "CPU_REPACK") == 0) {
            return *buft;
        }
    }
    return NULL;
}

/* Every quantized type through the CPU backend, on three threads: six rows -- llamafile's
 * tinyBLAS_Q0_VXE takes only even row and column counts, tinyBLAS_QK_VXE any -- times column counts
 * that take each of tinyBLAS_QK_VXE's tile widths (4, 2, 1) and the one-column path ggml keeps for
 * itself; two columns reach tinyBLAS_Q0_VXE, one and seven the type's dot product. Then the types a
 * CPU may repack, through the CPU_REPACK buffer, for one column (gemv) and seven (gemm). Returns 0 on
 * success. */
static int jllama_check_kquant_matmul(void) {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (backend == NULL) {
        fprintf(stderr, "c_abi: matmul: no CPU backend\n");
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
    const enum ggml_type k_types[] = {GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,
                                      GGML_TYPE_Q6_K};
    const enum ggml_type types[] = {GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K,  GGML_TYPE_Q5_K,
                                    GGML_TYPE_Q6_K, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,  GGML_TYPE_Q5_0,
                                    GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_MXFP4, GGML_TYPE_IQ4_NL,
                                    GGML_TYPE_IQ4_XS};
    const enum ggml_type repack_types[] = {GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_0,
                                           GGML_TYPE_MXFP4};
    const int64_t columns[] = {1, 2, 7};
    const int64_t repack_columns[] = {1, 7};
    int failed = 0;
    for (size_t t = 0; t < sizeof(k_types) / sizeof(k_types[0]); t++) {
        for (size_t c = 0; c < sizeof(columns) / sizeof(columns[0]); c++) {
            failed |= jllama_kq_check_type(backend, NULL, k_types[t], JLLAMA_KQ_RANDOM, 6, columns[c],
                                           2 * JLLAMA_KQ_BLOCK, (uint32_t) (17 + 31 * t + c));
        }
    }
    for (size_t t = 0; t < sizeof(types) / sizeof(types[0]); t++) {
        for (size_t c = 0; c < sizeof(columns) / sizeof(columns[0]); c++) {
            failed |= jllama_kq_check_type(backend, NULL, types[t], JLLAMA_KQ_GRID, 6, columns[c],
                                           2 * JLLAMA_KQ_BLOCK, (uint32_t) (101 + 31 * t + c));
            failed |= jllama_kq_check_type(backend, NULL, types[t], JLLAMA_KQ_EXTREME, 6, columns[c],
                                           2 * JLLAMA_KQ_BLOCK, (uint32_t) (211 + 31 * t + c));
        }
    }
    ggml_backend_buffer_type_t repack = jllama_kq_repack_buffer_type(backend);
    if (repack == NULL) {
        printf("c_abi: matmul: this build has no CPU_REPACK buffer\n");
    } else {
        for (size_t t = 0; t < sizeof(repack_types) / sizeof(repack_types[0]); t++) {
            for (size_t c = 0; c < sizeof(repack_columns) / sizeof(repack_columns[0]); c++) {
                failed |= jllama_kq_check_type(backend, repack, repack_types[t], JLLAMA_KQ_GRID, 16,
                                               repack_columns[c], 2 * JLLAMA_KQ_BLOCK, (uint32_t) (307 + 31 * t + c));
                failed |= jllama_kq_check_type(backend, repack, repack_types[t], JLLAMA_KQ_EXTREME, 16,
                                               repack_columns[c], 2 * JLLAMA_KQ_BLOCK, (uint32_t) (401 + 31 * t + c));
            }
        }
    }
    ggml_backend_free(backend);
    return failed;
}

#endif
