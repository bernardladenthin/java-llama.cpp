#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Write a GGUF with random weights in the real shape of a published model, for throughput work.

The output of such a model is noise; what is real is every tensor's shape and type, i.e. the work
and the bytes a token costs. Tensors are written one at a time, so a 12 GB model needs no 12 GB of
RAM.

  qwen3-0.6b / qwen3-4b   F16, to be quantized with llama-quantize (Q4_K_M gives the real mix)
  gpt-oss-20b             ggml-org's gpt-oss-20b-mxfp4.gguf layout: experts MXFP4, every other
                          matrix Q8_0 (llama-quantize's MXFP4_MOE rule), norms/biases/router F32

Needs gguf-py (PYTHONPATH=<llama.cpp>/gguf-py) and numpy.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import gguf

QK = 32  # block size of Q8_0 and MXFP4

MODELS = {
    # Qwen/Qwen3-0.6B config.json
    "qwen3-0.6b": dict(arch="qwen3", n_embd=1024, n_layer=28, n_head=16, n_head_kv=8, head_dim=128,
                       n_ff=3072, n_vocab=151936, n_ctx=40960, rope_theta=1_000_000.0, eps=1e-6),
    # Qwen/Qwen3-4B-Instruct-2507 config.json
    "qwen3-4b": dict(arch="qwen3", n_embd=2560, n_layer=36, n_head=32, n_head_kv=8, head_dim=128,
                     n_ff=9728, n_vocab=151936, n_ctx=262144, rope_theta=5_000_000.0, eps=1e-6),
    # openai/gpt-oss-20b config.json
    "gpt-oss-20b": dict(arch="gpt-oss", n_embd=2880, n_layer=24, n_head=64, n_head_kv=8, head_dim=64,
                        n_ff=2880, n_vocab=201088, n_ctx=131072, rope_theta=150_000.0, eps=1e-5,
                        n_expert=32, n_expert_used=4, sliding_window=128),
}


def vocab(n_vocab: int) -> tuple[list[str], list[int]]:
    """GPT-2 byte tokens first (any text tokenizes, byte by byte), then unique fillers."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    tokens = [chr(c) for c in cs]
    tokens += [f"tok{i}" for i in range(n_vocab - 256 - 2)]
    tokens += ["<|bos|>", "<|eos|>"]
    types = [1] * (n_vocab - 2) + [3, 3]  # normal, control
    return tokens, types


class Writer:
    def __init__(self, path: str, arch: str, seed: int):
        self.w = gguf.GGUFWriter(path, arch)
        self.rng = np.random.default_rng(seed)
        self.pending: list[tuple[str, callable]] = []

    def f32(self, name, shape, mean=0.0, std=0.02):
        n = int(np.prod(shape))
        self.w.add_tensor_info(name, shape, np.dtype(np.float32), n * 4)
        self.pending.append((name, lambda: (mean + std * self.rng.standard_normal(n, dtype=np.float32)).reshape(shape)))

    def f16(self, name, shape, std=0.02):
        n = int(np.prod(shape))
        self.w.add_tensor_info(name, shape, np.dtype(np.float16), n * 2)
        self.pending.append((name, lambda: (std * self.rng.standard_normal(n, dtype=np.float32)).astype(np.float16).reshape(shape)))

    def q8_0(self, name, shape, std=0.02):
        # shape in numpy order (..., ne0); block: f16 d, 32 x int8
        *outer, ne0 = shape
        nb = int(np.prod(outer)) * ne0 // QK
        byte_shape = (*outer, ne0 // QK * 34)
        self.w.add_tensor_info(name, byte_shape, np.dtype(np.uint8), nb * 34, raw_dtype=gguf.GGMLQuantizationType.Q8_0)

        def gen():
            blk = np.empty(nb, dtype=[("d", "<f2"), ("qs", "i1", QK)])
            blk["d"] = np.float16(std * 3 / 127)
            blk["qs"] = np.clip(np.rint(self.rng.standard_normal((nb, QK), dtype=np.float32) * 42), -127, 127).astype(np.int8)
            return blk.view(np.uint8).reshape(byte_shape)
        self.pending.append((name, gen))

    def mxfp4(self, name, shape):
        # block: e8m0 exponent, 16 bytes of nibbles; values kvalues_mxfp4 * 2^(e-127) / 2
        *outer, ne0 = shape
        nb = int(np.prod(outer)) * ne0 // QK
        byte_shape = (*outer, ne0 // QK * 17)
        self.w.add_tensor_info(name, byte_shape, np.dtype(np.uint8), nb * 17, raw_dtype=gguf.GGMLQuantizationType.MXFP4)

        def gen():
            blk = np.empty((nb, 17), dtype=np.uint8)
            blk[:, 0] = self.rng.integers(118, 122, nb, dtype=np.uint8)  # |w| <= 12 * 2^-6 / 2
            blk[:, 1:] = self.rng.integers(0, 256, (nb, 16), dtype=np.uint8)
            return blk.reshape(byte_shape)
        self.pending.append((name, gen))

    def finish(self):
        self.w.write_header_to_file()
        self.w.write_kv_data_to_file()
        self.w.write_ti_data_to_file()
        for i, (name, gen) in enumerate(self.pending):
            self.w.write_tensor_data(gen())
            print(f"[{i + 1}/{len(self.pending)}] {name}", file=sys.stderr, flush=True)
        self.w.close()


def common_kv(w: gguf.GGUFWriter, m: dict, n_layer: int, name: str):
    w.add_name(f"mock-{name}")
    w.add_context_length(m["n_ctx"])
    w.add_embedding_length(m["n_embd"])
    w.add_block_count(n_layer)
    w.add_feed_forward_length(m["n_ff"])
    w.add_head_count(m["n_head"])
    w.add_head_count_kv(m["n_head_kv"])
    w.add_key_length(m["head_dim"])
    w.add_value_length(m["head_dim"])
    w.add_rope_freq_base(m["rope_theta"])
    w.add_layer_norm_rms_eps(m["eps"])
    tokens, types = vocab(m["n_vocab"])
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("default")
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges(["Ġ Ġ"])
    w.add_bos_token_id(m["n_vocab"] - 2)
    w.add_eos_token_id(m["n_vocab"] - 1)


def write_qwen3(out: str, m: dict, n_layer: int, seed: int, name: str):
    wr = Writer(out, m["arch"], seed)
    common_kv(wr.w, m, n_layer, name)
    wr.w.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
    E, H, KV, D, F = m["n_embd"], m["n_head"], m["n_head_kv"], m["head_dim"], m["n_ff"]
    wr.f16("token_embd.weight", (m["n_vocab"], E))  # tied: also the output head
    wr.f32("output_norm.weight", (E,), 1.0, 0.05)
    for i in range(n_layer):
        p = f"blk.{i}."
        wr.f32(p + "attn_norm.weight", (E,), 1.0, 0.05)
        wr.f16(p + "attn_q.weight", (H * D, E))
        wr.f16(p + "attn_k.weight", (KV * D, E))
        wr.f16(p + "attn_v.weight", (KV * D, E))
        wr.f16(p + "attn_output.weight", (E, H * D))
        wr.f32(p + "attn_q_norm.weight", (D,), 1.0, 0.05)
        wr.f32(p + "attn_k_norm.weight", (D,), 1.0, 0.05)
        wr.f32(p + "ffn_norm.weight", (E,), 1.0, 0.05)
        wr.f16(p + "ffn_gate.weight", (F, E))
        wr.f16(p + "ffn_up.weight", (F, E))
        wr.f16(p + "ffn_down.weight", (E, F))
    wr.finish()


def write_gpt_oss(out: str, m: dict, n_layer: int, seed: int, name: str):
    wr = Writer(out, m["arch"], seed)
    w = wr.w
    common_kv(w, m, n_layer, name)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_MXFP4_MOE)
    w.add_expert_count(m["n_expert"])
    w.add_expert_used_count(m["n_expert_used"])
    w.add_expert_feed_forward_length(m["n_ff"])
    w.add_sliding_window(m["sliding_window"])
    w.add_rope_scaling_type(gguf.RopeScalingType.YARN)
    w.add_rope_scaling_factor(32.0)
    w.add_rope_scaling_orig_ctx_len(4096)
    w.add_rope_scaling_yarn_beta_fast(32.0)
    w.add_rope_scaling_yarn_beta_slow(1.0)
    E, H, KV, D, F, X = m["n_embd"], m["n_head"], m["n_head_kv"], m["head_dim"], m["n_ff"], m["n_expert"]
    wr.q8_0("token_embd.weight", (m["n_vocab"], E))
    wr.f32("output_norm.weight", (E,), 1.0, 0.05)
    wr.q8_0("output.weight", (m["n_vocab"], E))
    for i in range(n_layer):
        p = f"blk.{i}."
        wr.f32(p + "attn_norm.weight", (E,), 1.0, 0.05)
        wr.f32(p + "post_attention_norm.weight", (E,), 1.0, 0.05)
        wr.q8_0(p + "attn_q.weight", (H * D, E))
        wr.f32(p + "attn_q.bias", (H * D,))
        wr.q8_0(p + "attn_k.weight", (KV * D, E))
        wr.f32(p + "attn_k.bias", (KV * D,))
        wr.q8_0(p + "attn_v.weight", (KV * D, E))
        wr.f32(p + "attn_v.bias", (KV * D,))
        wr.q8_0(p + "attn_output.weight", (E, H * D))
        wr.f32(p + "attn_output.bias", (E,))
        wr.f32(p + "attn_sinks.weight", (H,), 0.0, 1.0)
        wr.f32(p + "ffn_gate_inp.weight", (X, E), 0.0, 0.05)
        wr.f32(p + "ffn_gate_inp.bias", (X,))
        wr.mxfp4(p + "ffn_gate_exps.weight", (X, F, E))
        wr.f32(p + "ffn_gate_exps.bias", (X, F))
        wr.mxfp4(p + "ffn_up_exps.weight", (X, F, E))
        wr.f32(p + "ffn_up_exps.bias", (X, F))
        wr.mxfp4(p + "ffn_down_exps.weight", (X, E, F))
        wr.f32(p + "ffn_down_exps.bias", (X, E))
    wr.finish()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=sorted(MODELS))
    ap.add_argument("out")
    ap.add_argument("--layers", type=int, help="fewer layers than the real model (same per-layer work)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    m = MODELS[a.model]
    n_layer = a.layers or m["n_layer"]
    if m["arch"] == "gpt-oss":
        write_gpt_oss(a.out, m, n_layer, a.seed, a.model)
    else:
        write_qwen3(a.out, m, n_layer, a.seed, a.model)


if __name__ == "__main__":
    main()
