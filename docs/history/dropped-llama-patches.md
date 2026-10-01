<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT
-->

# Dropped llama.cpp patches

The local patches under `llama/patches/` that were retired because upstream fixed the defect
themselves. Moved here from `CLAUDE.md`, which lists only the patches still carried. Each note
records why the patch went and what guard, if any, stays behind -- read the matching note before
reintroducing a patch for the same defect.

**`0010` was dropped at the b11080 bump.** Upstream merged
[ggml-org/llama.cpp#28518](https://github.com/ggml-org/llama.cpp/pull/28518)
("json: Fixed json enum handling", first tagged at **b11080**), which fixes the defect at its
root instead of at the emit site: `common_json_value` gains an `std::is_enum`-gated constructor
that delegates to `std::underlying_type`, and `common_json_is_value` now accepts enums. So an
unscoped enum no longer binds to `common_json_value(bool)` anywhere, and the one-line `(int)`
cast `0010` added to upstream's `get_res_model_info()` became a **redundant carry**.

**This is the drop-check firing, and it is the case that makes the check worth running by hand.**
`0010` still applied cleanly at b11080 — upstream never touched the emit site — so the fail-loud
applier said nothing, exactly as `CLAUDE.md` warned it could not ("the applier detects 'does not
apply', never 'upstream already fixed this'"). The standing check is phrased as *"did upstream
cast the value themselves?"*; the honest reading is *"is the defect still there?"*, and it was
not. Dropped, not refreshed, per the `0009`/`0011`/`0013` precedent.

**The runnable guard stays, re-pointed** (the `0011` precedent again): the `CommonJsonEnumTrap`
pair in `src/test/cpp/test_json_helpers.cpp` is now the `CommonJsonEnum` trio, and it pins
upstream's contract — an **uncast** enum serialises as its numeric value, an explicit
`static_cast<int>` is equivalent, and a real `bool` is still a boolean (the new constructor sits
next to `common_json_value(bool)`, so that one is worth pinning too). A future bump that loses
the enum constructor reds `C++ Tests` on every platform, and the response is to reinstate both
the cast and the patch. `jllama.cpp` keeps the explicit casts at its own two `"vocab_type"`
sites: they are correct either way and survive such a revert.

`.github/verify-patches-applied.sh` lost its third check with the patch — `0010` was the only
patch with no runnable guard, which is what that check existed for. The script keeps its two
generic assertions (every patch on disk is in the stamp; the patched tree is dirty), so it still
catches an applier that never ran or a tree reverted after the fact.

**`0011` was dropped at the b11069 bump.** Upstream merged
[ggml-org/llama.cpp#29161](https://github.com/ggml-org/llama.cpp/pull/29161)
("common/peg : handle invalid utf-8 sequences in the AST", commit `3d82ef62d`, first tagged at
**b11063**) — an independent and broader fix for the same defect; the patch itself had never been
filed upstream. Where `0011` made the until-parser's `INVALID` branch honour `ctx.is_lenient()` by
returning what was scanned *before* the bad byte (so the text after it was lost), upstream now
**consumes** every undecodable run, in strict mode too, records its `{pos, len}` on the AST node
(`common_peg_invalid_utf8`, carried up through `common_peg_parse_result`), and
`common_chat_peg_mapper` emits `node.sanitized_text()`, which substitutes exactly one U+FFFD per run
(the Unicode "maximal subpart" rule: `\xE4\xB8` followed by `c` is one run, `\xFF\xFE` is two).
`common/unicode.cpp` now reports the valid-prefix length in `bytes_consumed` for `INVALID` /
`INCOMPLETE` results to make that possible. The lenient incomplete-at-end-of-input branch (withhold
the trailing bytes, more may arrive on a stream) is unchanged. The applier failed loud with "does not
apply" at `common/peg-parser.cpp:680` exactly as designed — the `INVALID` branch it patched no longer
exists. Dropped, not refreshed, per the `0009`/`0013` precedent. **The runnable guard stays,
re-pointed:** the `ContentOnlyParseUtf8` tests in `src/test/cpp/test_utils.cpp` now pin upstream's
replacement contract (invalid byte → U+FFFD with the text after it kept; a trailing incomplete
sequence still withheld), so a future upstream revert to `FAIL`, or a change to the replacement rule,
still reds `C++ Tests` on every platform. The strict-mode `FAIL` on invalid UTF-8 that `0011`
deliberately preserved is gone upstream as well — `tests/peg-parser/test-unicode.cpp`'s
`malformed UTF-8` block now asserts `SUCCESS` plus the sanitized text. **Behavioural note for
consumers:** a completion containing a stray byte now returns the *full* text with U+FFFD in place of
the byte, where the patched builds returned the text up to the byte.

**`0013` was dropped at the b10948 bump.** Upstream merged this project's own PR
[ggml-org/llama.cpp#28775](https://github.com/ggml-org/llama.cpp/pull/28775)
("ggml-cpu(s390x): guard VXE-only repack helpers", commit `6978052`, first tagged at **b10948**):
`ggml/src/ggml-cpu/arch/s390/repack.cpp` now wraps `vxe_dot_acc` / `vxe_splat_granule` / `vxe_fold`
in the same `#if defined(__VXE__) || defined(__VXE2__)` guard the patch added — byte-identical apart
from the trailing `// __VXE__ || __VXE2__` comment the patch put on the `#endif`. So the non-VXE
s390x compile break the patch fixed (three `does not name a type` errors before any skipped code is
reached) no longer exists upstream, and the applier failed loud with "does not apply cleanly" at
configure time exactly as designed — the guard it wants to add is already there. Dropped, not
refreshed, per the `0009` precedent.

**The s390x configuration this leaves in place is still ours to know, because it is what made the
patch necessary and it did not change.** `ggml/CMakeLists.txt` declares
`option(GGML_VXE "ggml: enable vxe" ${GGML_NATIVE})`, so the `build-linux-s390x` job's
`-DGGML_NATIVE=OFF` (correct for a cross-build — an x86 host must not bake `-march=native` into an
s390x artifact) silently switches VXE off too: no `-mvx -mzvector`, `__VEC__` undefined, and
`ggml-cpu-impl.h`'s `#if defined(__s390x__) && defined(__VEC__)` self-define of `__VXE__`/`__VXE2__`
never fires. The job therefore ships a **scalar** s390x binary, which is right for what it is (a
big-endian *correctness* gate for this project's own layer, not a performance target). **Do not
"fix" a future VXE-related compile error with `-DGGML_VXE=ON`** — measured, not assumed: that
self-define sets `__VXE__` *and* `__VXE2__` together while `-march` stays at the toolchain default
`arch11`, so a handful of errors becomes dozens of `'__builtin_s390_vec_*' matching variant requires
z14 or higher`. The only flag-side alternative is `-DGGML_VXE=ON` **plus** `-march=z15`, which
compiles but raises the artifact's hardware floor to z15 and makes the qemu `ctest` gate depend on
VXE2 emulation. If a comparable break resurfaces, re-check upstream's guard against this description
before reintroducing a local patch.

**`0009` was dropped at the b10280 bump.** Upstream merged
[sheredom/subprocess.h#104](https://github.com/sheredom/subprocess.h/pull/104) — the exact fix this
patch submitted — via [ggml-org/llama.cpp#26606](https://github.com/ggml-org/llama.cpp/pull/26606)
("vendor: apply patches for subprocess.h"): `vendor/sheredom/subprocess.h` at b10280 already defines
the `SUBPROCESS_HAVE_CWD` compile-time probe and the `ENOSYS` fallback our patch added, so the
old-glibc build break `0009` fixed no longer exists upstream. The applier's idempotent
`git apply --reverse --check` could **not** auto-skip this one (unlike a byte-identical carry): the
same PR also rewrote the neighboring Windows argv-quoting logic and added a `__NetBSD__` branch next
to the `process_cwd` guard, shifting the patch's second hunk's context, so it failed loud with "does
not apply cleanly" at configure time exactly as designed — confirming the patch needed to be dropped,
not refreshed. Also folded into that PR: a fix for `-std=c++20` unknown to GCC 8
([#105](https://github.com/sheredom/subprocess.h/pull/105)) and a `posix_spawn` exec-failure
reporting fix pre-glibc-2.24 ([#106](https://github.com/sheredom/subprocess.h/pull/106)) — neither
affects this project. If a regression surfaces on old-glibc builds (manylinux2014/manylinux_2_28),
re-check `vendor/sheredom/subprocess.h`'s `SUBPROCESS_HAVE_CWD` guard against this description before
reintroducing a local patch.

**`0005` was dropped at the b9981 bump.** Upstream's own `server-context.cpp` picked up an
equivalent — and broader — fix for the same checkpoint-starvation problem: `create_checkpoint`
now tracks `id_task` and evicts a stale checkpoint that sits within `checkpoint_min_step` of a
newer one (instead of our pre-creation duplicate-position check), and the `do_checkpoint` gate
now exempts **every** near-prompt-end checkpoint from the min-step spacing (`near_prompt_end`,
unconditionally) rather than only recurrent/hybrid (`ctx_tgt_seq_rm_type` `FULL`/`RS`) models as
our patch scoped it. The upstream version is a strict superset of what `0005` did, so it applies
cleanly with the patch removed and needs no replacement. If a regression in agentic multi-turn
prefill behavior shows up, re-check `tools/server/server-context.cpp`'s `create_checkpoint` /
`do_checkpoint` logic against this description before reintroducing a local patch.

**`0004` was dropped at the b9982 bump.** Upstream merged
[ggml-org/llama.cpp#23116](https://github.com/ggml-org/llama.cpp/pull/23116) verbatim: the
`b9981...b9982` diff carries the exact same `oaicompat_chat_params_parse` precedence change
(`reasoning_budget_tokens` > `thinking_budget_tokens` alias > server default, plus the per-request
`reasoning_budget_message` override) and the same two `tests/test-chat.cpp` regression cases
(`test_reasoning_budget_tokens_per_request` / `test_reasoning_budget_message_per_request`,
byte-identical body). No local patch needed — the tree already matches what `0004` used to add.
