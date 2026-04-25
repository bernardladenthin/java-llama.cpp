# Upstream patch: expose `slot_prompt_similarity` setter on `server_context`

> **Tracked artefact, not a build input.** This file documents a proposed
> patch against `ggerganov/llama.cpp` so embedders (including
> `java-llama.cpp`) can mutate `slot_prompt_similarity` after
> `server_context::load_model` returns. It is referenced from
> `REFACTORING_2.md` and from the comment block in
> `src/main/cpp/jllama.cpp::Java_de_kherud_llama_LlamaModel_configureParallelInference`.

---

## Rationale (suitable for the upstream PR description)

`server_context::load_model(common_params &)` initialises
`server_context_impl::slot_prompt_similarity` from
`params_base.slot_prompt_similarity` (server-context.cpp:858) and never
touches it again. The field is private inside the pimpl, so embedders
that drive `server_context` from a non-HTTP front-end (Java/JNI, Rust
FFI, embedded C++) cannot change the slot-selection threshold at runtime
without unloading and reloading the model.

This patch adds a public getter/setter pair on `server_context` that
forwards to the existing `impl` field. No new state, no behavioural
change for existing call sites — the slot-selection logic in
`get_available_slot()` (server-context.cpp:1077-1097) reads the field on
every selection pass, so the next request automatically picks up any
runtime change. No invalidation, no broadcast.

The setter clamps to `[0.0, 1.0]` because the only existing input
contract for this value (the JSON parser in some embedder shells) caps
it there; soft-clamp is safer than throwing across an FFI boundary.
Threading guarantees mirror `get_meta()`: not thread-safe, main thread
only.

---

## Patch (against tag `b8913`)

Pinned base: `https://github.com/ggerganov/llama.cpp/tree/b8913`.
Apply with `git apply` from the repository root:

```bash
git checkout b8913
git apply --check llama-cpp.patch       # extract from the diff block below
git apply llama-cpp.patch
```

### Hunk 1 — `tools/server/server-context.h`

Insert a getter/setter pair after the existing `get_meta()` declaration,
before `on_sleeping_changed`. Verbatim against b8913 line numbers:

```diff
--- a/tools/server/server-context.h
+++ b/tools/server/server-context.h
@@ -77,6 +77,15 @@ struct server_context {
     // not thread-safe, should only be used from the main thread
     server_context_meta get_meta() const;

+    // get the slot-prompt-similarity threshold used during slot selection.
+    // returns 0.0f when LCP-based slot reuse is disabled.
+    // not thread-safe, should only be used from the main thread.
+    float get_slot_prompt_similarity() const;
+
+    // update the slot-prompt-similarity threshold at runtime.
+    // value is clamped to [0.0, 1.0]; the next slot-selection pass picks
+    // up the new value automatically.
+    // not thread-safe, should only be used from the main thread.
+    void set_slot_prompt_similarity(float value);
+
     // register a callback to be called when sleeping state changes
     // must be set before load_model() is called
     void on_sleeping_changed(std::function<void(bool)> callback);
```

### Hunk 2 — `tools/server/server-context.cpp`

Add the two definitions immediately after
`server_context::get_response_reader()` (line 3092-3094) and before
`server_context::get_meta()` (line 3096), so the getter/setter pair sits
next to the related `get_meta()` accessor:

```diff
--- a/tools/server/server-context.cpp
+++ b/tools/server/server-context.cpp
@@ -3092,6 +3092,18 @@ server_response_reader server_context::get_response_reader() {
     return impl->get_response_reader();
 }

+float server_context::get_slot_prompt_similarity() const {
+    return impl->slot_prompt_similarity;
+}
+
+void server_context::set_slot_prompt_similarity(float value) {
+    if (value < 0.0f) value = 0.0f;
+    if (value > 1.0f) value = 1.0f;
+    impl->slot_prompt_similarity = value;
+    SRV_INF("slot_prompt_similarity updated to %.3f at runtime\n", value);
+}
+
 server_context_meta server_context::get_meta() const {
     auto bos_id = llama_vocab_bos(impl->vocab);
     auto eos_id = llama_vocab_eos(impl->vocab);
```

The forwarding pattern matches every other public method on
`server_context` (`load_model`, `start_loop`, `terminate`,
`get_llama_context`, `get_response_reader`, `get_meta`).
`SRV_INF` is the existing log macro used elsewhere in this file for
similar lifecycle notifications.

### What the patch deliberately does NOT do

- **No `server_context_meta` change.** That struct is the read-only
  introspection payload returned by `get_meta()`. A standalone
  getter/setter pair is a better fit here because it sits next to the
  setter and signals mutability.
- **No upstream HTTP route.** This patch is C++-API-only. Adding a
  `POST /slot-prompt-similarity` HTTP endpoint to `server.cpp` is a
  separate concern; embedders that need it can wire one themselves.
- **No mutex.** Matches the documented threading contract on
  `get_meta()` ("not thread-safe, should only be used from the main
  thread"). Adding atomicity would be scope creep for the PR — embedders
  that need it can wrap the setter externally.
- **No setter for `n_threads` / `n_threads_batch`.** `llama.h:946`
  already exposes `llama_set_n_threads(ctx, n, nb)`; embedders can call
  it directly via `server_context::get_llama_context()`. No upstream
  change needed.

---

## Manual verification

Before opening the PR, build the patched upstream and verify the setter
takes effect from a small embedder driver. The driver below is a
minimal reproduction that does not require the HTTP server.

```bash
# 1. Apply the patch on a fresh checkout of b8913.
cd /tmp && git clone https://github.com/ggerganov/llama.cpp.git llama.cpp-patched
cd llama.cpp-patched && git checkout b8913
git apply /home/user/java-llama.cpp/llama-cpp.patch

# 2. Build with the patched server library.
cmake -B build -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build --config Release -j$(nproc) --target llama-server

# 3. Run an existing server smoke test against a tiny model — confirms
#    the patch does not break unrelated behaviour.
./build/bin/llama-server -m /path/to/tiny-model.gguf \
    --slot-prompt-similarity 0.5 --port 8080 &
SERVER_PID=$!
sleep 2
curl -sS http://127.0.0.1:8080/v1/models | jq '.data[0].id'
kill $SERVER_PID

# 4. C++ micro-driver to exercise the new getter/setter directly.
cat > /tmp/driver.cpp <<'CPP'
#include "server-context.h"
#include "common.h"
#include <cassert>
#include <cstdio>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: driver <model.gguf>\n");
        return 1;
    }
    common_params params;
    params.model.path             = argv[1];
    params.slot_prompt_similarity = 0.5f;

    server_context srv;
    if (!srv.load_model(params)) return 2;

    assert(srv.get_slot_prompt_similarity() == 0.5f);
    srv.set_slot_prompt_similarity(0.8f);
    assert(srv.get_slot_prompt_similarity() == 0.8f);
    srv.set_slot_prompt_similarity(2.0f);   // out of range -> clamps to 1.0
    assert(srv.get_slot_prompt_similarity() == 1.0f);
    srv.set_slot_prompt_similarity(-1.0f);  // out of range -> clamps to 0.0
    assert(srv.get_slot_prompt_similarity() == 0.0f);

    srv.terminate();
    printf("OK\n");
    return 0;
}
CPP

# Compile and run the driver against the patched build.
g++ -std=c++17 -I build/_deps/llama.cpp-src/tools/server \
    -I build/_deps/llama.cpp-src/common \
    -I build/_deps/llama.cpp-src/include \
    /tmp/driver.cpp \
    build/tools/server/CMakeFiles/server-context.dir/server-context.cpp.o \
    -L build -lllama -lcommon -lggml \
    -o /tmp/driver
/tmp/driver /path/to/tiny-model.gguf
# Expected output: OK
```

---

## Land sequence (this repo + upstream)

1. **Done** — Plan A landed on `claude/analyze-refactoring-changes-ovOFq`
   (commit `d051b1c`). `n_threads` / `n_threads_batch` work today;
   `slot_prompt_similarity` is still validate-only with a reserved
   comment block in `configureParallelInference` waiting for the
   upstream setter.
2. **Done** — `llama-cpp.patch.md` (this file) lands in the repo root,
   sibling of `REFACTORING.md` and `REFACTORING_2.md`. Tracked artefact;
   not consumed by any build step.
3. **Next** — open a PR against `ggerganov/llama.cpp:master` using the
   diff above. Reference upstream coding style (4-space indent, no
   exceptions, `SRV_INF` for logs).
4. **After the PR is merged into a tagged build `b<N>`** — bump the
   pinned llama.cpp version in this repo (the three-line change
   documented under "Upgrading/Downgrading llama.cpp Version" in
   `CLAUDE.md`):
   - `CMakeLists.txt:100` (`GIT_TAG b8913` → `b<N>`)
   - `README.md` badge + link
   - `CLAUDE.md` "Current llama.cpp pinned version" line

   Then a tiny follow-up commit on the same branch:
   - Uncomment the reserved block in
     `src/main/cpp/jllama.cpp::Java_de_kherud_llama_LlamaModel_configureParallelInference`:
     ```cpp
     if (slot_sim_opt.has_value()) {
         ctx_server->set_slot_prompt_similarity(*slot_sim_opt);
     }
     ```
   - Drop the `(void)slot_sim_opt;` line above it.
   - Update `REFACTORING_2.md`'s "Forward references" section to mark
     the gap closed.
   - Add one Java integration test that calls
     `configureParallelInference("{\"slot_prompt_similarity\":0.8}")`
     followed by a completion request, asserting no exception.

   At that point `configureParallelInference` is fully functional for
   all three documented fields.

---

## Files touched by this artefact

| File | Touch |
|------|-------|
| `llama-cpp.patch.md` | This document. Tracked artefact; not consumed by any build step. |
| `src/main/cpp/jllama.cpp` | (No change in this commit — Plan A already reserved the future call site as a comment.) |
| `REFACTORING_2.md` | Already references this file under "Forward references". |
| `CLAUDE.md` | (No change in this commit — gets updated together with the pin bump in step 4.) |

No CMake changes. No tests changes. No behavioural change for the
project until step 4 lands.
