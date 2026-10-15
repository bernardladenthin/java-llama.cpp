<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT
-->

# The C API of libjllama (`jllama_api.h`)

Every natives directory of this project (CLAUDE.md, "Natives jars") holds two layers a program
without a JVM can use:

| Layer | Files next to `libjllama` | Header | Who defines it |
|---|---|---|---|
| A, upstream's own | `libllama.so` / `llama.dll`, `libmtmd.so` / `mtmd.dll`, `libggml*.so` / `ggml*.dll` | upstream's `llama.h`, `mtmd.h`, `ggml.h` (at the pinned tag, with [this project's patches](patches.md)) | ggml-org/llama.cpp |
| B, this project's | `libjllama.so` / `jllama.dll` / `libjllama.dylib` | `jllama_api.h`, shipped in the directory | this repository |

Layer A is the drop-in surface: the same file names and symbols as upstream's release zips, built
from the patched source. Layer B is what upstream will never offer in that form -- llama.cpp's
*server* core (OpenAI chat completions with templates, tool calls and reasoning, streaming by task
id, the log sink) as plain C functions over an opaque handle, with JSON text in and out. The Java
entry points of `LlamaModel` are a thin shell over the same functions, so what is tested here is
what Java gets.

## Status of the surface

Every operation of `LlamaModel` and `LlamaQuantizer` is in it (the roadmap's "B.2 spike" and the
`LlamaModel` half of "B.3", TODO.md item 7):

| Function | What it does |
|---|---|
| `jllama_api_version()` | the `net.ladenthin:llama` version this library was built as, e.g. `5.2.0`; the C API is versioned with the library (a function present in `x.y.z` keeps its semantics in every later `x.y.*`) |
| `jllama_build_info()` | the llama.cpp build inside, `b<number>-<commit>` |
| `jllama_last_error()` | the message of the last failed call on this thread (thread-local) |
| `jllama_string_free()`, `jllama_floats_free()`, `jllama_tokens_free()` | release what the library returned through an out-parameter |
| `jllama_model_load()` / `jllama_model_free()` | a model from llama-server's command line (`-m`, `-c`, `-ngl`, ... and `--vocab-only`), with an optional load-progress callback |
| `jllama_model_meta()` | the model's metadata object (vocab, context, modalities, chat template, the GGUF key/value map) |
| `jllama_chat_completions()` | one OpenAI chat completion, blocking |
| `jllama_chat_completions_stream_request()` / `_receive()` / `jllama_task_release()` | a streamed one: a task id, then one `{"data": <chunk>, "stop": <bool>}` envelope per call until `stop` |
| `jllama_chat_completions_request()` | a streamed chat completion whose results are llama.cpp's raw result objects (received with `jllama_completions_stream_receive`) |
| `jllama_completions()`, `jllama_completions_oai()`, `jllama_infill()` | a raw `/completion`, an OpenAI `/v1/completions`, a fill-in-the-middle request, blocking |
| `jllama_completions_stream_request()` / `jllama_completions_stream_receive()` | a raw streamed completion (an infill one when the body has the infill fields): each result object with a `"stop"` key |
| `jllama_apply_template()` | the chat template applied to a messages body: the prompt text |
| `jllama_embed()`, `jllama_embeddings()` | one embedding vector; an `/embeddings` or `/v1/embeddings` request (`--embedding`) |
| `jllama_rerank()` | documents scored against a query (`--reranking`) |
| `jllama_tokenize()`, `jllama_tokenize_json()`, `jllama_detokenize()`, `jllama_detokenize_json()` | ids or the `/tokenize` response (with pieces on request); bytes with their length, or the `/detokenize` response |
| `jllama_metrics()` | the counters plus the `slots` array (`LlamaModel.getMetrics()`) |
| `jllama_slot_action()` | save, restore or erase a slot's KV cache |
| `jllama_lora_adapters()`, `jllama_set_lora_adapters()` | the LoRA list; the scales from `[{"id", "scale"}]` |
| `jllama_configure()` | `n_threads`, `n_threads_batch`, `slot_prompt_similarity` at run time |
| `jllama_systemone()` | a `/v1/systemone` request to a decision model |
| `jllama_json_schema_to_grammar()` | a JSON schema as GBNF (no model needed) |
| `jllama_quantize()` | `llama_model_quantize` over a GGUF file (no model needed) |
| `jllama_set_log_callback()` | every log line of llama.cpp, ggml and the server to a callback; null restores the console after a synchronous flush |

Not yet in it (the rest of "B.3"): the embedded HTTP server (`NativeServer`, attach mode, router
mode), the RPC server, text to speech, the trainer. Their Java entry points live in their own
translation units over engines that are already JNI-free; each slice moves one of them behind
`jllama_*` functions.

## Contract

- **Handle.** `jllama_model` is opaque. One handle may be used from several threads; `jllama_model_free`
  waits for the calls in flight on it and must not race a new call -- the caller owns the lifetime.
- **Strings.** Requests and responses are JSON text, the bodies llama.cpp's own HTTP server reads
  and writes (an OpenAI chat-completion request in, a `chat.completion` object out; for a stream,
  `chat.completion.chunk` objects). Strings the caller passes are read during the call only. Strings
  returned through an out-parameter are the caller's and go to `jllama_string_free`.
- **Status.** Every call returns a `jllama_status`: `OK`, `ERROR` (server side), `INVALID_REQUEST`
  (the request's fault -- what the HTTP server answers with 400), `INVALID_ARGUMENT` (a null handle
  or out-parameter), `INTERRUPTED` (the model was freed under the call). On anything but `OK`,
  `jllama_last_error()` has the message. The Java side maps `INVALID_REQUEST` to
  `InvalidRequestException` and the rest to `LlamaException`, the same rule as before.
- **Logging.** `jllama_set_log_callback(cb, user_data)` is process-wide and routes *every* line
  (llama.cpp, ggml, the server's `srv`/`slot` lines) to `cb` instead of the console, on the logger's
  worker thread. `jllama_set_log_callback(NULL, NULL)` drains the lines queued for the previous
  callback before it returns, so its `user_data` may be released afterwards. The verbosity threshold
  is process-wide and reset by every load (`-lv` on the argv), as for Java.

## Example

```c
#include "jllama_api.h"
#include <stdio.h>

int main(void) {
    const char *argv[] = {"jllama", "-m", "model.gguf", "-c", "4096", "-ngl", "99"};
    jllama_model *model = NULL;
    if (jllama_model_load(7, argv, NULL, NULL, &model) != JLLAMA_STATUS_OK) {
        fprintf(stderr, "load failed: %s\n", jllama_last_error());
        return 1;
    }
    char *response = NULL;
    const char *request = "{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":64}";
    if (jllama_chat_completions(model, request, &response) == JLLAMA_STATUS_OK) {
        puts(response); /* an OpenAI chat.completion object */
        jllama_string_free(response);
    } else {
        fprintf(stderr, "%s\n", jllama_last_error());
    }
    jllama_model_free(model);
    return 0;
}
```

Build it against the natives directory (a library jar unpacked, or `net.ladenthin.llama.lib.path`'s
directory; README "Natives outside Maven"):

```sh
cc -I<natives dir> example.c -L<natives dir> -ljllama -Wl,-rpath,<natives dir> -o example
```

On Windows link `jllama.lib` is not shipped: load the DLL with `LoadLibrary` and resolve the
`jllama_*` entry points with `GetProcAddress`, or generate an import library from the DLL's export
table. Streaming:

```c
int task = 0;
const char *request = "{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"stream\":true}";
if (jllama_chat_completions_stream_request(model, request, &task) == JLLAMA_STATUS_OK) {
    bool stop = false;
    while (!stop) {
        char *envelope = NULL; /* {"data": <chunk or [chunks]>, "stop": <bool>} */
        if (jllama_chat_completions_stream_receive(model, task, &envelope, &stop) != JLLAMA_STATUS_OK) {
            fprintf(stderr, "%s\n", jllama_last_error());
            jllama_task_release(model, task); /* a task not received to its end is released by hand */
            break;
        }
        puts(envelope);
        jllama_string_free(envelope);
    }
}
```

## How it is held

- `src/test/cpp/test_c_api.cpp` (gtest, `C++ Tests` on every desktop platform): a model is loaded
  from a tiny random llama-architecture GGUF the test writes itself (hermetic: no download, no
  network), a chat completion answered with a `chat.completion` object, a stream received to its
  stop and found released, the log sink fed with the server's lines and flushed synchronously, the
  status codes of a malformed body, a refused body, a missing model and null arguments pinned;
  then the model meta, raw and OpenAI completions, raw streaming (and the chat-template variant),
  the template, a tokenize/detokenize round trip, metrics, slot erase, the LoRA list, configure,
  the refusals of a model loaded without `--embedding`/`--reranking` and without FIM tokens, the
  grammar conversion, and a quantisation of the random model to Q8_0 that loads again.
- `src/test/c/test_c_abi.c` (the ctest `c_abi` of every modular build job): plain C11 linking the
  shipped `libjllama` and `libllama`; it compiles the header as C and resolves `jllama_api_version`
  and `jllama_build_info` through the library's exports, next to its layer-A checks.
- `.github/verify-native-deps.py` (`package`): `libjllama` exports `JNI_OnLoad`, `JNI_OnUnload`,
  `Java_*` and `jllama_*`, nothing else.
- The Java suite: every native method of `LlamaModel` and `LlamaQuantizer` calls these functions
  (the JNI file holds no server logic any more), so every model-backed Java test runs through the
  C API -- including the ones the random model cannot exercise here (embeddings, rerank, infill,
  a decision model), which CI runs against the real models.

## Implementation notes

`src/main/cpp/jllama_engine.{hpp,cpp}` is the implementation both shells share: C++ over
`server_context`, exceptions for errors (`jllama_invalid_request` for the request's fault,
`jllama_interrupted` for a wait cut by a free, `std::runtime_error` otherwise), no JNI in it.
`jllama_api.cpp` turns that into status codes, a thread-local error string and malloc'd strings, and
runs every call inside the user-reference guard (`jllama_context.hpp`) so `jllama_model_free` can
wait for the calls in flight -- the same protocol `LlamaModel.close()` relies on. The inference loop
is a plain thread: it attaches to nothing, so the same loop serves a C caller without a JVM.
