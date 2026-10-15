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
| `jllama_server_start()`, `jllama_server_start_attached()`, `jllama_server_stop()`, `jllama_server_is_running()`, `jllama_server_set_worker_command()` | the embedded llama-server (`NativeServer`): the full upstream server from an argv, WebUI included, or the HTTP frontend attached to a loaded model; router mode's worker command. Not on Android (server.cpp does not build there) |
| `jllama_rpc_server_serve()`, `jllama_rpc_server_stop()`, `jllama_rpc_server_listening()`, `jllama_rpc_server_devices()`, `jllama_strings_free()` | the RPC server (`RpcServer`): this process's devices served to llama.cpp RPC clients, one per process |
| `jllama_tts_load()`, `jllama_tts_synthesize()`, `jllama_tts_free()`, `jllama_bytes_free()` | text to speech (`TextToSpeech`): a Qwen3-TTS backbone and mmproj, a WAV per call |
| `jllama_finetune()` | fine-tuning (`LlamaTrainer`): one training run from its JSON configuration |

That is every native entry point of the Java API: `LlamaModel`, `LlamaQuantizer`, `NativeServer`,
`RpcServer`, `TextToSpeech` and `LlamaTrainer` are all shells over these functions.

And the `jllama_api.h` header ships in every natives directory, so a consumer of a natives jar
(or of `net.ladenthin.llama.lib.path`'s directory) has the declarations next to the library.

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

On Windows the import library `jllama.lib` ships next to `jllama.dll` in the same directory (named
in `jllama-files.txt`, extracted and never loaded), so `cl example.c /I<natives dir> <natives
dir>\jllama.lib` links the entry points directly; `LoadLibrary` + `GetProcAddress` works as well.
Streaming:

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
  grammar conversion, and a quantisation of the random model to Q8_0 that loads again; the
  embedded server started in router mode and stopped, the frontend attached to the random model
  and the model still answering after the server stopped, the RPC server serving the CPU over
  loopback and stopped, text to speech refusing missing models, and a fine-tuning run over the
  random F32 model that writes a GGUF.
- `src/test/c/test_c_abi.c` (the ctest `c_abi` of every modular build job): plain C11 linking the
  shipped `libjllama` and `libllama`; it compiles the header as C and resolves `jllama_api_version`
  and `jllama_build_info` through the library's exports, next to its layer-A checks.
- `.github/verify-native-deps.py` (`package`): `libjllama` exports `JNI_OnLoad`, `JNI_OnUnload`,
  `Java_*` and `jllama_*`, nothing else.
- The Java suite: every native method of the Java API calls these functions (no JNI file holds
  engine logic any more), so every model-backed Java test runs through the C API -- including
  the ones the random model cannot exercise here (embeddings, rerank, infill, a decision model,
  real speech, the router with workers), which CI runs against the real models.

## Implementation notes

`src/main/cpp/jllama_engine.{hpp,cpp}` is the implementation both shells share for the model: C++
over `server_context`, exceptions for errors (`jllama_invalid_request` for the request's fault,
`jllama_interrupted` for a wait cut by a free, `std::runtime_error` otherwise), no JNI in it. The
embedded server has its own engine (`native_server_engine.{hpp,cpp}`), as the RPC support
(`rpc_support.hpp`), text to speech (`tts_engine.{h,cpp}`) and the trainer (`train_engine.{h,cpp}`)
already had; `jllama_api.cpp` is the one C shell over all of them.
`jllama_api.cpp` turns that into status codes, a thread-local error string and malloc'd strings, and
runs every call inside the user-reference guard (`jllama_context.hpp`) so `jllama_model_free` can
wait for the calls in flight -- the same protocol `LlamaModel.close()` relies on. The inference loop
is a plain thread: it attaches to nothing, so the same loop serves a C caller without a JVM.
