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

This is the first slice (the roadmap's "B.2 spike", TODO.md). It covers:

| Function | What it does |
|---|---|
| `jllama_api_version()` | the `net.ladenthin:llama` version this library was built as, e.g. `5.2.0`; the C API is versioned with the library (a function present in `x.y.z` keeps its semantics in every later `x.y.*`) |
| `jllama_build_info()` | the llama.cpp build inside, `b<number>-<commit>` |
| `jllama_last_error()` | the message of the last failed call on this thread (thread-local) |
| `jllama_string_free()` | releases a string the library returned |
| `jllama_model_load()` / `jllama_model_free()` | a model from llama-server's command line (`-m`, `-c`, `-ngl`, ... and `--vocab-only`), with an optional load-progress callback |
| `jllama_chat_completions()` | one OpenAI chat completion, blocking |
| `jllama_chat_completions_stream_request()` / `_receive()` / `jllama_task_release()` | a streamed one: a task id, then one `{"data": <chunk>, "stop": <bool>}` envelope per call until `stop` |
| `jllama_set_log_callback()` | every log line of llama.cpp, ggml and the server to a callback; null restores the console after a synchronous flush |

Not yet in it (the roadmap's "B.3", engine by engine): completions, infill, embeddings, rerank,
tokenize/detokenize, metrics and slots, LoRA, the embedded HTTP server (`NativeServer`), the RPC
server, text to speech, the quantizer, the trainer. Their Java entry points still carry their own
glue; each slice moves one of them behind a `jllama_*` function.

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
  status codes of a malformed body, a refused body, a missing model and null arguments pinned.
- `src/test/c/test_c_abi.c` (the ctest `c_abi` of every modular build job): plain C11 linking the
  shipped `libjllama` and `libllama`; it compiles the header as C and resolves `jllama_api_version`
  and `jllama_build_info` through the library's exports, next to its layer-A checks.
- `.github/verify-native-deps.py` (`package`): `libjllama` exports `JNI_OnLoad`, `JNI_OnUnload`,
  `Java_*` and `jllama_*`, nothing else.
- The Java suite: `handleChatCompletions`, `requestChatCompletionStream`,
  `receiveChatCompletionChunk`, `releaseTask`, `cancelCompletion`, `loadModel`, `close` and
  `setLogger` call these functions, so every model-backed Java test runs through the C API.

## Implementation notes

`src/main/cpp/jllama_engine.{hpp,cpp}` is the implementation both shells share: C++ over
`server_context`, exceptions for errors (`jllama_invalid_request` for the request's fault,
`jllama_interrupted` for a wait cut by a free, `std::runtime_error` otherwise), no JNI in it.
`jllama_api.cpp` turns that into status codes, a thread-local error string and malloc'd strings, and
runs every call inside the user-reference guard (`jllama_context.hpp`) so `jllama_model_free` can
wait for the calls in flight -- the same protocol `LlamaModel.close()` relies on. The inference loop
is a plain thread: it attaches to nothing, so the same loop serves a C caller without a JVM.
