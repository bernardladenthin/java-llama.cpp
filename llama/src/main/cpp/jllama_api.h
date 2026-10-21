/*
 * SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * jllama_api.h -- the C API of libjllama (layer B of the delivery roadmap, TODO.md item 7).
 *
 * Everything LlamaModel can do, callable without a JVM: from C, C++, Rust, Python (ctypes/cffi),
 * or Java 22+ through the FFM API. One implementation serves both call paths -- the Java_* JNI
 * entry points of this library are a thin shell over these functions -- so a behaviour tested here
 * is the behaviour Java sees. The library next to this header is the natives directory
 * (CLAUDE.md "Natives jars"): `jllama` imports upstream's own `llama`, `mtmd` and `ggml` shared
 * libraries from the same directory, whose headers (llama.h, mtmd.h, ggml.h) are upstream's layer A
 * and stay unchanged; this header is the project's own layer on top (chat completions over
 * llama.cpp's server core, streaming, the log sink, and -- slice by slice -- the rest).
 *
 * Conventions:
 *   - Opaque handle: jllama_model, created by jllama_model_load, released by jllama_model_free.
 *   - Requests and responses are JSON text, the same bodies llama.cpp's own HTTP server reads and
 *     writes (an OpenAI chat-completion request in, an OpenAI chat-completion object out).
 *   - Every string the library returns through an out-parameter is owned by the caller and is
 *     released with jllama_string_free. Strings the caller passes are read during the call only.
 *   - Every function returns a jllama_status; on anything but JLLAMA_STATUS_OK, jllama_last_error()
 *     holds the message of the failure (thread-local; valid until the next call on this thread).
 *   - A model handle may be used from several threads; jllama_model_free waits for the calls in
 *     flight on it and must not race a new call (the caller owns the handle's lifetime).
 *   - The surface grows per release and is versioned with the library: jllama_api_version() is
 *     the net.ladenthin:llama version this library was built as (x.y.z or x.y.z-SNAPSHOT); a
 *     function present in version x.y.z stays, with its semantics, in every later x.y.*.
 */

#ifndef JLLAMA_API_H
#define JLLAMA_API_H

#include <stdbool.h>
#include <stddef.h>

#if defined(_WIN32) && !defined(__MINGW32__)
#if defined(JLLAMA_BUILD)
#define JLLAMA_API __declspec(dllexport)
#else
#define JLLAMA_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define JLLAMA_API __attribute__((visibility("default")))
#else
#define JLLAMA_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** A loaded model with its inference loop; opaque. */
typedef struct jllama_model jllama_model;

/** The outcome of every call. */
typedef enum jllama_status {
    /** Done. */
    JLLAMA_STATUS_OK = 0,
    /** A server-side failure (a model that does not load, a failed task, ...): jllama_last_error(). */
    JLLAMA_STATUS_ERROR = 1,
    /** The request is the caller's fault: what llama.cpp's HTTP server answers with 400. */
    JLLAMA_STATUS_INVALID_REQUEST = 2,
    /** A null handle or out-parameter where one is required. */
    JLLAMA_STATUS_INVALID_ARGUMENT = 3,
    /** A wait was cut short because the model is being freed. */
    JLLAMA_STATUS_INTERRUPTED = 4
} jllama_status;

/** Log levels, the values of ggml_log_level (so a level can be passed on to ggml's own callback). */
typedef enum jllama_log_level {
    JLLAMA_LOG_LEVEL_NONE = 0,
    JLLAMA_LOG_LEVEL_DEBUG = 1,
    JLLAMA_LOG_LEVEL_INFO = 2,
    JLLAMA_LOG_LEVEL_WARN = 3,
    JLLAMA_LOG_LEVEL_ERROR = 4,
    /** A continuation of the previous line. */
    JLLAMA_LOG_LEVEL_CONT = 5
} jllama_log_level;

/**
 * Load progress, 0.0 to 1.0, called on the loading thread; return false to abort the load (the
 * load then fails with JLLAMA_STATUS_ERROR).
 */
typedef bool (*jllama_load_progress_fn)(float progress, void *user_data);

/**
 * A log line: `level` is a jllama_log_level, `text` the formatted message (no prefix, no
 * timestamp, no colours; the final newline included where the source wrote one). Called on the
 * logger's own worker thread, never on the thread that logged.
 */
typedef void (*jllama_log_fn)(int level, const char *text, void *user_data);

/** The net.ladenthin:llama version this library was built as, e.g. "6.0.0" (static string). */
JLLAMA_API const char *jllama_api_version(void);

/** The llama.cpp build inside, "b<number>-<commit>" (static string), as LlamaModel.getLlamaCppBuildInfo(). */
JLLAMA_API const char *jllama_build_info(void);

/** The message of the last failed call on this thread, or "" (thread-local; do not free). */
JLLAMA_API const char *jllama_last_error(void);

/** Releases a string the library returned through an out-parameter; null is ignored. */
JLLAMA_API void jllama_string_free(char *s);

/**
 * Loads a model from llama-server's command line: argv[0] is a program name (any), the rest the
 * flags llama-server accepts (`-m <gguf>`, `-c`, `-ngl`, ...; `--vocab-only` loads the tokenizer
 * only). `progress` may be null. On JLLAMA_STATUS_OK, *out holds the handle.
 */
JLLAMA_API jllama_status jllama_model_load(int argc, const char *const *argv, jllama_load_progress_fn progress,
                                           void *progress_user_data, jllama_model **out);

/**
 * Frees a model: cancels its streaming tasks, waits for the calls in flight on it, stops its
 * inference loop and releases the weights. Null is ignored.
 */
JLLAMA_API void jllama_model_free(jllama_model *model);

/**
 * An OpenAI chat completion, blocking: `request_json` is the request body
 * (`{"messages":[...], "max_tokens": ..., ...}`), *response_json the chat.completion object
 * (jllama_string_free). JLLAMA_STATUS_INVALID_REQUEST for a body the server refuses, including
 * "n" > 1: a request takes one completion (send separate requests).
 */
JLLAMA_API jllama_status jllama_chat_completions(jllama_model *model, const char *request_json, char **response_json);

/**
 * Starts a streaming OpenAI chat completion (`"stream": true` in the body makes the server emit
 * chat.completion.chunk deltas) and returns its task id in *task_id. Receive with
 * jllama_chat_completions_stream_receive until it reports stop; a task not received to its end is
 * released with jllama_task_release. "n" > 1 is JLLAMA_STATUS_INVALID_REQUEST, as for
 * jllama_chat_completions.
 */
JLLAMA_API jllama_status jllama_chat_completions_stream_request(jllama_model *model, const char *request_json,
                                                                int *task_id);

/**
 * The next event of a streaming task (a chat one, or one from jllama_completions_oai_stream_request),
 * blocking until one is available: *envelope_json is
 * `{"data": <chunk object, or the array of final chunks>, "stop": <bool>}` (jllama_string_free),
 * *stop is the same flag. After stop the task is released. JLLAMA_STATUS_ERROR with "Task not
 * found" for an unknown or already released id.
 */
JLLAMA_API jllama_status jllama_chat_completions_stream_receive(jllama_model *model, int task_id, char **envelope_json,
                                                                bool *stop);

/** Cancels or releases a streaming task; an unknown id is ignored. */
JLLAMA_API jllama_status jllama_task_release(jllama_model *model, int task_id);

/**
 * Routes every log line of the library (llama.cpp, ggml, the server) to `callback` instead of the
 * console; null restores the console and first flushes the lines queued for the previous callback
 * (synchronously, so a callback's user_data may be released after this returns). Process-wide.
 */
JLLAMA_API jllama_status jllama_set_log_callback(jllama_log_fn callback, void *user_data);

/* ------------------------------------------------------------------------------------------------
 * The rest of LlamaModel. Request and response bodies are the ones llama.cpp's HTTP server reads
 * and writes for the route of the same name; every out-string is released with jllama_string_free,
 * every out-array with the free function named next to it.
 * ---------------------------------------------------------------------------------------------- */

/** Releases a float array the library returned (jllama_embed); null is ignored. */
JLLAMA_API void jllama_floats_free(float *values);

/** Releases a token array the library returned (jllama_tokenize); null is ignored. */
JLLAMA_API void jllama_tokens_free(int *tokens);

/**
 * The model's metadata object (LlamaModel.getModelMeta()): vocab type and size, context length,
 * embedding size, parameter count, file size, modalities, name, architecture, chat template, the
 * special tokens and the full GGUF key/value map.
 */
JLLAMA_API jllama_status jllama_model_meta(jllama_model *model, char **response_json);

/**
 * A raw completion (llama.cpp's /completion body: `{"prompt": ..., "n_predict": ..., ...}`),
 * blocking; *response_json is the result object. A request takes one prompt and one completion:
 * several prompts or "n" > 1 are JLLAMA_STATUS_INVALID_REQUEST (send separate requests). A body with
 * `input_prefix`/`input_suffix` is an infill one, prepared as for jllama_infill (both fields required,
 * JLLAMA_STATUS_ERROR when the model has no FIM tokens). One with `messages` completes the prompt the
 * chat template renders from them (as jllama_apply_template); a non-empty `prompt` or the infill
 * fields beside them are JLLAMA_STATUS_INVALID_REQUEST.
 */
JLLAMA_API jllama_status jllama_completions(jllama_model *model, const char *request_json, char **response_json);

/**
 * An OpenAI /v1/completions request, blocking; *response_json the text_completion object. One prompt
 * and one choice, as for jllama_completions.
 */
JLLAMA_API jllama_status jllama_completions_oai(jllama_model *model, const char *request_json, char **response_json);

/**
 * Starts a streaming OpenAI /v1/completions request (the body as for jllama_completions_oai) and
 * returns its task id in *task_id. Receive with jllama_chat_completions_stream_receive, whose
 * envelopes then carry text_completion chunks, the last with usage; release an unfinished task with
 * jllama_task_release. One prompt and one choice, as for jllama_completions.
 */
JLLAMA_API jllama_status jllama_completions_oai_stream_request(jllama_model *model, const char *request_json,
                                                               int *task_id);

/**
 * A fill-in-the-middle request (`input_prefix`, `input_suffix`, optional `input_extra` and
 * `prompt`), blocking. JLLAMA_STATUS_ERROR when the model has no FIM tokens; "n" > 1 is
 * JLLAMA_STATUS_INVALID_REQUEST, as for jllama_completions.
 */
JLLAMA_API jllama_status jllama_infill(jllama_model *model, const char *request_json, char **response_json);

/**
 * Starts a raw streaming completion (`"stream": true`; an infill one when the body carries
 * `input_prefix`/`input_suffix`, a templated one when it carries `messages`, both prepared as for
 * jllama_completions) and returns its task id; receive with
 * jllama_completions_stream_receive. One prompt and one completion, as for jllama_completions.
 */
JLLAMA_API jllama_status jllama_completions_stream_request(jllama_model *model, const char *request_json, int *task_id);

/**
 * Starts a streaming chat completion whose results are llama.cpp's raw result objects (the chat
 * template applied, no OpenAI chunk formatting); receive with jllama_completions_stream_receive.
 * "n" > 1 is JLLAMA_STATUS_INVALID_REQUEST, as for jllama_chat_completions.
 */
JLLAMA_API jllama_status jllama_chat_completions_request(jllama_model *model, const char *request_json, int *task_id);

/**
 * The next result of a raw streaming task, blocking until one is available: the result object
 * with a `"stop"` key added; *stop is the same flag, after which the task is released.
 * JLLAMA_STATUS_ERROR with "Task not found" for an unknown or already released id.
 */
JLLAMA_API jllama_status jllama_completions_stream_receive(jllama_model *model, int task_id, char **response_json,
                                                           bool *stop);

/** The chat template applied to an OpenAI chat body (`{"messages": [...]}`): the prompt text. */
JLLAMA_API jllama_status jllama_apply_template(jllama_model *model, const char *request_json, char **prompt);

/**
 * The embedding of one prompt (a model loaded with `--embedding`): *values holds *count floats
 * (jllama_floats_free).
 */
JLLAMA_API jllama_status jllama_embed(jllama_model *model, const char *prompt, float **values, size_t *count);

/**
 * An /embeddings request (`{"content": ...}` or `{"input": ...}`), blocking; with oai_compat the
 * OpenAI /v1/embeddings response shape (`encoding_format` honoured), else the raw result array.
 */
JLLAMA_API jllama_status jllama_embeddings(jllama_model *model, const char *request_json, bool oai_compat,
                                           char **response_json);

/**
 * Reranks `n_documents` documents against `query` (a model loaded with `--reranking`); the result
 * array, one entry per document with `document` (its text), `index` (its position in `documents`),
 * `score` and `tokens_evaluated` (the tokens the model evaluated for it, 0 if not reported; an
 * OpenAI-shaped `usage` sums them).
 */
JLLAMA_API jllama_status jllama_rerank(jllama_model *model, const char *query, int n_documents,
                                       const char *const *documents, char **response_json);

/** Tokenises `text`: *tokens holds *count ids (jllama_tokens_free). */
JLLAMA_API jllama_status jllama_tokenize(jllama_model *model, const char *text, bool add_special, int **tokens,
                                         size_t *count);

/** The /tokenize response for `text` (`{"tokens": [...]}`, ids or `{id, piece}` objects). */
JLLAMA_API jllama_status jllama_tokenize_json(jllama_model *model, const char *text, bool add_special, bool with_pieces,
                                              char **response_json);

/**
 * Detokenises `count` ids: *text holds the UTF-8 bytes with *length (jllama_string_free; the bytes
 * may include NUL, which is why the length is returned).
 */
JLLAMA_API jllama_status jllama_detokenize(jllama_model *model, const int *tokens, size_t count, char **text,
                                           size_t *length);

/** The /detokenize response for `count` ids (`{"content": "..."}`). */
JLLAMA_API jllama_status jllama_detokenize_json(jllama_model *model, const int *tokens, size_t count,
                                                char **response_json);

/** The server metrics object (LlamaModel.getMetrics()): the counters plus the `slots` array. */
JLLAMA_API jllama_status jllama_metrics(jllama_model *model, char **response_json);

typedef enum jllama_slot_op {
    /** Saves a slot's KV cache to `filename` (under `--slot-save-path`). */
    JLLAMA_SLOT_SAVE = 1,
    /** Restores a slot's KV cache from `filename`. */
    JLLAMA_SLOT_RESTORE = 2,
    /** Erases a slot's KV cache (`filename` ignored). */
    JLLAMA_SLOT_ERASE = 3
} jllama_slot_op;

/** A slot save/restore/erase; *response_json is the server's result object. */
JLLAMA_API jllama_status jllama_slot_action(jllama_model *model, jllama_slot_op action, int slot_id,
                                            const char *filename, char **response_json);

/** The LoRA adapters of the model (the GET /lora-adapters array). */
JLLAMA_API jllama_status jllama_lora_adapters(jllama_model *model, char **response_json);

/** Sets the LoRA adapter scales from a JSON array of `{"id": n, "scale": f}`; the resulting list. */
JLLAMA_API jllama_status jllama_set_lora_adapters(jllama_model *model, const char *adapters_json, char **response_json);

/**
 * Runtime tuning from a JSON object with any of `n_threads`, `n_threads_batch` (positive
 * integers) and `slot_prompt_similarity` (0.0 to 1.0).
 */
JLLAMA_API jllama_status jllama_configure(jllama_model *model, const char *config_json);

/** A /v1/systemone request (a decision model); *response_json is the handler's answer body. */
JLLAMA_API jllama_status jllama_systemone(jllama_model *model, const char *request_json, char **response_json);

/** A JSON schema converted to a GBNF grammar (needs no model). */
JLLAMA_API jllama_status jllama_json_schema_to_grammar(const char *schema_json, char **grammar);

/**
 * Quantises a GGUF file to `ftype` (a llama_ftype value) with `nthread` threads (0 = all);
 * `allow_requantize` permits a non-f32/f16 input. Needs no model; initialises the backend.
 */
JLLAMA_API jllama_status jllama_quantize(const char *input_path, const char *output_path, int ftype, int nthread,
                                         bool allow_requantize);

/* ------------------------------------------------------------------------------------------------
 * The embedded HTTP server (NativeServer): upstream's full llama-server, WebUI included, on a
 * worker thread inside this library. Not available on Android (server.cpp does not build there);
 * these functions are absent from the Android library. ONE server per process.
 * ---------------------------------------------------------------------------------------------- */

/** A running embedded server; opaque. */
typedef struct jllama_server jllama_server;

/**
 * Starts the server from llama-server's command line (argv[0] is a program name, any; the rest
 * every llama-server flag, including router mode with `--models-dir`). It loads its own model and
 * returns as soon as the worker thread runs; a model that fails to load ends the worker, which
 * jllama_server_is_running reports.
 */
JLLAMA_API jllama_status jllama_server_start(int argc, const char *const *argv, jllama_server **out);

/**
 * Starts the HTTP frontend over an already-loaded model (attach mode): one copy of the weights,
 * the model's own inference loop serves the routes; the argv configures the HTTP side (`--host`,
 * `--port`, `--api-key`, ...). The server holds the model until it is stopped: stop the server
 * before freeing the model. JLLAMA_STATUS_ERROR for a vocab-only model.
 */
JLLAMA_API jllama_status jllama_server_start_attached(jllama_model *model, int argc, const char *const *argv,
                                                      jllama_server **out);

/** Stops the server (blocks until its worker returned) and releases the handle; null is ignored. */
JLLAMA_API jllama_status jllama_server_stop(jllama_server *server);

/** True while the server's worker runs; false for null and after it returned on its own. */
JLLAMA_API bool jllama_server_is_running(const jllama_server *server);

/**
 * The command router mode spawns its model workers with (LLAMA_SERVER_WORKER_CMD, whitespace
 * split, replaces the binary-path token; patches/0008), e.g. `java -cp app.jar
 * net.ladenthin.llama.server.NativeServer`; "" clears it. Process-wide.
 */
JLLAMA_API jllama_status jllama_server_set_worker_command(const char *command);

/* ------------------------------------------------------------------------------------------------
 * The RPC server (RpcServer): this process's devices served to llama.cpp RPC clients, ggml's own
 * loop (patches/0015 makes it stoppable). ONE server per process; no authentication, no
 * encryption -- bind to loopback unless you mean it.
 * ---------------------------------------------------------------------------------------------- */

/** Releases an array of strings the library returned (jllama_rpc_server_devices); null is ignored. */
JLLAMA_API void jllama_strings_free(char **strings, size_t count);

/**
 * Serves `devices` (comma-separated device names such as "CPU" or "CUDA0"; "" = every accelerator,
 * else the CPU, never an RPC device) on host:port, blocking until jllama_rpc_server_stop() is
 * called from another thread -- or returning at once when the socket cannot be bound, which
 * jllama_rpc_server_listening() tells apart. `host` is an IPv4 literal; `cache_dir` may be null.
 * JLLAMA_STATUS_ERROR for an unknown device name or no device to serve.
 */
JLLAMA_API jllama_status jllama_rpc_server_serve(const char *host, int port, int threads, const char *cache_dir,
                                                 const char *devices);

/**
 * Stops the RPC server and returns once jllama_rpc_server_serve() has returned -- also for a stop
 * that arrives before the socket is bound (device initialization), which ggml alone would lose.
 * JLLAMA_STATUS_ERROR when serve has not returned 30 s after the request. Called on the serving
 * thread itself (from a callback that runs there before the server starts), it returns at once and
 * serve returns without starting the server.
 *
 * A no-op when no serve call is in progress, and it does not carry over to the next one: a stop
 * issued before the serving thread has entered jllama_rpc_server_serve() is lost, so a caller that
 * stops a server that may not have come up yet repeats the stop until serve has returned, the way
 * the Java RpcServer does.
 */
JLLAMA_API jllama_status jllama_rpc_server_stop(void);

/** True while the RPC server listens. */
JLLAMA_API bool jllama_rpc_server_listening(void);

/**
 * The names of the devices `devices` selects (the same rule as jllama_rpc_server_serve), as
 * *count strings in *names (jllama_strings_free). JLLAMA_STATUS_ERROR for an unknown name, with
 * the available ones in the message.
 */
JLLAMA_API jllama_status jllama_rpc_server_devices(const char *devices, char ***names, size_t *count);

/* ------------------------------------------------------------------------------------------------
 * Text to speech (TextToSpeech): llama.cpp's Qwen3-TTS pipeline over a backbone GGUF and an mmproj
 * GGUF (speaker encoder, code predictor, code2wav decoder).
 * ---------------------------------------------------------------------------------------------- */

/** A loaded text-to-speech model pair; opaque. */
typedef struct jllama_tts_model jllama_tts_model;

/** Releases a byte buffer the library returned (jllama_tts_synthesize); null is ignored. */
JLLAMA_API void jllama_bytes_free(unsigned char *bytes);

/** Loads the backbone and the mmproj (`n_gpu_layers` -1 = auto, `n_threads` <= 0 = default). */
JLLAMA_API jllama_status jllama_tts_load(const char *model_path, const char *mmproj_path, int n_gpu_layers,
                                         int n_threads, jllama_tts_model **out);

/**
 * Synthesises `text` to a 24 kHz mono 16-bit WAV in *wav (*length bytes, jllama_bytes_free).
 * `speaker_reference_path` (a WAV to clone the voice from) and `lang` ("english", "chinese", ...)
 * may be null; `max_frames` bounds the generation, `top_k` and `seed` the sampling. One synthesis
 * at a time per handle.
 */
JLLAMA_API jllama_status jllama_tts_synthesize(jllama_tts_model *tts, const char *text,
                                               const char *speaker_reference_path, const char *lang, int max_frames,
                                               int top_k, unsigned int seed, unsigned char **wav, size_t *length);

/** Frees the model pair; null is ignored. */
JLLAMA_API void jllama_tts_free(jllama_tts_model *tts);

/* ------------------------------------------------------------------------------------------------
 * Fine-tuning (LlamaTrainer): llama.cpp's ggml-opt training path over a GGUF and a text corpus.
 * ---------------------------------------------------------------------------------------------- */

/**
 * Runs one fine-tuning job from its JSON configuration (`model_path`, `training_text` or
 * `training_file`, `output_path`, `epochs`, `learning_rate`, `lr_min`, `decay_epochs`,
 * `weight_decay`, `optimizer`, `n_ctx`, `n_gpu_layers`, `val_split`, `n_batch`, `n_ubatch`; the
 * keys of LlamaTrainer's TrainingParameters) and writes the fine-tuned GGUF. Blocks for the whole
 * run; needs no model handle. A configuration that does not parse is "invalid training config".
 */
JLLAMA_API jllama_status jllama_finetune(const char *config_json);

#ifdef __cplusplus
}
#endif

#endif /* JLLAMA_API_H */
