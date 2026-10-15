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

/** The net.ladenthin:llama version this library was built as, e.g. "5.2.0" (static string). */
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
 * (jllama_string_free). JLLAMA_STATUS_INVALID_REQUEST for a body the server refuses.
 */
JLLAMA_API jllama_status jllama_chat_completions(jllama_model *model, const char *request_json, char **response_json);

/**
 * Starts a streaming OpenAI chat completion (`"stream": true` in the body makes the server emit
 * chat.completion.chunk deltas) and returns its task id in *task_id. Receive with
 * jllama_chat_completions_stream_receive until it reports stop; a task not received to its end is
 * released with jllama_task_release.
 */
JLLAMA_API jllama_status jllama_chat_completions_stream_request(jllama_model *model, const char *request_json,
                                                                int *task_id);

/**
 * The next event of a streaming task, blocking until one is available: *envelope_json is
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
 * blocking; *response_json is the array of result objects.
 */
JLLAMA_API jllama_status jllama_completions(jllama_model *model, const char *request_json, char **response_json);

/** An OpenAI /v1/completions request, blocking; *response_json the text_completion object(s). */
JLLAMA_API jllama_status jllama_completions_oai(jllama_model *model, const char *request_json, char **response_json);

/**
 * A fill-in-the-middle request (`input_prefix`, `input_suffix`, optional `input_extra` and
 * `prompt`), blocking. JLLAMA_STATUS_ERROR when the model has no FIM tokens.
 */
JLLAMA_API jllama_status jllama_infill(jllama_model *model, const char *request_json, char **response_json);

/**
 * Starts a raw streaming completion (`"stream": true`; an infill one when the body carries
 * `input_prefix`/`input_suffix`) and returns its task id; receive with
 * jllama_completions_stream_receive.
 */
JLLAMA_API jllama_status jllama_completions_stream_request(jllama_model *model, const char *request_json, int *task_id);

/**
 * Starts a streaming chat completion whose results are llama.cpp's raw result objects (the chat
 * template applied, no OpenAI chunk formatting); receive with jllama_completions_stream_receive.
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
 * array, each entry with its `index` and `score`.
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

#ifdef __cplusplus
}
#endif

#endif /* JLLAMA_API_H */
