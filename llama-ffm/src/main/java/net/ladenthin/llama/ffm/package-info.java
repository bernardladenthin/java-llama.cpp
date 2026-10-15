// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

/**
 * java-llama.cpp without JNI: the C API of {@code libjllama} ({@code jllama_api.h}, shipped in every
 * natives directory) called through the Java 22+ Foreign Function and Memory API.
 *
 * <ul>
 *   <li>{@link net.ladenthin.llama.ffm.FfmLlamaModel} &mdash; a loaded model: chat and raw completions,
 *       streaming by task id, embeddings, rerank, tokenisation, metrics, slots, LoRA, runtime
 *       configuration</li>
 *   <li>{@link net.ladenthin.llama.ffm.FfmNativeServer} &mdash; the embedded llama-server, classic or
 *       attached to a model</li>
 *   <li>{@link net.ladenthin.llama.ffm.FfmRpcServer} &mdash; the in-process ggml RPC server</li>
 *   <li>{@link net.ladenthin.llama.ffm.FfmTextToSpeech} &mdash; Qwen3-TTS synthesis</li>
 *   <li>{@link net.ladenthin.llama.ffm.Jllama} &mdash; the model-free calls: version and build info, the
 *       log callback, grammar conversion, quantisation, fine-tuning</li>
 * </ul>
 *
 * <p>The natives are the same jars as for {@code net.ladenthin:llama}, found and loaded by the core's
 * loader, {@code LlamaLoader} (classpath jars, {@code net.ladenthin.llama.lib.path}
 * or {@code java.library.path}, with the same build check); this module only resolves the
 * {@code jllama_*} symbols the loaded library exports. Requests and responses are JSON text, the
 * bodies llama.cpp's own HTTP server reads and writes; {@code net.ladenthin.llama.parameters.ModelParameters}
 * renders a model's argv and {@code InferenceParameters.toJson()} a request body, exactly as for the
 * core. Failures are the core's exceptions: {@link net.ladenthin.llama.exception.InvalidRequestException}
 * where llama.cpp answers 400, {@link net.ladenthin.llama.exception.LlamaException} otherwise.
 *
 * <p>The downcalls are restricted methods of the FFM API: run the JVM with
 * {@code --enable-native-access=ALL-UNNAMED} (or name the module) to silence the warning a Java 22+
 * JVM prints otherwise. Not available on Android, which has no FFM API.
 */
package net.ladenthin.llama.ffm;
