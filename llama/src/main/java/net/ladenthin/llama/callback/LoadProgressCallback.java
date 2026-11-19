// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.callback;

import net.ladenthin.llama.parameters.ModelParameters;

/**
 * Receives model-load progress updates from the native loader.
 * <p>
 * Pass an instance to {@link net.ladenthin.llama.LlamaModel#LlamaModel(ModelParameters, LoadProgressCallback)}
 * to observe the {@code llama_model_params.progress_callback} hook from llama.cpp. The
 * callback is invoked synchronously on the loader thread (the same thread that called
 * the constructor) with a value in {@code [0.0, 1.0]}.
 * </p>
 * <p>
 * Return {@code false} to abort the load. When {@code false} is returned, the constructor
 * throws {@link net.ladenthin.llama.exception.LlamaException} because the native loader aborts and reports failure.
 * </p>
 * <p>
 * A {@link RuntimeException} (or {@link Error}) thrown by the callback aborts the load as well, and
 * the constructor rethrows it unchanged; the callback is not called again. ({@code llama-ffm}'s
 * {@code FfmLlamaModel.load} reports it as a {@code LlamaException} instead: an exception cannot cross
 * an FFM upcall.)
 * </p>
 */
@FunctionalInterface
public interface LoadProgressCallback {

    /**
     * Receive a progress update.
     *
     * @param progress fraction in {@code [0.0, 1.0]}
     * @return {@code true} to continue loading, {@code false} to abort
     */
    boolean onProgress(float progress);
}
