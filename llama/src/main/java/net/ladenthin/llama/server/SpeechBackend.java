// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import org.jspecify.annotations.Nullable;

/**
 * What {@link OpenAiCompatServer} needs to answer {@code POST /v1/audio/speech} and {@code POST /tts}: text in, a WAV file
 * out. The seam between the HTTP route and the speech engine, so the route is tested without a
 * native library or a model; {@link net.ladenthin.llama.TextToSpeech} is the engine behind it in
 * {@link OpenAiCompatServer#OpenAiCompatServer(net.ladenthin.llama.LlamaModel, OpenAiServerConfig,
 * net.ladenthin.llama.TextToSpeech)}.
 */
@FunctionalInterface
public interface SpeechBackend {

    /**
     * Speak a text.
     *
     * @param text the text to speak
     * @param voice a predefined voice of the model by name, or {@code null} for its default
     * @param language the language of the text as the model names it, or {@code null} for its default
     * @return a WAV file: 16-bit PCM, mono, at the model's sample rate
     * @throws IOException if the synthesis fails
     * @throws IllegalArgumentException or {@link net.ladenthin.llama.exception.InvalidRequestException}
     *     if the request is at fault (an unknown voice), which the route answers with {@code 400}
     */
    byte[] synthesize(String text, @Nullable String voice, @Nullable String language) throws IOException;

    /**
     * Speak a text with the options of llama.cpp's {@code POST /tts} (a speaker reference, the frame
     * cap, the sampling). The default ignores the options and calls
     * {@link #synthesize(String, String, String)}; an engine that can apply them overrides it.
     *
     * @param text the text to speak
     * @param voice a predefined voice of the model by name, or {@code null} for its default
     * @param language the language of the text as the model names it, or {@code null} for its default
     * @param options what else the request set
     * @return a WAV file: 16-bit PCM, mono, at the model's sample rate
     * @throws IOException if the synthesis fails
     */
    default byte[] synthesize(String text, @Nullable String voice, @Nullable String language, SpeechOptions options)
            throws IOException {
        return synthesize(text, voice, language);
    }

    /**
     * The predefined voices of the model, by the names {@link #synthesize} takes.
     *
     * @return the voice names; empty for a model without predefined voices
     */
    default List<String> voices() {
        return Collections.emptyList();
    }
}
