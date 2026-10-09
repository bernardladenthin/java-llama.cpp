// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.exception;

/**
 * Thrown by the JNI layer when a request is rejected before or while it is validated &#x2014; the
 * cases llama.cpp's own HTTP server answers with status {@code 400} ({@code invalid_request_error}):
 * a body that is not valid JSON, a missing or malformed field ({@code "prompt"}, {@code "messages"},
 * {@code "input_prefix"}, &hellip;), an empty embedding input, a parameter outside its hard limits,
 * or a prompt that exceeds the context size. The message is the bare reason, as upstream puts it
 * into {@code error.message}.
 *
 * <p>Everything else the native layer reports &#x2014; a failed load, an operation the model was
 * not loaded for, an inference failure &#x2014; stays a plain {@link LlamaException}, which the
 * OpenAI-compatible server ({@code net.ladenthin.llama.server.OpenAiCompatServer}) answers with
 * {@code 500}. This subclass is what lets it answer {@code 400} instead, the way upstream does,
 * without matching on message text.</p>
 */
public class InvalidRequestException extends LlamaException {

    /**
     * Creates a new {@link InvalidRequestException} with the given message.
     *
     * @param message the detail message; may be {@code null}
     */
    public InvalidRequestException(String message) {
        super(message);
    }

    /**
     * Creates a new {@link InvalidRequestException} with the given message and cause.
     *
     * @param message the detail message; may be {@code null}
     * @param cause   the underlying cause; may be {@code null}
     */
    public InvalidRequestException(String message, Throwable cause) {
        super(message, cause);
    }
}
