// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

/**
 * Channel of a prefilled assistant message that the model continues from
 * when {@code continue_final_message} is used on the chat completions endpoint.
 *
 * <p>Maps to the string-valued branch of llama.cpp's
 * {@code common_chat_continuation_parse}. The boolean form
 * ({@code true}/{@code false}) is exposed separately via
 * {@code InferenceParameters.setContinueFinalMessage(boolean)}.
 */
public enum ContinuationMode {

    /** Continue inside the reasoning channel of the last assistant message. */
    REASONING_CONTENT("reasoning_content"),

    /** Continue inside the content channel of the last assistant message. */
    CONTENT("content");

    private final String value;

    ContinuationMode(String value) {
        this.value = value;
    }

    public String getValue() {
        return value;
    }
}
