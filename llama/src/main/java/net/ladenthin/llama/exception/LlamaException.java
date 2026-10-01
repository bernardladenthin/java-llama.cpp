// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.exception;

/**
 * Base unchecked exception raised by the JNI layer when a llama.cpp operation
 * fails. Specific failure modes may extend this class with typed subclasses
 * (e.g. {@link ModelUnavailableException}).
 *
 * <p>This was historically package-private; it was promoted to {@code public}
 * to allow external callers to {@code catch} the typed subclasses by their
 * common base. Existing callers that caught {@link RuntimeException} continue
 * to work unchanged.</p>
 */
public class LlamaException extends RuntimeException {

    /**
     * Creates a new {@link LlamaException} with the given message.
     *
     * @param message the detail message; may be {@code null}
     */
    public LlamaException(String message) {
        super(message);
    }

    /**
     * Creates a new {@link LlamaException} with the given message and cause.
     *
     * @param message the detail message; may be {@code null}
     * @param cause   the underlying cause; may be {@code null}
     */
    public LlamaException(String message, Throwable cause) {
        super(message, cause);
    }
}
