// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import org.jspecify.annotations.Nullable;

/**
 * Reads the media type of a base64-encoded image from its encoded magic bytes, for the protocol
 * translations that receive an image without one (Ollama's bare base64 {@code images}, an Anthropic
 * {@code base64} source without {@code media_type}). llama.cpp decodes an image by its content and
 * requires only the {@code data:image/...;base64,} shape of the URL, so the type is informational; each
 * caller picks its own fallback for bytes this does not recognise. The first three bytes of an image
 * fix the first four base64 characters, so the encoded prefix is compared rather than decoded.
 */
final class ImageMediaTypes {

    private ImageMediaTypes() {}

    /**
     * The media type of the base64-encoded image, read from its magic bytes.
     *
     * @param base64 the encoded image
     * @return {@code image/png}, {@code image/jpeg}, {@code image/gif} or {@code image/webp}, or
     *     {@code null} for bytes of another format
     */
    static @Nullable String ofBase64(String base64) {
        if (base64.startsWith("iVBORw0KGgo")) {
            return "image/png";
        }
        if (base64.startsWith("/9j/")) {
            return "image/jpeg";
        }
        if (base64.startsWith("R0lGOD")) {
            return "image/gif";
        }
        if (base64.startsWith("UklGR")) {
            return "image/webp";
        }
        return null;
    }
}
