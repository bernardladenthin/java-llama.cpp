// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.value;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import java.util.Locale;
import java.util.Objects;
import lombok.EqualsAndHashCode;
import lombok.ToString;
import org.jspecify.annotations.Nullable;

/**
 * One piece of a {@link ChatMessage}'s multimodal content array: either a text
 * fragment or an image URL (typically a {@code data:image/...;base64,...} URI).
 * Mirrors the OpenAI-compatible {@code content} part shape understood by the
 * upstream {@code llama.cpp} server. The JNI bridge preserves decoded media
 * buffers and submits them through the upstream multimodal task path. An
 * image-bearing message is serialized to
 * <pre>
 * {"role":"user","content":[
 *   {"type":"text","text":"What is in this image?"},
 *   {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}
 * ]}
 * </pre>
 * and the upstream {@code oaicompat_chat_params_parse} routes it through the
 * compiled-in {@code mtmd} pipeline (requires
 * {@link net.ladenthin.llama.parameters.ModelParameters#setMmproj(String)} to be wired).
 * <p>
 * Instances are immutable and safe to share across threads. Use the static
 * factories &#x2014; the constructor is private.
 * </p>
 */
@ToString
@EqualsAndHashCode
public final class ContentPart {

    /** Discriminator for the two part kinds the OAI multipart schema supports. */
    public enum Type {
        /** A plain-text fragment. */
        TEXT,
        /** An image reference (data URI or remote URL). */
        IMAGE_URL,
        /** An audio clip (base64 {@code data} + {@code format}), for audio-input models. */
        INPUT_AUDIO,
        /**
         * A reference to a request tool by name (llama.cpp b11940): reveals a deferred
         * {@link ToolDefinition} to the model, whose template renders its definition here.
         */
        TOOL_REFERENCE
    }

    private final Type type;
    private final @Nullable String text;
    private final @Nullable String imageUrl;
    private final @Nullable String audioData;
    private final @Nullable String audioFormat;
    private final @Nullable String toolName;

    private ContentPart(
            Type type,
            @Nullable String text,
            @Nullable String imageUrl,
            @Nullable String audioData,
            @Nullable String audioFormat) {
        this(type, text, imageUrl, audioData, audioFormat, null);
    }

    private ContentPart(
            Type type,
            @Nullable String text,
            @Nullable String imageUrl,
            @Nullable String audioData,
            @Nullable String audioFormat,
            @Nullable String toolName) {
        this.type = type;
        this.text = text;
        this.imageUrl = imageUrl;
        this.audioData = audioData;
        this.audioFormat = audioFormat;
        this.toolName = toolName;
    }

    /**
     * Build a text part.
     *
     * @param text the text fragment (must not be {@code null})
     * @return a TEXT part wrapping {@code text}
     */
    public static ContentPart text(String text) {
        Objects.requireNonNull(text, "text");
        return new ContentPart(Type.TEXT, text, null, null, null);
    }

    /**
     * Build an image part from a pre-formed URL or data URI. Pass either an
     * HTTP(S) URL (if the server is configured to fetch it) or a complete
     * {@code data:image/...;base64,...} string.
     *
     * @param url image URL or data URI (must not be {@code null})
     * @return an IMAGE_URL part wrapping {@code url}
     */
    public static ContentPart imageUrl(String url) {
        Objects.requireNonNull(url, "url");
        return new ContentPart(Type.IMAGE_URL, null, url, null, null);
    }

    /**
     * Build an image part from raw bytes plus an explicit MIME type. The bytes
     * are base64-encoded and wrapped in a {@code data:} URI.
     *
     * @param bytes raw image bytes (must not be {@code null})
     * @param mimeType MIME type, e.g. {@code "image/png"} (must not be {@code null} or empty)
     * @return an IMAGE_URL part carrying the data URI
     */
    public static ContentPart imageBytes(byte[] bytes, String mimeType) {
        Objects.requireNonNull(bytes, "bytes");
        Objects.requireNonNull(mimeType, "mimeType");
        if (mimeType.isEmpty()) {
            throw new IllegalArgumentException("mimeType must not be empty (bytes.length=" + bytes.length + ")");
        }
        String encoded = Base64.getEncoder().encodeToString(bytes);
        return new ContentPart(Type.IMAGE_URL, null, "data:" + mimeType + ";base64," + encoded, null, null);
    }

    /**
     * Build an image part by reading a file from disk and detecting its MIME
     * type from the file extension. Recognised extensions: {@code .png},
     * {@code .jpg}, {@code .jpeg}, {@code .webp}, {@code .gif}. Anything else
     * throws {@link IllegalArgumentException}; use {@link #imageBytes(byte[], String)}
     * to force a MIME type explicitly.
     *
     * @param imagePath path to the image file (must not be {@code null})
     * @return an IMAGE_URL part carrying the data URI
     * @throws IOException if the file cannot be read
     */
    public static ContentPart imageFile(Path imagePath) throws IOException {
        Objects.requireNonNull(imagePath, "imagePath");
        Path fileNamePath = imagePath.getFileName();
        if (fileNamePath == null) {
            throw new IllegalArgumentException("imagePath has no file name component: " + imagePath);
        }
        String name = fileNamePath.toString().toLowerCase(Locale.ROOT);
        String mimeType;
        if (name.endsWith(".png")) {
            mimeType = "image/png";
        } else if (name.endsWith(".jpg") || name.endsWith(".jpeg")) {
            mimeType = "image/jpeg";
        } else if (name.endsWith(".webp")) {
            mimeType = "image/webp";
        } else if (name.endsWith(".gif")) {
            mimeType = "image/gif";
        } else {
            throw new IllegalArgumentException("Cannot infer MIME type from extension: " + imagePath
                    + " — use ContentPart.imageBytes(bytes, mimeType) instead");
        }
        return imageBytes(Files.readAllBytes(imagePath), mimeType);
    }

    /**
     * Build an audio part from raw bytes plus an explicit container format. Mirrors the OpenAI
     * {@code input_audio} content part the upstream {@code llama.cpp} server understands, routed
     * through the {@code mtmd} audio pipeline (requires an audio-capable {@code --mmproj}). The bytes
     * are base64-encoded.
     *
     * @param audioBytes raw audio bytes (must not be {@code null})
     * @param format container format, {@code "wav"} or {@code "mp3"} (case-insensitive)
     * @return an INPUT_AUDIO part carrying the base64 data and normalised format
     * @throws IllegalArgumentException if {@code format} is not {@code "wav"} or {@code "mp3"}
     */
    public static ContentPart inputAudio(byte[] audioBytes, String format) {
        Objects.requireNonNull(audioBytes, "audioBytes");
        Objects.requireNonNull(format, "format");
        String normalized = format.toLowerCase(Locale.ROOT);
        if (!normalized.equals("wav") && !normalized.equals("mp3")) {
            throw new IllegalArgumentException("audio format must be 'wav' or 'mp3', was: " + format);
        }
        String encoded = Base64.getEncoder().encodeToString(audioBytes);
        return new ContentPart(Type.INPUT_AUDIO, null, null, encoded, normalized);
    }

    /**
     * Build a tool reference: names a tool of the request, typically a deferred one
     * ({@link ToolDefinition#isDeferLoading()}), whose definition the chat template then renders at
     * this point of the conversation -- the result of a tool search, for instance (llama.cpp b11940).
     * Sent as {@code {"type":"tool_reference","name":...}}; the request must declare the tool.
     *
     * @param toolName the name of a tool of the request (must not be {@code null} or empty)
     * @return a TOOL_REFERENCE part
     * @throws IllegalArgumentException if {@code toolName} is empty
     */
    public static ContentPart toolReference(String toolName) {
        Objects.requireNonNull(toolName, "toolName");
        if (toolName.isEmpty()) {
            throw new IllegalArgumentException("toolName must not be empty (length=" + toolName.length() + ")");
        }
        return new ContentPart(Type.TOOL_REFERENCE, null, null, null, null, toolName);
    }

    /**
     * Build an audio part by reading a file from disk and detecting its format from the file
     * extension. Recognised extensions: {@code .wav}, {@code .mp3}. Anything else throws
     * {@link IllegalArgumentException}; use {@link #inputAudio(byte[], String)} to force a format.
     *
     * @param audioPath path to the audio file (must not be {@code null})
     * @return an INPUT_AUDIO part carrying the data
     * @throws IOException if the file cannot be read
     */
    public static ContentPart audioFile(Path audioPath) throws IOException {
        Objects.requireNonNull(audioPath, "audioPath");
        Path fileNamePath = audioPath.getFileName();
        if (fileNamePath == null) {
            throw new IllegalArgumentException("audioPath has no file name component: " + audioPath);
        }
        String name = fileNamePath.toString().toLowerCase(Locale.ROOT);
        String format;
        if (name.endsWith(".wav")) {
            format = "wav";
        } else if (name.endsWith(".mp3")) {
            format = "mp3";
        } else {
            throw new IllegalArgumentException("Cannot infer audio format from extension: " + audioPath
                    + " — use ContentPart.inputAudio(bytes, format) instead");
        }
        return inputAudio(Files.readAllBytes(audioPath), format);
    }

    /**
     * Part-kind accessor.
     * @return the discriminator selecting {@link #getText()} or {@link #getImageUrl()}
     */
    public Type getType() {
        return type;
    }

    /**
     * Text accessor (only set for {@link Type#TEXT}).
     * @return the text fragment, or {@code null} for {@link Type#IMAGE_URL} parts
     */
    public @Nullable String getText() {
        return text;
    }

    /**
     * Image URL accessor (only set for {@link Type#IMAGE_URL}).
     * @return the URL or data URI, or {@code null} for {@link Type#TEXT} parts
     */
    public @Nullable String getImageUrl() {
        return imageUrl;
    }

    /**
     * Base64 audio-data accessor (only set for {@link Type#INPUT_AUDIO}).
     * @return the base64-encoded audio bytes, or {@code null} for non-audio parts
     */
    public @Nullable String getAudioData() {
        return audioData;
    }

    /**
     * Audio container-format accessor (only set for {@link Type#INPUT_AUDIO}).
     * @return {@code "wav"} or {@code "mp3"}, or {@code null} for non-audio parts
     */
    public @Nullable String getAudioFormat() {
        return audioFormat;
    }

    /**
     * Tool-name accessor (only set for {@link Type#TOOL_REFERENCE}).
     * @return the name of the referenced tool, or {@code null} for other parts
     */
    public @Nullable String getToolName() {
        return toolName;
    }
}
