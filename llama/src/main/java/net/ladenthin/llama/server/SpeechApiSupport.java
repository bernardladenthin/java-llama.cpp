// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import com.fasterxml.jackson.databind.JsonNode;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Base64;
import org.jspecify.annotations.Nullable;

/**
 * The request and response shaping of OpenAI's speech route ({@code POST /v1/audio/speech}), with
 * the rules of llama.cpp's own route (b12238, {@code convert_speech_to_tts}): {@code input} is the
 * text, {@code voice} a name or {@code {"id": name}}, {@code response_format} {@code wav} (the
 * default here, where OpenAI's is {@code mp3}) or {@code pcm} (raw signed 16-bit little-endian mono
 * samples), {@code speed} only {@code 1.0}, {@code instructions} ignored. Pure: no HTTP, no model.
 */
final class SpeechApiSupport {

    /** {@code response_format} of a whole WAV file. */
    static final String FORMAT_WAV = "wav";

    /** {@code response_format} of the samples alone, without the WAV header. */
    static final String FORMAT_PCM = "pcm";

    /** {@code response_format} of {@code /tts} for signed 16-bit samples. */
    static final String FORMAT_PCM16 = "pcm16";

    /** One past the largest magnitude of a signed 16-bit sample. */
    private static final String FIELD_N_PREDICT = "n_predict";

    private static final float PCM16_SCALE = 32768.0f;

    /** The {@code Content-Type} of a WAV answer. */
    static final String CONTENT_TYPE_WAV = "audio/wav";

    /** The size of the header {@link net.ladenthin.llama.TextToSpeech} writes: RIFF, fmt, data. */
    static final int WAV_HEADER_BYTES = 44;

    /** Where a canonical WAV header carries its sample rate, as a little-endian 32-bit integer. */
    private static final int WAV_SAMPLE_RATE_OFFSET = 24;

    private SpeechApiSupport() {}

    /** What a request asked the audio to be answered as. */
    enum Format {
        /** A whole WAV file. */
        WAV,
        /** Signed 16-bit little-endian mono samples. */
        PCM16,
        /** 32-bit float little-endian mono samples. */
        PCM_FLOAT
    }

    /** One parsed speech request. */
    static final class Request {
        private final String input;
        private final @Nullable String voice;
        private final @Nullable String language;
        private final Format format;
        private final SpeechOptions options;

        Request(String input, @Nullable String voice, @Nullable String language, Format format, SpeechOptions options) {
            this.input = input;
            this.voice = voice;
            this.language = language;
            this.format = format;
            this.options = options;
        }

        String input() {
            return input;
        }

        @Nullable
        String voice() {
            return voice;
        }

        @Nullable
        String language() {
            return language;
        }

        Format format() {
            return format;
        }

        SpeechOptions options() {
            return options;
        }

        boolean pcm() {
            return format == Format.PCM16;
        }

        /**
         * The language to speak in: the request's, else the server's default.
         *
         * @param serverDefault the server's default language, or {@code null}
         * @return the language, or {@code null} for the model's own default
         */
        @Nullable
        String languageOr(@Nullable String serverDefault) {
            return language != null ? language : serverDefault;
        }

        /**
         * The voice to speak with: the request's, else the server's default -- which a speaker
         * reference in the request overrides, as in llama.cpp's server (b12283).
         *
         * @param serverDefault the server's default voice, or {@code null}
         * @return the voice, or {@code null} for the model's own default
         */
        @Nullable
        String voiceOr(@Nullable String serverDefault) {
            if (voice != null || options.getSpeakerReference() != null) {
                return voice;
            }
            return serverDefault;
        }
    }

    /**
     * Parse a speech request.
     *
     * @param request the request body
     * @return the parsed request
     * @throws IllegalArgumentException if the request is one this route does not serve
     */
    static Request parse(JsonNode request) {
        JsonNode input = request.path("input");
        if (!input.isTextual() || input.asText().isEmpty()) {
            throw new IllegalArgumentException("'input' must be a non-empty string");
        }
        String format = request.path("response_format").asText(FORMAT_WAV);
        if (!FORMAT_WAV.equals(format) && !FORMAT_PCM.equals(format)) {
            throw new IllegalArgumentException("Only 'wav' and 'pcm' response_format are supported for speech");
        }
        // "audio" asks for the answer as it becomes available; this server answers it in one piece,
        // which is the same bytes. "sse" is another protocol and refused, as upstream does.
        String streamFormat = request.path("stream_format").asText("");
        if (!streamFormat.isEmpty() && !"audio".equals(streamFormat)) {
            throw new IllegalArgumentException("Only 'audio' stream_format is supported for speech");
        }
        JsonNode speed = request.path("speed");
        if (!speed.isMissingNode() && !speed.isNull() && (!speed.isNumber() || speed.asDouble() != 1.0)) {
            throw new IllegalArgumentException("'speed' is not supported for speech");
        }
        JsonNode voice = request.path("voice");
        String voiceName = voice.isObject() ? voice.path("id").asText("") : voice.asText("");
        String language = request.path("lang").asText("");
        return new Request(
                input.asText(),
                voiceName.isEmpty() ? null : voiceName,
                language.isEmpty() ? null : language,
                FORMAT_PCM.equals(format) ? Format.PCM16 : Format.WAV,
                SpeechOptions.defaults());
    }

    /**
     * Parse a request of llama.cpp's own speech route, {@code POST /tts}: {@code input} (or
     * {@code prompt}) is the text, {@code voice} and {@code lang} as the model names them,
     * {@code response_format} {@code wav}, {@code pcm} (32-bit float samples) or {@code pcm16},
     * {@code speaker_ref_b64} an audio file to clone the voice from, {@code n_predict} (or
     * {@code max_tokens}) the frame cap, {@code top_k}, {@code seed} and {@code temperature} the
     * sampling. {@code top_p} and {@code repeat_penalty} are read by upstream's server only and are
     * ignored here; {@code stream} is refused, this server answers the audio in one piece.
     *
     * @param request the request body
     * @return the parsed request
     * @throws IllegalArgumentException if the request is one this route does not serve
     */
    static Request parseTts(JsonNode request) {
        // upstream's alias "prompt" stands in for an absent input; neither is coerced from another type
        JsonNode text = request.path("input");
        if (text.isMissingNode() || text.isNull()) {
            text = request.path("prompt");
        }
        if (!text.isTextual() || text.asText().isEmpty()) {
            throw new IllegalArgumentException("\"input\" must be a non-empty string");
        }
        String input = text.asText();
        if (request.path("stream").asBoolean(false)) {
            throw new IllegalArgumentException("\"stream\" is not supported for /tts on this server");
        }
        String name = request.path("response_format").asText(FORMAT_WAV);
        final Format format;
        if (FORMAT_WAV.equals(name)) {
            format = Format.WAV;
        } else if (FORMAT_PCM.equals(name)) {
            format = Format.PCM_FLOAT;
        } else if (FORMAT_PCM16.equals(name)) {
            format = Format.PCM16;
        } else {
            throw new IllegalArgumentException("\"response_format\" must be \"wav\", \"pcm\" or \"pcm16\"");
        }
        byte[] reference = null;
        String encoded = request.path("speaker_ref_b64").asText("");
        if (!encoded.isEmpty()) {
            try {
                reference = Base64.getDecoder().decode(encoded);
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException("\"speaker_ref_b64\" is not valid base64", e);
            }
        }
        JsonNode frames = request.path(request.hasNonNull(FIELD_N_PREDICT) ? FIELD_N_PREDICT : "max_tokens");
        JsonNode temperature = request.path("temperature");
        if (temperature.isNumber() && temperature.asDouble() < 0.0) {
            throw new IllegalArgumentException("Field 'temperature': must not be negative");
        }
        String voice = request.path("voice").asText("");
        String language = request.path("lang").asText("");
        return new Request(
                input,
                voice.isEmpty() ? null : voice,
                language.isEmpty() ? null : language,
                format,
                new SpeechOptions(
                        reference,
                        integerOrNull(frames, FIELD_N_PREDICT),
                        integerOrNull(request.path("top_k"), "top_k"),
                        integerOrNull(request.path("seed"), "seed"),
                        temperature.isNumber() ? Float.valueOf((float) temperature.asDouble()) : null));
    }

    /** An absent or null field is unset; anything but an integer is the request's fault. */
    private static @Nullable Integer integerOrNull(JsonNode node, String name) {
        if (node.isMissingNode() || node.isNull()) {
            return null;
        }
        if (!node.isIntegralNumber() || !node.canConvertToInt()) {
            throw new IllegalArgumentException("Field '" + name + "': must be an integer");
        }
        return node.asInt();
    }

    /**
     * The body a request's format asks for.
     *
     * @param wav the engine's WAV file
     * @param format the format the request named
     * @return the file itself, or its samples as that format has them
     */
    static byte[] body(byte[] wav, Format format) {
        switch (format) {
            case PCM16:
                return pcm(wav);
            case PCM_FLOAT:
                return pcmFloat(wav);
            default:
                return wav;
        }
    }

    /**
     * The {@code Content-Type} of {@link #body}.
     *
     * @param wav the engine's WAV file
     * @param format the format the request named
     * @return the content type, which for raw samples carries their rate and encoding
     */
    static String contentType(byte[] wav, Format format) {
        switch (format) {
            case PCM16:
                return pcmContentType(sampleRate(wav));
            case PCM_FLOAT:
                return "audio/pcm;rate=" + sampleRate(wav) + ";encoding=float;bits=32";
            default:
                return CONTENT_TYPE_WAV;
        }
    }

    /**
     * The samples of a WAV file as 32-bit floats in [-1, 1), little-endian: what {@code /tts}
     * answers for {@code response_format: pcm}.
     *
     * @param wav a 16-bit mono WAV file with the canonical 44-byte header
     * @return four bytes per sample
     */
    static byte[] pcmFloat(byte[] wav) {
        byte[] samples = pcm(wav);
        ByteBuffer in = ByteBuffer.wrap(samples).order(ByteOrder.LITTLE_ENDIAN);
        ByteBuffer out = ByteBuffer.allocate(samples.length / 2 * 4).order(ByteOrder.LITTLE_ENDIAN);
        while (in.remaining() >= 2) {
            out.putFloat(in.getShort() / PCM16_SCALE);
        }
        return out.array();
    }

    /**
     * The sample rate a WAV file declares.
     *
     * @param wav a WAV file with the canonical 44-byte header
     * @return its sample rate in Hz
     * @throws IllegalArgumentException if the bytes are no such file
     */
    static int sampleRate(byte[] wav) {
        if (wav.length < WAV_HEADER_BYTES || wav[0] != 'R' || wav[1] != 'I' || wav[2] != 'F' || wav[3] != 'F') {
            throw new IllegalArgumentException("the speech engine did not return a WAV file");
        }
        return (wav[WAV_SAMPLE_RATE_OFFSET] & 0xFF)
                | (wav[WAV_SAMPLE_RATE_OFFSET + 1] & 0xFF) << 8
                | (wav[WAV_SAMPLE_RATE_OFFSET + 2] & 0xFF) << 16
                | (wav[WAV_SAMPLE_RATE_OFFSET + 3] & 0xFF) << 24;
    }

    /**
     * The samples of a WAV file without its header: what {@code response_format: pcm} answers.
     *
     * @param wav a WAV file with the canonical 44-byte header
     * @return the signed 16-bit little-endian samples
     */
    static byte[] pcm(byte[] wav) {
        sampleRate(wav); // validates the header
        return Arrays.copyOfRange(wav, WAV_HEADER_BYTES, wav.length);
    }

    /**
     * The {@code Content-Type} of a {@code pcm} answer, which carries what the missing header would
     * have said, as upstream's does.
     *
     * @param sampleRate the sample rate in Hz
     * @return the content type
     */
    static String pcmContentType(int sampleRate) {
        return "audio/pcm;rate=" + sampleRate + ";encoding=signed-int;bits=16";
    }
}
