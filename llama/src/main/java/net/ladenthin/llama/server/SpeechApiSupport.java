// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import com.fasterxml.jackson.databind.JsonNode;
import java.util.Arrays;
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

    /** The {@code Content-Type} of a WAV answer. */
    static final String CONTENT_TYPE_WAV = "audio/wav";

    /** The size of the header {@link net.ladenthin.llama.TextToSpeech} writes: RIFF, fmt, data. */
    static final int WAV_HEADER_BYTES = 44;

    /** Where a canonical WAV header carries its sample rate, as a little-endian 32-bit integer. */
    private static final int WAV_SAMPLE_RATE_OFFSET = 24;

    private SpeechApiSupport() {}

    /** One parsed speech request. */
    static final class Request {
        private final String input;
        private final @Nullable String voice;
        private final @Nullable String language;
        private final boolean pcm;

        Request(String input, @Nullable String voice, @Nullable String language, boolean pcm) {
            this.input = input;
            this.voice = voice;
            this.language = language;
            this.pcm = pcm;
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

        boolean pcm() {
            return pcm;
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
                FORMAT_PCM.equals(format));
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
