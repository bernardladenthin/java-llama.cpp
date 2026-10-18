// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import lombok.EqualsAndHashCode;
import org.jspecify.annotations.Nullable;

/**
 * Immutable configuration for {@link OpenAiCompatServer}.
 *
 * <p>Sensible localhost defaults are provided; build instances with {@link #builder()}. The API key is
 * deliberately excluded from {@link #toString()} so it is never written to logs (but it <em>is</em> part
 * of {@link #equals(Object)}/{@link #hashCode()}, which are never logged).
 */
@EqualsAndHashCode
public final class OpenAiServerConfig {

    /** Default bind address: loopback only, so the endpoint is not exposed off-host. */
    public static final String DEFAULT_HOST = "127.0.0.1";

    /** Default TCP port. */
    public static final int DEFAULT_PORT = 8080;

    /** Default advertised model id (the {@code id} echoed by {@code GET /v1/models}). */
    public static final String DEFAULT_MODEL_ID = "local-model";

    /** Default advertised maximum input tokens. */
    public static final int DEFAULT_MAX_INPUT_TOKENS = 8192;

    /** Default advertised maximum output tokens. */
    public static final int DEFAULT_MAX_OUTPUT_TOKENS = 2048;

    /** Default Server-Sent-Events heartbeat interval, in milliseconds. */
    public static final long DEFAULT_HEARTBEAT_MILLIS = 15_000L;

    /**
     * Default {@code Access-Control-Allow-Origin} value: {@code "*"}. Browser- and webview-based clients
     * send a CORS preflight and require this header; {@code "*"} is the pragmatic default for a server
     * that binds loopback and authenticates with a bearer token (not cookies).
     */
    public static final String DEFAULT_CORS_ALLOW_ORIGIN = "*";

    /** Default maximum accepted request-body size, in bytes (16 MiB). */
    public static final int DEFAULT_MAX_REQUEST_BODY_BYTES = 16 * 1024 * 1024;

    private final String host;
    private final int port;
    private final @Nullable String apiKey;
    private final String modelId;
    private final int maxInputTokens;
    private final int maxOutputTokens;
    private final long heartbeatMillis;
    private final String corsAllowOrigin;
    private final boolean supportsVision;
    private final int maxRequestBodyBytes;
    private final String modelFtype;

    private OpenAiServerConfig(Builder builder) {
        this.host = builder.host;
        this.port = builder.port;
        this.apiKey = builder.apiKey;
        this.modelId = builder.modelId;
        this.maxInputTokens = builder.maxInputTokens;
        this.maxOutputTokens = builder.maxOutputTokens;
        this.heartbeatMillis = builder.heartbeatMillis;
        this.corsAllowOrigin = builder.corsAllowOrigin;
        this.supportsVision = builder.supportsVision;
        this.maxRequestBodyBytes = builder.maxRequestBodyBytes;
        this.modelFtype = builder.modelFtype;
    }

    /**
     * Returns a new builder seeded with the localhost defaults.
     *
     * @return a fresh {@link Builder}
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * The bind address (loopback by default).
     *
     * @return the host the server binds to
     */
    public String getHost() {
        return host;
    }

    /**
     * The TCP port.
     *
     * @return the port the server listens on
     */
    public int getPort() {
        return port;
    }

    /**
     * The optional bearer API key. When {@code null}, no {@code Authorization} header is required.
     *
     * @return the configured API key, or {@code null} when authentication is disabled
     */
    public @Nullable String getApiKey() {
        return apiKey;
    }

    /**
     * The advertised model id.
     *
     * @return the model id reported by {@code GET /v1/models}
     */
    public String getModelId() {
        return modelId;
    }

    /**
     * The advertised maximum input-token budget.
     *
     * @return the advertised max input tokens
     */
    public int getMaxInputTokens() {
        return maxInputTokens;
    }

    /**
     * The advertised maximum output-token budget.
     *
     * @return the advertised max output tokens
     */
    public int getMaxOutputTokens() {
        return maxOutputTokens;
    }

    /**
     * The Server-Sent-Events heartbeat interval.
     *
     * @return the heartbeat interval in milliseconds
     */
    public long getHeartbeatMillis() {
        return heartbeatMillis;
    }

    /**
     * The {@code Access-Control-Allow-Origin} value sent on every response and CORS preflight.
     *
     * @return the allowed CORS origin
     */
    public String getCorsAllowOrigin() {
        return corsAllowOrigin;
    }

    /**
     * Maximum accepted request-body size, in bytes. Bodies larger than this are rejected with HTTP 413.
     *
     * @return the request-body cap in bytes
     */
    public int getMaxRequestBodyBytes() {
        return maxRequestBodyBytes;
    }

    /**
     * Whether the served model supports image input (a multimodal projector was configured). Advertised
     * to clients that gate on a vision capability (e.g. Copilot's Ollama provider via {@code /api/show}).
     *
     * @return {@code true} if vision/image input is available
     */
    public boolean isSupportsVision() {
        return supportsVision;
    }

    /**
     * The served model's file type (quantization) as a human-readable string, e.g. {@code "Q8_0"}
     * or {@code "Q4_K - Medium"}, advertised in the {@code GET /v1/models} {@code data[].ftype} field
     * (matching the upstream llama.cpp server). Empty when unknown.
     *
     * @return the quantization file-type label, or {@code ""} if unknown
     */
    public String getModelFtype() {
        return modelFtype;
    }

    /**
     * Whether bearer-token authentication is enabled (an API key is configured).
     *
     * @return {@code true} if requests must present a matching bearer token
     */
    public boolean isAuthenticationEnabled() {
        return apiKey != null && !apiKey.isEmpty();
    }

    /**
     * Renders the configuration without exposing the API key.
     *
     * @return a log-safe description of this configuration
     */
    @Override
    public String toString() {
        return "OpenAiServerConfig{host="
                + host
                + ", port="
                + port
                + ", authEnabled="
                + isAuthenticationEnabled()
                + ", modelId="
                + modelId
                + ", maxInputTokens="
                + maxInputTokens
                + ", maxOutputTokens="
                + maxOutputTokens
                + ", heartbeatMillis="
                + heartbeatMillis
                + ", corsAllowOrigin="
                + corsAllowOrigin
                + '}';
    }

    /** Mutable builder for {@link OpenAiServerConfig}; not thread-safe. */
    public static final class Builder {

        private String host = DEFAULT_HOST;
        private int port = DEFAULT_PORT;
        private @Nullable String apiKey;
        private String modelId = DEFAULT_MODEL_ID;
        private int maxInputTokens = DEFAULT_MAX_INPUT_TOKENS;
        private int maxOutputTokens = DEFAULT_MAX_OUTPUT_TOKENS;
        private long heartbeatMillis = DEFAULT_HEARTBEAT_MILLIS;
        private String corsAllowOrigin = DEFAULT_CORS_ALLOW_ORIGIN;
        private boolean supportsVision;
        private int maxRequestBodyBytes = DEFAULT_MAX_REQUEST_BODY_BYTES;
        private String modelFtype = "";

        private Builder() {}

        /**
         * Sets the bind address.
         *
         * @param host the host to bind (e.g. {@code "127.0.0.1"})
         * @return this builder
         */
        public Builder host(String host) {
            this.host = host;
            return this;
        }

        /**
         * Sets the TCP port.
         *
         * @param port the port to listen on
         * @return this builder
         */
        public Builder port(int port) {
            this.port = port;
            return this;
        }

        /**
         * Sets the optional bearer API key. Pass {@code null} (the default) to disable authentication.
         *
         * @param apiKey the required bearer token, or {@code null} for no authentication
         * @return this builder
         */
        public Builder apiKey(@Nullable String apiKey) {
            this.apiKey = apiKey;
            return this;
        }

        /**
         * Sets the advertised model id.
         *
         * @param modelId the model id to advertise
         * @return this builder
         */
        public Builder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        /**
         * Sets the advertised maximum input tokens.
         *
         * @param maxInputTokens the advertised max input tokens
         * @return this builder
         */
        public Builder maxInputTokens(int maxInputTokens) {
            this.maxInputTokens = maxInputTokens;
            return this;
        }

        /**
         * Sets the advertised maximum output tokens.
         *
         * @param maxOutputTokens the advertised max output tokens
         * @return this builder
         */
        public Builder maxOutputTokens(int maxOutputTokens) {
            this.maxOutputTokens = maxOutputTokens;
            return this;
        }

        /**
         * Sets the Server-Sent-Events heartbeat interval.
         *
         * @param heartbeatMillis the heartbeat interval in milliseconds; must be positive
         * @return this builder
         * @throws IllegalArgumentException if {@code heartbeatMillis} is not positive
         */
        public Builder heartbeatMillis(long heartbeatMillis) {
            // the interval is a scheduleAtFixedRate period; a non-positive one would make every
            // streaming request fail with an IllegalArgumentException reported as the client's error
            if (heartbeatMillis <= 0L) {
                throw new IllegalArgumentException("heartbeatMillis must be positive, was " + heartbeatMillis);
            }
            this.heartbeatMillis = heartbeatMillis;
            return this;
        }

        /**
         * Sets the {@code Access-Control-Allow-Origin} value (CORS).
         *
         * @param corsAllowOrigin the allowed origin (e.g. {@code "*"} or a specific scheme/host/port)
         * @return this builder
         */
        public Builder corsAllowOrigin(String corsAllowOrigin) {
            this.corsAllowOrigin = corsAllowOrigin;
            return this;
        }

        /**
         * Sets whether the served model supports image input (a multimodal projector is configured).
         *
         * @param supportsVision {@code true} if vision/image input is available
         * @return this builder
         */
        public Builder supportsVision(boolean supportsVision) {
            this.supportsVision = supportsVision;
            return this;
        }

        /**
         * Sets the served model's file type (quantization) label to advertise in {@code /v1/models}.
         *
         * @param modelFtype the quantization file-type label (e.g. {@code "Q4_K - Medium"}); {@code null}
         *     is treated as empty (unknown)
         * @return this builder
         */
        public Builder modelFtype(@Nullable String modelFtype) {
            this.modelFtype = modelFtype == null ? "" : modelFtype;
            return this;
        }

        /**
         * Sets the maximum accepted request-body size in bytes. Bodies larger than this are rejected
         * with HTTP 413 before being buffered.
         *
         * @param maxRequestBodyBytes the request-body cap in bytes
         * @return this builder
         */
        public Builder maxRequestBodyBytes(int maxRequestBodyBytes) {
            this.maxRequestBodyBytes = maxRequestBodyBytes;
            return this;
        }

        /**
         * Builds the immutable configuration.
         *
         * @return a new {@link OpenAiServerConfig}
         */
        public OpenAiServerConfig build() {
            return new OpenAiServerConfig(this);
        }
    }
}
