// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link OpenAiServerConfig}: builder defaults, the authentication predicate, the CORS /
 * vision knobs, and the security contract that {@link OpenAiServerConfig#toString()} never leaks the API
 * key. Pure — no socket, no model.
 */
public class OpenAiServerConfigTest {

    @Test
    public void builderAppliesLocalhostDefaults() {
        OpenAiServerConfig config = OpenAiServerConfig.builder().build();
        assertThat(config.getHost(), is(OpenAiServerConfig.DEFAULT_HOST));
        assertThat(config.getPort(), is(OpenAiServerConfig.DEFAULT_PORT));
        assertThat(config.getModelId(), is(OpenAiServerConfig.DEFAULT_MODEL_ID));
        assertThat(config.getMaxInputTokens(), is(OpenAiServerConfig.DEFAULT_MAX_INPUT_TOKENS));
        assertThat(config.getMaxOutputTokens(), is(OpenAiServerConfig.DEFAULT_MAX_OUTPUT_TOKENS));
        assertThat(config.getHeartbeatMillis(), is(OpenAiServerConfig.DEFAULT_HEARTBEAT_MILLIS));
        assertThat(config.getCorsAllowOrigin(), is(OpenAiServerConfig.DEFAULT_CORS_ALLOW_ORIGIN));
        assertThat(config.isSupportsVision(), is(false));
        assertThat(config.getModelFtype(), is(""));
        assertThat(config.getApiKey(), is((String) null));
        assertThat(config.isAuthenticationEnabled(), is(false));
    }

    /**
     * Both servers listen on llama.cpp's default port: {@code NativeServer} has since b11521 (upstream
     * #30159), and this one said 8080, so the two transports of one library defaulted differently.
     */
    @Test
    public void theDefaultPortIsLlamaCppsDefaultPort() {
        assertThat(OpenAiServerConfig.DEFAULT_PORT, is(9931));
        assertThat(OpenAiServerConfig.builder().build().getPort(), is(9931));
    }

    @Test
    public void modelFtypeIsConfigurableAndNullBecomesEmpty() {
        assertThat(
                OpenAiServerConfig.builder().modelFtype("Q4_K - Medium").build().getModelFtype(), is("Q4_K - Medium"));
        // null is normalized to the empty "unknown" marker
        assertThat(OpenAiServerConfig.builder().modelFtype(null).build().getModelFtype(), is(""));
    }

    @Test
    public void authenticationEnabledOnlyForNonEmptyKey() {
        assertThat(OpenAiServerConfig.builder().build().isAuthenticationEnabled(), is(false));
        assertThat(OpenAiServerConfig.builder().apiKey("").build().isAuthenticationEnabled(), is(false));
        assertThat(OpenAiServerConfig.builder().apiKey("secret").build().isAuthenticationEnabled(), is(true));
    }

    @Test
    public void toStringNeverLeaksTheApiKey() {
        String secret = "sk-super-secret-value-1234567890";
        OpenAiServerConfig config = OpenAiServerConfig.builder().apiKey(secret).build();
        String rendered = config.toString();
        // The key value must not appear; only the boolean auth state is exposed.
        assertThat(rendered, not(containsString(secret)));
        assertThat(rendered, containsString("authEnabled=true"));
    }

    @Test
    public void corsAndVisionAreConfigurable() {
        OpenAiServerConfig config = OpenAiServerConfig.builder()
                .corsAllowOrigin("https://editor.example")
                .supportsVision(true)
                .build();
        assertThat(config.getCorsAllowOrigin(), is("https://editor.example"));
        assertThat(config.isSupportsVision(), is(true));
    }

    @Test
    public void tokenBudgetsAndPortAreConfigurable() {
        OpenAiServerConfig config = OpenAiServerConfig.builder()
                .port(0)
                .modelId("local-qwen")
                .maxInputTokens(6144)
                .maxOutputTokens(2048)
                .build();
        assertThat(config.getPort(), is(0));
        assertThat(config.getModelId(), is("local-qwen"));
        assertThat(config.getMaxInputTokens(), is(6144));
        assertThat(config.getMaxOutputTokens(), is(2048));
    }

    /**
     * The interval is a {@code scheduleAtFixedRate} period, which must be positive. Accepted, a zero
     * made every streaming request fail with an {@code invalid_request_error} naming
     * {@code IllegalArgumentException} and no {@code [DONE]}, which blames the client for a server setting.
     */
    @Test
    public void heartbeatIntervalOfZeroIsRejected() {
        assertThrows(
                IllegalArgumentException.class,
                () -> OpenAiServerConfig.builder().heartbeatMillis(0L));
    }

    @Test
    public void totalSlotsDefaultsToOneAndKeepsTheConfiguredCount() {
        assertThat(OpenAiServerConfig.builder().build().getTotalSlots(), is(1));
        assertThat(OpenAiServerConfig.builder().totalSlots(4).build().getTotalSlots(), is(4));
    }

    @Test
    public void totalSlotsBelowOneIsRejected() {
        assertThrows(
                IllegalArgumentException.class,
                () -> OpenAiServerConfig.builder().totalSlots(0));
    }
}
