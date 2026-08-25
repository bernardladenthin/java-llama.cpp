// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertThrows;

import net.ladenthin.llama.args.CacheType;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link OpenAiServerCli}: parsing of long/short/alias flags, defaults, model-id
 * derivation, rejection of bad input, and the {@link OpenAiServerCli.Options#toServerConfig()}
 * projection (auth + context-derived token budgets). Pure — no socket and no native model.
 */
public class OpenAiServerCliTest {

    @Test
    public void minimalArgsApplyDefaults() {
        OpenAiServerCli.Options options = OpenAiServerCli.parse("--model", "models/Qwen3-0.6B.gguf");
        assertThat(options.getModelPath(), is("models/Qwen3-0.6B.gguf"));
        assertThat(options.getHost(), is(OpenAiServerCli.DEFAULT_HOST));
        assertThat(options.getPort(), is(OpenAiServerCli.DEFAULT_PORT));
        assertThat(options.getCtxSize(), is(0));
        assertThat(options.getGpuLayers(), is(0));
        assertThat(options.getThreads(), is(0));
        assertThat(options.getParallel(), is(0));
        assertThat(options.isEmbedding(), is(false));
        assertThat(options.getApiKey(), is((String) null));
        // Model id defaults to the model file name.
        assertThat(options.getModelId(), is("Qwen3-0.6B.gguf"));
    }

    @Test
    public void allLongFlagsParsed() {
        OpenAiServerCli.Options options = OpenAiServerCli.parse(
                "--model",
                "m.gguf",
                "--host",
                "0.0.0.0",
                "--port",
                "9090",
                "--ctx-size",
                "4096",
                "--n-gpu-layers",
                "99",
                "--threads",
                "8",
                "--parallel",
                "2",
                "--model-id",
                "my-model",
                "--api-key",
                "secret",
                "--embedding");
        assertThat(options.getModelPath(), is("m.gguf"));
        assertThat(options.getHost(), is("0.0.0.0"));
        assertThat(options.getPort(), is(9090));
        assertThat(options.getCtxSize(), is(4096));
        assertThat(options.getGpuLayers(), is(99));
        assertThat(options.getThreads(), is(8));
        assertThat(options.getParallel(), is(2));
        assertThat(options.getModelId(), is("my-model"));
        assertThat(options.getApiKey(), is("secret"));
        assertThat(options.isEmbedding(), is(true));
    }

    @Test
    public void shortFlagsParsed() {
        OpenAiServerCli.Options options =
                OpenAiServerCli.parse("-m", "m.gguf", "-p", "1234", "-c", "512", "-ngl", "10", "-t", "4");
        assertThat(options.getPort(), is(1234));
        assertThat(options.getCtxSize(), is(512));
        assertThat(options.getGpuLayers(), is(10));
        assertThat(options.getThreads(), is(4));
    }

    @Test
    public void legacyAliasFlagsAccepted() {
        OpenAiServerCli.Options options = OpenAiServerCli.parse(
                "-m", "m.gguf", "--ctx", "256", "--gpu-layers", "5", "--model-alias", "aliased", "--embeddings");
        assertThat(options.getCtxSize(), is(256));
        assertThat(options.getGpuLayers(), is(5));
        assertThat(options.getModelId(), is("aliased"));
        assertThat(options.isEmbedding(), is(true));
    }

    @Test
    public void mmprojFlagParsed() {
        OpenAiServerCli.Options options = OpenAiServerCli.parse("-m", "m.gguf", "--mmproj", "proj.gguf");
        assertThat(options.getMmproj(), is("proj.gguf"));
        assertThat(OpenAiServerCli.parse("-m", "m.gguf").getMmproj(), is((String) null));
    }

    @Test
    public void mmprojDeviceFlagParsedUnderBothSpellings() {
        assertThat(
                OpenAiServerCli.parse("-m", "m.gguf", "--mmproj-device", "CUDA1")
                        .getMmprojDevice(),
                is("CUDA1"));
        assertThat(OpenAiServerCli.parse("-m", "m.gguf", "-mmdev", "CUDA1").getMmprojDevice(), is("CUDA1"));
        assertThat(OpenAiServerCli.parse("-m", "m.gguf").getMmprojDevice(), is((String) null));
    }

    @Test
    public void mmprojDeviceReachesModelParameters() {
        assertThat(
                OpenAiServerCli.parse("-m", "m.gguf", "--mmproj", "proj.gguf", "-mmdev", "none")
                        .toModelParameters()
                        .toString(),
                containsString("--mmproj-device"));
    }

    @Test
    public void mmprojDeviceDoesNotByItselfEnableVisionCapability() {
        // The vision hint is derived from --mmproj; naming a device without a projector must not
        // make the server advertise image support.
        assertThat(
                OpenAiServerCli.parse("-m", "m.gguf", "-mmdev", "CUDA1")
                        .toServerConfig()
                        .isSupportsVision(),
                is(false));
    }

    @Test
    public void mmprojEnablesVisionCapabilityInServerConfig() {
        assertThat(
                OpenAiServerCli.parse("-m", "m.gguf", "--mmproj", "proj.gguf")
                        .toServerConfig()
                        .isSupportsVision(),
                is(true));
        assertThat(OpenAiServerCli.parse("-m", "m.gguf").toServerConfig().isSupportsVision(), is(false));
    }

    @Test
    public void loadedModelVisionCapabilityOverridesMmprojHint() {
        OpenAiServerCli.Options options = OpenAiServerCli.parse("-m", "m.gguf", "--mmproj", "proj.gguf");
        assertThat(options.toServerConfig(false).isSupportsVision(), is(false));
        assertThat(OpenAiServerCli.parse("-m", "m.gguf").toServerConfig(true).isSupportsVision(), is(true));
    }

    @Test
    public void rerankingFlagParsed() {
        assertThat(OpenAiServerCli.parse("-m", "m.gguf", "--reranking").isReranking(), is(true));
        assertThat(OpenAiServerCli.parse("-m", "m.gguf", "--rerank").isReranking(), is(true));
        assertThat(OpenAiServerCli.parse("-m", "m.gguf").isReranking(), is(false));
    }

    @Test
    public void modelIdDerivedFromNestedPath() {
        OpenAiServerCli.Options options = OpenAiServerCli.parse("-m", "/opt/models/Llama-3.gguf");
        assertThat(options.getModelId(), is("Llama-3.gguf"));
    }

    @Test
    public void explicitModelIdOverridesDerivation() {
        OpenAiServerCli.Options options = OpenAiServerCli.parse("-m", "/opt/models/Llama-3.gguf", "--model-id", "x");
        assertThat(options.getModelId(), is("x"));
    }

    @Test
    public void missingModelThrows() {
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, OpenAiServerCli::parse);
        assertThat(ex.getMessage(), containsString("--model"));
    }

    @Test
    public void unknownFlagThrows() {
        IllegalArgumentException ex =
                assertThrows(IllegalArgumentException.class, () -> OpenAiServerCli.parse("-m", "m.gguf", "--bogus"));
        assertThat(ex.getMessage(), containsString("Unknown argument: --bogus"));
    }

    @Test
    public void missingValueThrows() {
        IllegalArgumentException ex =
                assertThrows(IllegalArgumentException.class, () -> OpenAiServerCli.parse("-m", "m.gguf", "--port"));
        assertThat(ex.getMessage(), containsString("Missing value for --port"));
    }

    @Test
    public void nonIntegerPortThrows() {
        IllegalArgumentException ex = assertThrows(
                IllegalArgumentException.class, () -> OpenAiServerCli.parse("-m", "m.gguf", "--port", "abc"));
        assertThat(ex.getMessage(), containsString("expects an integer"));
    }

    @Test
    public void helpRequestedDetection() {
        assertThat(OpenAiServerCli.isHelpRequested("-h"), is(true));
        assertThat(OpenAiServerCli.isHelpRequested("--help"), is(true));
        assertThat(OpenAiServerCli.isHelpRequested("--model", "m.gguf"), is(false));
    }

    @Test
    public void usageMentionsEndpointsAndRequiredFlag() {
        String usage = OpenAiServerCli.usage();
        assertThat(usage, containsString("--model"));
        assertThat(usage, containsString("/v1/chat/completions"));
        assertThat(usage, containsString("/v1/embeddings"));
        assertThat(usage, containsString("/health"));
    }

    @Test
    public void serverConfigCarriesHostPortAndModelId() {
        OpenAiServerConfig config = OpenAiServerCli.parse("-m", "m.gguf", "--host", "0.0.0.0", "-p", "1234")
                .toServerConfig();
        assertThat(config.getHost(), is("0.0.0.0"));
        assertThat(config.getPort(), is(1234));
        assertThat(config.getModelId(), is("m.gguf"));
        assertThat(config.isAuthenticationEnabled(), is(false));
    }

    @Test
    public void apiKeyEnablesAuthInServerConfig() {
        OpenAiServerConfig config =
                OpenAiServerCli.parse("-m", "m.gguf", "--api-key", "secret").toServerConfig();
        assertThat(config.isAuthenticationEnabled(), is(true));
        assertThat(config.getApiKey(), is("secret"));
    }

    @Test
    public void contextSizeDerivesAdvertisedTokenBudgets() {
        // ctx 8192 -> output capped at the 2048 default, input = 8192 - 2048 = 6144 (README example).
        OpenAiServerConfig config =
                OpenAiServerCli.parse("-m", "m.gguf", "-c", "8192").toServerConfig();
        assertThat(config.getMaxOutputTokens(), is(2048));
        assertThat(config.getMaxInputTokens(), is(6144));
    }

    @Test
    public void modelParametersIncludeModelPath() {
        String json =
                OpenAiServerCli.parse("-m", "models/m.gguf").toModelParameters().toString();
        assertThat(json, containsString("models/m.gguf"));
    }

    @Test
    public void tuningFlagsDefaultToSentinels() {
        OpenAiServerCli.Options options = OpenAiServerCli.parse("-m", "m.gguf");
        assertThat(options.getBatchSize(), is(0));
        assertThat(options.getUbatchSize(), is(0));
        assertThat(options.getThreadsBatch(), is(0));
        assertThat(options.getCacheTypeK(), is((CacheType) null));
        assertThat(options.getCacheTypeV(), is((CacheType) null));
        assertThat(options.isJinja(), is(false));
        assertThat(options.getChatTemplateKwargs(), is((Object) null));
    }

    @Test
    public void tuningShortFlagsParsed() {
        OpenAiServerCli.Options options = OpenAiServerCli.parse(
                "-m", "m.gguf", "-b", "4096", "-ub", "2048", "-tb", "16", "-ctk", "q8_0", "-ctv", "q8_0");
        assertThat(options.getBatchSize(), is(4096));
        assertThat(options.getUbatchSize(), is(2048));
        assertThat(options.getThreadsBatch(), is(16));
        assertThat(options.getCacheTypeK(), is(CacheType.Q8_0));
        assertThat(options.getCacheTypeV(), is(CacheType.Q8_0));
    }

    @Test
    public void tuningLongFlagsParsed() {
        OpenAiServerCli.Options options = OpenAiServerCli.parse(
                "-m",
                "m.gguf",
                "--batch-size",
                "512",
                "--ubatch-size",
                "256",
                "--threads-batch",
                "6",
                "--cache-type-k",
                "f16",
                "--cache-type-v",
                "q4_0",
                "--jinja");
        assertThat(options.getBatchSize(), is(512));
        assertThat(options.getUbatchSize(), is(256));
        assertThat(options.getThreadsBatch(), is(6));
        assertThat(options.getCacheTypeK(), is(CacheType.F16));
        assertThat(options.getCacheTypeV(), is(CacheType.Q4_0));
        assertThat(options.isJinja(), is(true));
    }

    @Test
    public void cacheTypeIsCaseInsensitive() {
        OpenAiServerCli.Options options = OpenAiServerCli.parse("-m", "m.gguf", "-ctk", "Q8_0");
        assertThat(options.getCacheTypeK(), is(CacheType.Q8_0));
    }

    @Test
    public void unknownCacheTypeThrows() {
        IllegalArgumentException ex = assertThrows(
                IllegalArgumentException.class, () -> OpenAiServerCli.parse("-m", "m.gguf", "-ctk", "q3_k"));
        assertThat(ex.getMessage(), containsString("expects one of"));
        assertThat(ex.getMessage(), containsString("q8_0"));
        assertThat(ex.getMessage(), containsString("q3_k"));
    }

    @Test
    public void chatTemplateKwargsParsedToRawJsonValues() {
        OpenAiServerCli.Options options = OpenAiServerCli.parse(
                "-m", "m.gguf", "--chat-template-kwargs", "{\"reasoning_effort\":\"low\",\"enable_thinking\":true}");
        assertThat(options.getChatTemplateKwargs().get("reasoning_effort"), is("\"low\""));
        assertThat(options.getChatTemplateKwargs().get("enable_thinking"), is("true"));
    }

    @Test
    public void chatTemplateKwargsInvalidJsonThrows() {
        IllegalArgumentException ex = assertThrows(
                IllegalArgumentException.class,
                () -> OpenAiServerCli.parse("-m", "m.gguf", "--chat-template-kwargs", "{not json"));
        assertThat(ex.getMessage(), containsString("--chat-template-kwargs expects a JSON object"));
    }

    @Test
    public void chatTemplateKwargsNonObjectThrows() {
        IllegalArgumentException ex = assertThrows(
                IllegalArgumentException.class,
                () -> OpenAiServerCli.parse("-m", "m.gguf", "--chat-template-kwargs", "\"low\""));
        assertThat(ex.getMessage(), containsString("--chat-template-kwargs expects a JSON object"));
    }

    @Test
    public void toModelParametersCarriesTuningFlags() {
        String argv = OpenAiServerCli.parse(
                        "-m",
                        "m.gguf",
                        "-b",
                        "4096",
                        "-ub",
                        "2048",
                        "-tb",
                        "16",
                        "-ctk",
                        "q8_0",
                        "-ctv",
                        "q8_0",
                        "--jinja",
                        "--chat-template-kwargs",
                        "{\"reasoning_effort\":\"low\"}")
                .toModelParameters()
                .toString();
        assertThat(argv, containsString("--batch-size 4096"));
        assertThat(argv, containsString("--ubatch-size 2048"));
        assertThat(argv, containsString("--threads-batch 16"));
        assertThat(argv, containsString("--cache-type-k q8_0"));
        assertThat(argv, containsString("--cache-type-v q8_0"));
        assertThat(argv, containsString("--jinja"));
        assertThat(argv, containsString("--chat-template-kwargs"));
        assertThat(argv, containsString("reasoning_effort"));
    }

    @Test
    public void usageMentionsNewTuningFlags() {
        String usage = OpenAiServerCli.usage();
        assertThat(usage, containsString("--batch-size"));
        assertThat(usage, containsString("--ubatch-size"));
        assertThat(usage, containsString("--threads-batch"));
        assertThat(usage, containsString("--cache-type-k"));
        assertThat(usage, containsString("--jinja"));
        assertThat(usage, containsString("--chat-template-kwargs"));
    }
}
