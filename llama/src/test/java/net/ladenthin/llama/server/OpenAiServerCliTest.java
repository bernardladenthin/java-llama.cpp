// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertThrows;

import net.ladenthin.llama.args.CacheType;
import net.ladenthin.llama.value.RpcEndpoint;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

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
    public void rpcFlagParsesAListAndIsRepeatable() {
        OpenAiServerCli.Options options =
                OpenAiServerCli.parse("-m", "m.gguf", "--rpc", "10.0.0.2:50052,10.0.0.3:1", "--rpc", "host:2");
        assertThat(
                options.getRpcServers(),
                is(java.util.Arrays.asList(
                        RpcEndpoint.of("10.0.0.2", 50052), RpcEndpoint.of("10.0.0.3", 1), RpcEndpoint.of("host", 2))));
        assertThat(OpenAiServerCli.parse("-m", "m.gguf").getRpcServers().isEmpty(), is(true));
    }

    @Test
    public void rpcFlagReachesModelParameters() {
        assertThat(
                OpenAiServerCli.parse("-m", "m.gguf", "--rpc", "10.0.0.2:50052")
                        .toModelParameters()
                        .toString(),
                containsString("--rpc"));
        assertThat(
                OpenAiServerCli.parse("-m", "m.gguf")
                        .toModelParameters()
                        .toString()
                        .contains("--rpc"),
                is(false));
    }

    @Test
    public void rpcFlagWithAnInvalidEndpointNamesTheFlag() {
        IllegalArgumentException e = assertThrows(
                IllegalArgumentException.class, () -> OpenAiServerCli.parse("-m", "m.gguf", "--rpc", "no-port"));
        assertThat(e.getMessage(), containsString("--rpc"));
        assertThat(e.getMessage(), containsString("no port"));
    }

    @Test
    public void rpcFlagIsInTheUsage() {
        assertThat(OpenAiServerCli.usage(), containsString("--rpc"));
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

    /**
     * A port outside the TCP range is a usage error like a non-numeric one. Accepted here, it only
     * fails when the server binds, which happens after the model has loaded, and as a stack trace.
     */
    @ParameterizedTest
    @ValueSource(strings = {"-1", "65536", "70000"})
    public void portOutsideTheTcpRangeThrows(String port) {
        IllegalArgumentException ex = assertThrows(
                IllegalArgumentException.class, () -> OpenAiServerCli.parse("-m", "m.gguf", "--port", port));
        assertThat(ex.getMessage(), containsString("--port expects a port in 0..65535"));
    }

    @ParameterizedTest
    @ValueSource(ints = {0, 1, 65535})
    public void portAtTheEndsOfTheTcpRangeIsAccepted(int port) {
        assertThat(
                OpenAiServerCli.parse("-m", "m.gguf", "--port", Integer.toString(port))
                        .getPort(),
                is(port));
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
    public void loadedSlotContextIsAdvertisedWithoutContextSize() {
        // Without -c the advertised window was the fixed 8192 + 2048, whatever the model loaded with:
        // a slot of 2048 tokens was advertised as 10240 (measured at b11639).
        OpenAiServerConfig config = OpenAiServerCli.parse("-m", "m.gguf").toServerConfig(false, "", 2048);
        assertThat(config.getMaxOutputTokens(), is(1024));
        assertThat(config.getMaxInputTokens() + config.getMaxOutputTokens(), is(2048));
    }

    @Test
    public void loadedSlotContextWinsOverTheContextSizeArgument() {
        // -c 16384 with 4 slots would be 4096 per slot, but the load can end with less (llama.cpp's
        // --fit shrinks the context to the device memory); what the slot holds is what a request gets.
        OpenAiServerConfig config = OpenAiServerCli.parse("-m", "m.gguf", "-c", "16384", "--parallel", "4")
                .toServerConfig(false, "", 3000);
        assertThat(config.getMaxOutputTokens(), is(1500));
        assertThat(config.getMaxInputTokens(), is(1500));
    }

    @Test
    public void slotCountOfTheLoadedModelIsAdvertised() {
        // Without --parallel llama.cpp picks the slot count itself, so the loaded model is the only
        // source; /props reported a constant 1.
        assertThat(
                OpenAiServerCli.parse("-m", "m.gguf")
                        .toServerConfig(false, "", 0, 4)
                        .getTotalSlots(),
                is(4));
    }

    @Test
    public void unknownSlotCountFallsBackToTheParallelArgumentAndThenToOne() {
        assertThat(
                OpenAiServerCli.parse("-m", "m.gguf", "--parallel", "3")
                        .toServerConfig()
                        .getTotalSlots(),
                is(3));
        assertThat(OpenAiServerCli.parse("-m", "m.gguf").toServerConfig().getTotalSlots(), is(1));
    }

    @Test
    public void unknownSlotContextFallsBackToTheContextSizeArgument() {
        OpenAiServerConfig config =
                OpenAiServerCli.parse("-m", "m.gguf", "-c", "8192").toServerConfig(false, "", 0);
        assertThat(config.getMaxOutputTokens(), is(2048));
        assertThat(config.getMaxInputTokens(), is(6144));
    }

    @Test
    public void smallContextSizeSplitsTheWholeWindowBetweenInputAndOutput() {
        // Below twice the output default the output budget is ctx/2. The input budget must be the rest
        // of the window, not ctx minus the 2048 default: with -c 2048 that advertised a single input
        // token, and clients that size their prompt from it (Copilot's maxInputTokens) send none.
        for (int ctx : new int[] {1024, 2048, 3000}) {
            OpenAiServerConfig config = OpenAiServerCli.parse("-m", "m.gguf", "-c", String.valueOf(ctx))
                    .toServerConfig();
            assertThat(config.getMaxOutputTokens(), is(ctx / 2));
            assertThat(config.getMaxInputTokens() + config.getMaxOutputTokens(), is(ctx));
        }
    }

    @Test
    public void parallelSlotsShareTheContextSoTheBudgetIsPerSlot() {
        // With an explicit --parallel N (and no unified KV cache, which llama.cpp enables only when the
        // slot count is "auto") every slot gets n_ctx / N, and upstream's /props reports that per-slot
        // size. Advertising the whole -c made clients size prompts for 8192 tokens a request cannot hold.
        OpenAiServerConfig config = OpenAiServerCli.parse("-m", "m.gguf", "-c", "8192", "--parallel", "4")
                .toServerConfig();
        assertThat(config.getMaxInputTokens() + config.getMaxOutputTokens(), is(2048));
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
