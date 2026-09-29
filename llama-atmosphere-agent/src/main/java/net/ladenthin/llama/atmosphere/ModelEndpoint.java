// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.io.PrintStream;
import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.parameters.ModelParameters;
import net.ladenthin.llama.server.OpenAiCompatServer;
import net.ladenthin.llama.server.OpenAiServerConfig;
import org.jspecify.annotations.Nullable;

/**
 * Where the model is served: a server somebody else started ({@code --base-url}), or a GGUF loaded in this
 * JVM and served to the agent over a loopback {@link OpenAiCompatServer} ({@code --model}).
 *
 * <p>Every front end talks to the model the same way — over the OpenAI protocol — so this is the one place
 * that knows the difference.
 */
public final class ModelEndpoint implements AutoCloseable {

    private final String baseUrl;
    private final int contextSize;
    private final @Nullable LlamaModel model;
    private final @Nullable OpenAiCompatServer server;

    private ModelEndpoint(
            String baseUrl, int contextSize, @Nullable LlamaModel model, @Nullable OpenAiCompatServer server) {
        this.baseUrl = baseUrl;
        this.contextSize = contextSize;
        this.model = model;
        this.server = server;
    }

    /**
     * Reach the model the options name, loading it first with {@code --model}.
     *
     * @param options the parsed command line
     * @param err where the loading line goes
     * @return the endpoint
     * @throws Exception when the model cannot be loaded or the loopback server cannot start
     */
    public static ModelEndpoint open(AgentOptions options, PrintStream err) throws Exception {
        if (options.getModelPath() == null) {
            String baseUrl = options.getBaseUrl();
            if (baseUrl == null) {
                throw new IllegalStateException("no endpoint");
            }
            return new ModelEndpoint(baseUrl, ServerProps.contextSize(baseUrl, options.getApiKey()), null, null);
        }
        err.println("Loading " + options.getModelPath() + " (gpu layers: " + options.getGpuLayers() + ", ctx: "
                + options.getCtxSize() + ") ...");
        LlamaModel model = new LlamaModel(modelParameters(options));
        try {
            OpenAiCompatServer server = new OpenAiCompatServer(
                            model,
                            OpenAiServerConfig.builder()
                                    .host("127.0.0.1")
                                    .port(0)
                                    .apiKey(options.getApiKey())
                                    .modelId(options.getModelId())
                                    .build())
                    .start();
            return new ModelEndpoint(
                    "http://127.0.0.1:" + server.getPort() + "/v1", options.getCtxSize(), model, server);
        } catch (Exception | Error e) {
            model.close();
            throw e;
        }
    }

    /**
     * The native parameters for {@code --model}.
     *
     * <p>Visible for tests: the log threshold is the one knob whose effect is only observable on a console,
     * so the test pins the flags that leave here instead.
     *
     * @param options the parsed options
     * @return the parameters the in-process {@link LlamaModel} is loaded with
     */
    static ModelParameters modelParameters(AgentOptions options) {
        ModelParameters parameters = new ModelParameters()
                .setModel(options.getModelPath())
                .setCtxSize(options.getCtxSize())
                .setGpuLayers(options.getGpuLayers())
                .setFit(false)
                // Jinja rendering is what lets the native parser apply the model's tool-call template.
                .enableJinja();
        // llama.cpp logs to stderr, which shares the console with the streamed answer on stdout; the
        // default threshold keeps warnings and errors and drops the per-request INFO lines.
        if (options.isVerbose()) {
            parameters.setVerbose();
        } else {
            parameters.setLogVerbosity(options.getLogVerbosity());
        }
        if (options.getGpuLayers() == 0) {
            parameters.setDevices("none");
        }
        return parameters;
    }

    /**
     * The OpenAI-compatible base URL.
     *
     * @return e.g. {@code http://127.0.0.1:8080/v1}
     */
    public String baseUrl() {
        return baseUrl;
    }

    /**
     * The context window, for the status line and for compaction.
     *
     * @return the size in tokens, or {@link StatusLine#UNKNOWN_CONTEXT}
     */
    public int contextSize() {
        return contextSize;
    }

    /**
     * Whether the model runs in this JVM, i.e. whether its native log is ours to route.
     *
     * @return {@code true} with {@code --model}
     */
    public boolean inProcess() {
        return model != null;
    }

    @Override
    public void close() {
        if (server != null) {
            server.close();
        }
        if (model != null) {
            model.close();
        }
    }
}
