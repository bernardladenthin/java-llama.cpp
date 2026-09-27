// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.emptyOrNullString;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.File;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import net.ladenthin.llama.args.LogFormat;
import net.ladenthin.llama.exception.LlamaException;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.parameters.ModelParameters;
import org.junit.jupiter.api.Test;

/**
 * A real model over RPC, end to end in one JVM: an {@link RpcServer} on loopback serves this
 * process's own devices, and a {@link LlamaModel} loaded with {@code --rpc} offloads its layers
 * there. Uses the small cached draft model; self-skips without it or without libjllama.
 */
@ClaudeGenerated(
        purpose = "Prove layers really run on an RPC server (the load log names the RPC buffer), that "
                + "generation works through it, and that the RPC device a previous model registered is "
                + "kept out of a later load without --rpc -- ggml's registry is process-wide and never "
                + "forgets, so without that filter the second load would offload to a stopped server "
                + "and abort the JVM.")
public class RpcIntegrationTest {

    private static final InferenceParameters PROMPT =
            new InferenceParameters("def fibonacci(n):").withNPredict(8).withTemperature(0.0f);

    @Test
    public void aModelRunsOnAnRpcServerAndALaterLoadWithoutRpcIgnoresIt() throws Exception {
        assumeTrue(RpcServerTest.nativeLibraryOnClasspath(), "libjllama not on classpath");
        assumeTrue(new File(TestConstants.DRAFT_MODEL_PATH).exists(), "draft model not found");

        List<String> log = new CopyOnWriteArrayList<>();
        String viaRpc;
        String endpoint;
        // CPU, not the default choice: ggml-rpc's client treats every operation as supported, so a
        // GPU that cannot run one (the paravirtual Metal device of the macOS CI runners has no
        // MUL_MAT) would abort this JVM. Which device serves is not what this test is about.
        try (RpcServer server =
                RpcServer.startLocal(RpcServerTest.freePort(), 2, null, Collections.singletonList("CPU"))) {
            endpoint = server.getEndpoint().toString();
            LlamaModel.setLogger(LogFormat.TEXT, (level, text) -> log.add(text));
            try (LlamaModel model = new LlamaModel(new ModelParameters()
                    .setModel(TestConstants.DRAFT_MODEL_PATH)
                    .setCtxSize(256)
                    .setGpuLayers(99)
                    .setFit(false)
                    .setLogVerbosity(4)
                    .setRpcServers(server.getEndpoint()))) {
                viaRpc = model.complete(PROMPT);
            } finally {
                LlamaModel.setLogger(LogFormat.TEXT, null);
            }
        }
        assertThat(viaRpc, is(not(emptyOrNullString())));
        assertThat(
                "the load log must show a model buffer on the RPC server " + endpoint + ", got:\n"
                        + String.join("", log),
                log.stream().anyMatch(line -> line.contains("model buffer size") && line.contains(endpoint)),
                is(true));

        // The server is gone now, but its device is still in ggml's registry. A load that does not
        // ask for it must not touch it -- not even at -lv 4, where common_init() queries the memory
        // of every registered device (patches/0015 makes that query report 0/0 for a gone server).
        try (LlamaModel local = new LlamaModel(new ModelParameters()
                .setModel(TestConstants.DRAFT_MODEL_PATH)
                .setCtxSize(256)
                .setGpuLayers(Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL))
                .setFit(false)
                .setLogVerbosity(4))) {
            assertThat(local.complete(PROMPT), is(not(emptyOrNullString())));
        }

        // TextToSpeech builds its common_params itself instead of parsing an argv, so the argv guard
        // above never sees it. Before exclude_stale_devices() this load put the gone server into the
        // backbone's device list and aborted the JVM while fitting the context (CI: this test, then
        // TtsIntegrationTest). The draft model loads as a backbone; the missing mmproj then fails the
        // load the ordinary way -- with an exception, in a JVM that is still alive.
        LlamaException e = assertThrows(
                LlamaException.class,
                () -> new TextToSpeech(TestConstants.DRAFT_MODEL_PATH, "no-such-mmproj.gguf").close());
        assertThat(e.getMessage(), containsString("no-such-mmproj.gguf"));
    }
}
