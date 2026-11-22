// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.emptyArray;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;
import static org.hamcrest.Matchers.sameInstance;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicInteger;
import net.ladenthin.llama.callback.LoadProgressCallback;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.parameters.ModelParameters;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * A Java callback that throws inside a native call. The JNI trampoline calls the callback with
 * {@code CallBooleanMethod}; a pending exception reads as {@code false} there, which cancels the
 * native work, and the native side then leaves the pending exception in place instead of throwing
 * its own over it. So the caller gets <em>the callback's exception itself</em> -- not a
 * {@link net.ladenthin.llama.exception.LlamaException} wrapping it -- and the callback is not called
 * again while that exception is pending (a JNI call with a pending exception is undefined behaviour).
 */
@ClaudeGenerated(
        purpose = "Pin what reaches the caller when a LoadProgressCallback throws during a real model load "
                + "(the callback's own exception, unwrapped, with no further callback call after it) and during "
                + "LlamaQuantizer's importance-matrix computation, and that the next load and computation in the "
                + "same JVM still work.")
public class ThrowingCallbackIntegrationTest {

    @TempDir
    static Path tempDir;

    @BeforeAll
    public static void assumeDraftModel() {
        Assumptions.assumeTrue(
                new File(TestConstants.DRAFT_MODEL_PATH).exists(),
                "Draft model not found, skipping ThrowingCallbackIntegrationTest");
    }

    private static ModelParameters draftParameters() {
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        return new ModelParameters()
                .setModel(TestConstants.DRAFT_MODEL_PATH)
                .setCtxSize(128)
                .setGpuLayers(gpuLayers)
                .setFit(false);
    }

    /** A callback that throws {@code thrown} on its {@code throwOnCall}-th call and counts every call. */
    private static LoadProgressCallback throwingOnCall(
            int throwOnCall, IllegalStateException thrown, AtomicInteger calls) {
        return progress -> {
            if (calls.incrementAndGet() == throwOnCall) {
                throw thrown;
            }
            return true;
        };
    }

    /** The draft model loads and answers: what a load after the failed one must still do. */
    private static void assertTheDraftModelLoadsAndAnswers() {
        AtomicInteger progressCalls = new AtomicInteger();
        try (LlamaModel model = new LlamaModel(draftParameters(), progress -> {
            progressCalls.incrementAndGet();
            return true;
        })) {
            String answer = model.complete(new InferenceParameters("def main():").withNPredict(4));
            assertThat(answer, is(not((String) null)));
        }
        assertThat("the next load reports its progress again", progressCalls.get() > 0, is(true));
    }

    @Test
    public void aLoadProgressCallbackThrowingOnItsFirstCallReachesTheCallerItself() {
        // arrange
        IllegalStateException thrown = new IllegalStateException("thrown by the progress callback");
        AtomicInteger calls = new AtomicInteger();

        // act
        IllegalStateException caught = assertThrows(
                IllegalStateException.class,
                () -> new LlamaModel(draftParameters(), throwingOnCall(1, thrown, calls)).close());

        // assert
        assertThat(caught, is(sameInstance(thrown)));
        assertThat(caught.getSuppressed(), is(emptyArray()));
        assertThat("no call after the one that threw", calls.get(), is(1));
        assertTheDraftModelLoadsAndAnswers();
    }

    @Test
    public void aLoadProgressCallbackThrowingDuringTheLoadReachesTheCallerItself() {
        // arrange: count the calls of a whole load, then throw at one in the middle of it
        AtomicInteger total = new AtomicInteger();
        try (LlamaModel model = new LlamaModel(draftParameters(), progress -> {
            total.incrementAndGet();
            return true;
        })) {
            // loaded
        }
        Assumptions.assumeTrue(total.get() >= 3, "the load reports too few progress steps: " + total.get());
        int middle = total.get() / 2 + 1;
        IllegalStateException thrown = new IllegalStateException("thrown in the middle of the load");
        AtomicInteger calls = new AtomicInteger();

        // act
        IllegalStateException caught = assertThrows(
                IllegalStateException.class,
                () -> new LlamaModel(draftParameters(), throwingOnCall(middle, thrown, calls)).close());

        // assert
        assertThat(caught, is(sameInstance(thrown)));
        assertThat("no call after the one that threw", calls.get(), is(middle));
        assertTheDraftModelLoadsAndAnswers();
    }

    @Test
    public void anImportanceMatrixProgressCallbackThatThrowsReachesTheCallerItself() throws Exception {
        // arrange
        Path text = tempDir.resolve("calibration.txt");
        StringBuilder words = new StringBuilder();
        for (int i = 0; i < 400; i++) {
            words.append("def add(a, b):\n    return a + b\n\nfor value in range(")
                    .append(i)
                    .append("):\n    print(add(value, 1))\n\n");
        }
        Files.write(text, words.toString().getBytes(StandardCharsets.UTF_8));
        Path failed = tempDir.resolve("failed.imatrix.gguf");
        IllegalStateException thrown = new IllegalStateException("thrown by the imatrix callback");
        AtomicInteger calls = new AtomicInteger();

        // act
        IllegalStateException caught = assertThrows(
                IllegalStateException.class,
                () -> LlamaQuantizer.computeImportanceMatrix(
                        TestConstants.DRAFT_MODEL_PATH,
                        text.toString(),
                        failed.toString(),
                        64,
                        4,
                        0,
                        0,
                        throwingOnCall(2, thrown, calls)));

        // assert
        assertThat(caught, is(sameInstance(thrown)));
        assertThat("no call after the one that threw", calls.get(), is(2));
        assertThat("a run its callback stopped left a file behind", Files.exists(failed), is(false));
        Path imatrix = tempDir.resolve("next.imatrix.gguf");
        AtomicInteger nextCalls = new AtomicInteger();
        int chunks = LlamaQuantizer.computeImportanceMatrix(
                TestConstants.DRAFT_MODEL_PATH, text.toString(), imatrix.toString(), 64, 4, 0, 0, fraction -> {
                    nextCalls.incrementAndGet();
                    return true;
                });
        assertThat(chunks, is(4));
        assertThat(nextCalls.get(), is(4));
        assertThat(Files.size(imatrix) > 0, is(true));
        assertTheDraftModelLoadsAndAnswers();
    }
}
