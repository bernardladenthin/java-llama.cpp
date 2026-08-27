// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;

import java.io.File;
import java.nio.file.Path;
import java.util.List;
import net.ladenthin.llama.parameters.ModelParameters;
import net.ladenthin.llama.value.ChatMessage;
import net.ladenthin.llama.value.SessionCheckpoint;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@ClaudeGenerated(
        purpose = "Real-model coverage for the Session fork/rewind API: checkpoint + rewind "
                + "restores both the KV slot state and the transcript atomically and the "
                + "conversation continues from the branch point; fork produces an independent "
                + "session on a second slot carrying the same transcript; guard rails "
                + "(fork onto the own slot) fail fast.")
public class SessionForkRewindIntegrationTest {

    private static LlamaModel model;

    @TempDir
    static Path tempDir;

    @BeforeAll
    public static void loadModel() {
        String modelPath = TestConstants.resolveModelProperty(
                "net.ladenthin.llama.model.path", TestConstants.REASONING_MODEL_PATH);
        Assumptions.assumeTrue(new File(modelPath).exists(), "Model missing: " + modelPath);
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        model = new LlamaModel(new ModelParameters()
                .setModel(modelPath)
                .setCtxSize(2048)
                .setGpuLayers(gpuLayers)
                // Two slots: slot 0 hosts the primary session, slot 1 receives the fork.
                .setParallel(2));
    }

    @AfterAll
    public static void closeModel() {
        if (model != null) {
            model.close();
        }
    }

    /**
     * Thinking is suppressed with {@code reasoning_budget_tokens=0}, and that is load-bearing
     * rather than cosmetic. The fixture model is Qwen3-0.6B, whose chat template injects
     * {@code <think>} into the prompt, so the model reasons on every turn; llama.cpp's
     * {@code common_params.reasoning_format} defaults to {@code DEEPSEEK}, which strips the
     * {@code <think>…</think>} block out of {@code content} and into {@code reasoning_content}.
     * {@link Session#send(String)} returns {@code choices[0].message.content} only. Qwen3-0.6B
     * spends roughly 200 tokens thinking before it answers (see {@code ReasoningBudgetTest}, which
     * budgets 1500 for exactly this reason), so with the 24-token budget this test wants for speed
     * every token is consumed inside the thinking block and {@code content} comes back <em>empty</em>
     * — deterministically, on every platform. Suppressing thinking makes the model answer directly,
     * which is what these assertions are actually about: this class tests session state
     * (checkpoint/rewind/fork), not reasoning. Raising the budget instead would work but costs
     * ~200 extra tokens per turn across five turns for no added coverage.
     */
    private static Session newSession(int slotId) {
        return new Session(
                model,
                slotId,
                "You are terse.",
                p -> p.withNPredict(24).withTemperature(0.0f).withSeed(42).withReasoningBudgetTokens(0));
    }

    @Test
    public void rewindRestoresTranscriptAndConversationContinues() {
        try (Session session = newSession(0)) {
            session.send("Say OK.");
            List<ChatMessage> atCheckpoint = session.getMessages();
            SessionCheckpoint checkpoint =
                    session.checkpoint(tempDir.resolve("rewind.bin").toString());

            session.send("Say MORE.");
            assertThat(session.getMessages().size(), is(atCheckpoint.size() + 2));

            session.rewind(checkpoint);

            // Transcript is back at the branch point...
            assertThat(session.getMessages(), is(atCheckpoint));
            // ...and the session is fully usable from there (KV state restored with it).
            String retried = session.send("Say YES.");
            assertThat(retried.isEmpty(), is(false));
            assertThat(session.getMessages().size(), is(atCheckpoint.size() + 2));
        }
    }

    @Test
    public void forkCreatesIndependentSessionWithSameTranscript() {
        try (Session original = newSession(0)) {
            original.send("Say OK.");

            try (Session forked = original.fork(1, tempDir.resolve("fork.bin").toString())) {
                // The fork starts as an exact transcript copy...
                assertThat(forked.getMessages(), is(original.getMessages()));

                // ...and both continue independently from the branch point.
                String forkedReply = forked.send("Say A.");
                String originalReply = original.send("Say B.");
                assertThat(forkedReply.isEmpty(), is(false));
                assertThat(originalReply.isEmpty(), is(false));
                assertThat(forked.getMessages(), is(not(original.getMessages())));
                assertThat(
                        forked.getMessages().size(), is(original.getMessages().size()));
            }
        }
    }

    @Test
    public void forkOntoOwnSlotFailsFast() {
        try (Session session = newSession(0)) {
            org.junit.jupiter.api.Assertions.assertThrows(
                    IllegalArgumentException.class,
                    () -> session.fork(0, tempDir.resolve("self.bin").toString()));
        }
    }
}
