// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.concurrent.TimeUnit;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.parameters.ModelParameters;
import net.ladenthin.llama.value.ChatMessage;
import net.ladenthin.llama.value.ContentPart;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/**
 * Real-model coverage for <b>audio input</b> (llama.cpp discussion #13759). Loads an audio-capable
 * model (Ultravox / Qwen2.5-Omni) with its audio {@code --mmproj} and sends a multipart message
 * carrying a {@link ContentPart#audioFile(java.nio.file.Path)} clip, exercising:
 * <ul>
 *   <li>{@link ModelParameters#setMmproj(String)} wiring an audio encoder;</li>
 *   <li>{@code ParameterJsonSerializer.buildMessages} emitting the OAI {@code input_audio} part;</li>
 *   <li>the upstream {@code oaicompat_chat_params_parse} routing {@code input_audio} through the
 *       compiled-in {@code mtmd} audio pipeline.</li>
 * </ul>
 *
 * <p>The audio prompt defaults to the committed clip
 * {@link TestConstants#DEFAULT_AUDIO_INPUT_PATH} (override via
 * {@link TestConstants#PROP_AUDIO_PATH}). Self-skips when the model/mmproj properties
 * ({@link TestConstants#PROP_AUDIO_MODEL_PATH} / {@link TestConstants#PROP_AUDIO_MMPROJ_PATH}) are
 * unset or any referenced file is missing, so it runs only where the (large) audio model has been
 * staged.
 */
public class AudioInputIntegrationTest {

    private static LlamaModel model;
    private static String audioPath;

    @BeforeAll
    public static void setup() {
        String modelPath = TestConstants.resolveModelProperty(TestConstants.PROP_AUDIO_MODEL_PATH);
        String mmprojPath = TestConstants.resolveModelProperty(TestConstants.PROP_AUDIO_MMPROJ_PATH);
        audioPath = TestConstants.resolveModelProperty(
                TestConstants.PROP_AUDIO_PATH, TestConstants.DEFAULT_AUDIO_INPUT_PATH);

        Assumptions.assumeTrue(
                modelPath != null && !modelPath.isEmpty(),
                "Audio model path not set (-D" + TestConstants.PROP_AUDIO_MODEL_PATH + "=...)");
        Assumptions.assumeTrue(
                mmprojPath != null && !mmprojPath.isEmpty(),
                "Audio mmproj path not set (-D" + TestConstants.PROP_AUDIO_MMPROJ_PATH + "=...)");
        Assumptions.assumeTrue(new File(modelPath).exists(), "Audio model file missing: " + modelPath);
        Assumptions.assumeTrue(new File(mmprojPath).exists(), "Audio mmproj file missing: " + mmprojPath);
        Assumptions.assumeTrue(new File(audioPath).exists(), "Audio clip missing: " + audioPath);

        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        ModelParameters parameters = new ModelParameters()
                .setCtxSize(4096)
                .setModel(modelPath)
                .setMmproj(mmprojPath)
                .setGpuLayers(gpuLayers)
                .setFit(false);
        if (gpuLayers == 0) {
            parameters.setDevices("none").setMmprojOffload(false);
        }
        model = new LlamaModel(parameters);
        assertTrue(model.supportsAudio(), "loaded model + mmproj must advertise audio input");
    }

    @AfterAll
    public static void tearDown() {
        if (model != null) {
            model.close();
        }
    }

    @Test
    @DisplayName("an input_audio content part reaches the model and yields a non-empty reply")
    @Timeout(value = 240_000, unit = TimeUnit.MILLISECONDS)
    public void audioInputProducesNonEmptyReply() throws IOException {
        ChatMessage message = ChatMessage.userMultimodal(
                ContentPart.text("Transcribe the audio."), ContentPart.audioFile(Paths.get(audioPath)));

        String reply = model.chatCompleteText(InferenceParameters.empty()
                .withMessages(Collections.singletonList(message))
                .withNPredict(64));

        assertFalse(reply.trim().isEmpty(), "reply must be non-empty for an audio prompt; got: \"" + reply + "\"");
    }
}
