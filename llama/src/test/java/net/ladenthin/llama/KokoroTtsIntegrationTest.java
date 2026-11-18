// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

/**
 * The speech pipeline contract for Kokoro-82M, CI's pair
 * ({@link TestConstants#DEFAULT_KOKORO_MODEL_PATH}): a PL-BERT phoneme encoder and a StyleTTS2 decoder that write
 * the whole utterance from the prompt in one pass, not frame by frame (llama.cpp b12494). Its English voices take
 * the text through the model's {@code tts} template.
 */
public class KokoroTtsIntegrationTest extends AbstractTtsPipelineContract {

    @Override
    protected String modelPath() {
        return TestConstants.resolveModelProperty(
                TestConstants.PROP_KOKORO_MODEL, TestConstants.DEFAULT_KOKORO_MODEL_PATH);
    }

    @Override
    protected String mmprojPath() {
        return TestConstants.resolveModelProperty(
                TestConstants.PROP_KOKORO_MMPROJ, TestConstants.DEFAULT_KOKORO_MMPROJ_PATH);
    }
}
