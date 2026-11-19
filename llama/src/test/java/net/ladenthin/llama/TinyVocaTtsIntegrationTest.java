// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

/**
 * The speech pipeline contract for tiny-voca, CI's pair
 * ({@link TestConstants#DEFAULT_TINY_VOCA_MODEL_PATH}): a 4.5M-parameter phoneme encoder, a duration predictor
 * and a source-filter synthesizer that write the whole utterance from the prompt (llama.cpp b12497); its
 * {@code tts} template asks the projector's letter-to-sound model for words its lexicon lacks.
 */
public class TinyVocaTtsIntegrationTest extends AbstractTtsPipelineContract {

    @Override
    protected String modelPath() {
        return TestConstants.resolveModelProperty(
                TestConstants.PROP_TINY_VOCA_MODEL, TestConstants.DEFAULT_TINY_VOCA_MODEL_PATH);
    }

    @Override
    protected String mmprojPath() {
        return TestConstants.resolveModelProperty(
                TestConstants.PROP_TINY_VOCA_MMPROJ, TestConstants.DEFAULT_TINY_VOCA_MMPROJ_PATH);
    }

    @Override
    protected Boolean clonesVoice() {
        // tiny-voca speaks only with its predefined voices
        return false;
    }
}
