// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

/**
 * The speech pipeline contract for Pocket TTS without voice cloning, CI's pair
 * ({@link TestConstants#DEFAULT_POCKET_TTS_MODEL_PATH}): the one pipeline in the set that ships
 * predefined voices in its projector and reads the audio frame temperature.
 */
public class PocketTtsIntegrationTest extends AbstractTtsPipelineContract {

    @Override
    protected String modelPath() {
        return TestConstants.resolveModelProperty(
                TestConstants.PROP_POCKET_TTS_MODEL, TestConstants.DEFAULT_POCKET_TTS_MODEL_PATH);
    }

    @Override
    protected String mmprojPath() {
        return TestConstants.resolveModelProperty(
                TestConstants.PROP_POCKET_TTS_MMPROJ, TestConstants.DEFAULT_POCKET_TTS_MMPROJ_PATH);
    }
}
