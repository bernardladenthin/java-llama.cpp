// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

/**
 * The speech pipeline contract for Chatterbox (llama.cpp b12097), CI's multilingual pair
 * ({@link TestConstants#DEFAULT_CHATTERBOX_MODEL_PATH}).
 */
public class ChatterboxIntegrationTest extends AbstractTtsPipelineContract {

    @Override
    protected String modelPath() {
        return TestConstants.resolveModelProperty(
                TestConstants.PROP_CHATTERBOX_MODEL, TestConstants.DEFAULT_CHATTERBOX_MODEL_PATH);
    }

    @Override
    protected String mmprojPath() {
        return TestConstants.resolveModelProperty(
                TestConstants.PROP_CHATTERBOX_MMPROJ, TestConstants.DEFAULT_CHATTERBOX_MMPROJ_PATH);
    }
}
