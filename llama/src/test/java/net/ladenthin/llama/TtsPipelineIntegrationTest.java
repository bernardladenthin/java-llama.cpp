// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

/**
 * The speech pipeline contract for a pair outside CI's model set (Drama-3: 14.6 GB; Soprano):
 * self-skips unless {@link TestConstants#PROP_TTS_PIPELINE_MODEL} and
 * {@link TestConstants#PROP_TTS_PIPELINE_MMPROJ} name one.
 */
public class TtsPipelineIntegrationTest extends AbstractTtsPipelineContract {

    @Override
    protected String modelPath() {
        return TestConstants.resolveModelProperty(TestConstants.PROP_TTS_PIPELINE_MODEL);
    }

    @Override
    protected String mmprojPath() {
        return TestConstants.resolveModelProperty(TestConstants.PROP_TTS_PIPELINE_MMPROJ);
    }
}
