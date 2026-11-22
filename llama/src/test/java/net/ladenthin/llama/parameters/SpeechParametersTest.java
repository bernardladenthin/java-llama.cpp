// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.parameters;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.nullValue;

import org.junit.jupiter.api.Test;

public class SpeechParametersTest {

    @Test
    void theDefaultsAreWhatTheShortSynthesizeOverloadsPass() {
        SpeechParameters defaults = SpeechParameters.defaults();
        assertThat(defaults.getSpeakerReferenceAudioPath(), is(nullValue()));
        assertThat(defaults.getLanguage(), is(nullValue()));
        assertThat(defaults.getVoice(), is(nullValue()));
        assertThat(defaults.getMaxFrames(), is(512));
        assertThat(defaults.getTopK(), is(4));
        assertThat(defaults.getSeed(), is(0));
        assertThat(defaults.getTemperature(), is(-1.0f));
        assertThat(defaults.getTopP(), is(1.0f));
        assertThat(defaults.getRepeatPenalty(), is(1.0f));
        assertThat(defaults.getSpeed(), is(1.0f));
        assertThat(defaults.getLoudness(), is(0.0f));
        assertThat(defaults.getSpeakerReferenceText(), is(nullValue()));
        assertThat(defaults.getCfgScale(), is(0.0f));
        assertThat(defaults.getFlowSteps(), is(0));
        assertThat(defaults.isRawPrompt(), is(false));
    }

    @Test
    void aRawPromptIsKeptByACopy() {
        SpeechParameters raw = SpeechParameters.builder().rawPrompt(true).build();
        assertThat(raw.toBuilder().seed(2).build().isRawPrompt(), is(true));
        assertThat(raw.toString(), containsString("rawPrompt=true"));
    }

    @Test
    void theGuidanceAndSolverStepsAreKeptByACopy() {
        SpeechParameters song =
                SpeechParameters.builder().cfgScale(1.5f).flowSteps(16).build();
        SpeechParameters copy = song.toBuilder().seed(2).build();
        assertThat(copy.getCfgScale(), is(1.5f));
        assertThat(copy.getFlowSteps(), is(16));
        assertThat(copy.toString(), containsString("flowSteps=16"));
    }

    @Test
    void theTranscriptOfTheReferenceIsKeptByACopy() {
        SpeechParameters transcribed = SpeechParameters.builder()
                .speakerReferenceText("What the clip says.")
                .build();
        assertThat(transcribed.getSpeakerReferenceText(), is("What the clip says."));
        assertThat(transcribed.toBuilder().seed(2).build().getSpeakerReferenceText(), is("What the clip says."));
    }

    @Test
    void aCopyChangesOnlyWhatItSets() {
        SpeechParameters base = SpeechParameters.builder().voice("alba").seed(3).build();
        SpeechParameters faster = base.toBuilder().speed(1.5f).build();
        assertThat(faster.getVoice(), is("alba"));
        assertThat(faster.getSeed(), is(3));
        assertThat(faster.getSpeed(), is(1.5f));
        assertThat(base.getSpeed(), is(1.0f));
        assertThat(faster.toString(), containsString("speed=1.5"));
    }
}
