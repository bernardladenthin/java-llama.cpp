// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.value;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import net.ladenthin.llama.ClaudeGenerated;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose = "Verify the SessionCheckpoint value type: getter round-trip, defensive copy "
                + "and unmodifiability of the turn snapshot, the Lombok equals/hashCode "
                + "contract, and the compact handwritten toString.")
public class SessionCheckpointTest {

    private static List<Pair<String, String>> turns() {
        return Arrays.asList(new Pair<>("user", "hi"), new Pair<>("assistant", "hello"));
    }

    private static SessionCheckpoint sample() {
        return new SessionCheckpoint("cp.bin", turns());
    }

    @Test
    public void gettersRoundTrip() {
        SessionCheckpoint checkpoint = sample();
        assertThat(checkpoint.getFilename(), is("cp.bin"));
        assertThat(checkpoint.getTurns(), is(turns()));
    }

    @Test
    public void turnsAreCopiedAndUnmodifiable() {
        List<Pair<String, String>> input = new ArrayList<>(turns());
        SessionCheckpoint checkpoint = new SessionCheckpoint("cp.bin", input);
        input.clear(); // later mutation of the input must not leak in

        assertThat(checkpoint.getTurns().size(), is(2));
        assertThrows(
                UnsupportedOperationException.class, () -> checkpoint.getTurns().add(new Pair<>("user", "x")));
    }

    @Test
    public void equalsAndHashCode_sameValues() {
        assertEquals(sample(), sample());
        assertEquals(sample().hashCode(), sample().hashCode());
    }

    @Test
    public void equals_differsPerField() {
        SessionCheckpoint base = sample();
        assertNotEquals(base, new SessionCheckpoint("other.bin", turns()));
        assertNotEquals(base, new SessionCheckpoint("cp.bin", Collections.<Pair<String, String>>emptyList()));
    }

    @Test
    public void toString_isCompactSummary() {
        assertThat(sample().toString(), is("cp.bin (2 turns)"));
    }
}
