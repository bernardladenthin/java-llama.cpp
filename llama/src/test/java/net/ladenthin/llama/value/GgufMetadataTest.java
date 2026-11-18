// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.value;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import net.ladenthin.llama.ClaudeGenerated;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose = "Verify the GgufMetadata value type: raw/typed lookups (present, absent, "
                + "wrong-type), every convenience accessor incl. the architecture-dependent "
                + "context-length lookup, immutability of the entry view, the Lombok "
                + "equals/hashCode contract, and the compact handwritten toString.")
public class GgufMetadataTest {

    private static GgufMetadata sample() {
        Map<String, Object> entries = new LinkedHashMap<>();
        entries.put(GgufMetadata.KEY_ARCHITECTURE, "llama");
        entries.put(GgufMetadata.KEY_NAME, "Test Model");
        entries.put(GgufMetadata.KEY_PARAMETER_COUNT, 751_632_384L);
        entries.put(GgufMetadata.KEY_FILE_TYPE, 15L);
        entries.put("llama.context_length", 40_960L);
        entries.put(GgufMetadata.KEY_CHAT_TEMPLATE, "{{ messages }}");
        entries.put("general.some_flag", Boolean.TRUE);
        return new GgufMetadata(3, 291L, entries);
    }

    @Test
    public void headerGettersRoundTrip() {
        GgufMetadata meta = sample();
        assertThat(meta.getVersion(), is(3));
        assertThat(meta.getTensorCount(), is(291L));
        assertThat(meta.getEntries().size(), is(7));
    }

    @Test
    public void rawAndTypedLookups() {
        GgufMetadata meta = sample();
        assertThat(meta.getValue("general.some_flag").orElse(null), is((Object) Boolean.TRUE));
        assertThat(meta.getValue("absent").isPresent(), is(false));
        assertThat(meta.getString(GgufMetadata.KEY_NAME).orElse(""), is("Test Model"));
        // Wrong-type lookups answer empty rather than throwing.
        assertThat(meta.getString(GgufMetadata.KEY_FILE_TYPE).isPresent(), is(false));
        assertThat(meta.getLong(GgufMetadata.KEY_FILE_TYPE).orElse(0), is(15L));
        assertThat(meta.getLong(GgufMetadata.KEY_NAME).isPresent(), is(false));
        assertThat(meta.getLong("absent").isPresent(), is(false));
    }

    @Test
    public void convenienceAccessorsPresent() {
        GgufMetadata meta = sample();
        assertThat(meta.getArchitecture().orElse(""), is("llama"));
        assertThat(meta.getModelName().orElse(""), is("Test Model"));
        assertThat(meta.getParameterCount().orElse(0), is(751_632_384L));
        assertThat(meta.getFileType().orElse(0), is(15L));
        assertThat(meta.getChatTemplate().orElse(""), is("{{ messages }}"));
        assertThat(meta.getContextLength().orElse(0), is(40_960L));
    }

    @Test
    public void suppressTokensAreTheIdsOfTheArrayInFileOrder() {
        Map<String, Object> entries = new LinkedHashMap<>();
        entries.put(GgufMetadata.KEY_SUPPRESS_TOKENS, java.util.Arrays.<Object>asList(151643L, 7L, "not an id", 42L));
        GgufMetadata meta = new GgufMetadata(3, 0L, entries);
        assertEquals(java.util.Arrays.asList(151643, 7, 42), meta.getSuppressTokens());
        assertThrows(
                UnsupportedOperationException.class,
                () -> meta.getSuppressTokens().add(1));
    }

    @Test
    public void suppressTokensAreEmptyWithoutTheKeyOrWithoutAnArray() {
        assertEquals(
                Collections.emptyList(),
                new GgufMetadata(3, 0L, Collections.<String, Object>emptyMap()).getSuppressTokens());
        assertEquals(
                Collections.emptyList(),
                new GgufMetadata(3, 0L, Collections.<String, Object>singletonMap(GgufMetadata.KEY_SUPPRESS_TOKENS, 7L))
                        .getSuppressTokens());
        assertEquals(
                Collections.emptyList(),
                new GgufMetadata(
                                3,
                                0L,
                                Collections.<String, Object>singletonMap(
                                        GgufMetadata.KEY_SUPPRESS_TOKENS, Collections.emptyList()))
                        .getSuppressTokens());
    }

    @Test
    public void convenienceAccessorsAbsent() {
        GgufMetadata empty = new GgufMetadata(2, 0L, Collections.<String, Object>emptyMap());
        assertThat(empty.getArchitecture().isPresent(), is(false));
        assertThat(empty.getModelName().isPresent(), is(false));
        assertThat(empty.getParameterCount().isPresent(), is(false));
        assertThat(empty.getFileType().isPresent(), is(false));
        assertThat(empty.getChatTemplate().isPresent(), is(false));
        // No architecture -> the per-architecture context-length key cannot be derived.
        assertThat(empty.getContextLength().isPresent(), is(false));
    }

    @Test
    public void contextLengthRequiresBothArchitectureAndLengthKey() {
        GgufMetadata archOnly = new GgufMetadata(
                3, 1L, Collections.<String, Object>singletonMap(GgufMetadata.KEY_ARCHITECTURE, "llama"));
        assertThat(archOnly.getContextLength().isPresent(), is(false));
    }

    @Test
    public void entriesViewIsUnmodifiableAndDetachedFromInput() {
        Map<String, Object> input = new LinkedHashMap<>();
        input.put("k", 1L);
        GgufMetadata meta = new GgufMetadata(3, 1L, input);
        input.put("later", 2L); // mutation after construction must not leak in

        assertThat(meta.getEntries().size(), is(1));
        assertThrows(
                UnsupportedOperationException.class, () -> meta.getEntries().put("x", 1L));
    }

    @Test
    public void equalsAndHashCode_sameValues() {
        assertEquals(sample(), sample());
        assertEquals(sample().hashCode(), sample().hashCode());
    }

    @Test
    public void equals_differsPerField() {
        GgufMetadata base = sample();
        Map<String, Object> entries = new LinkedHashMap<>(base.getEntries());
        assertNotEquals(base, new GgufMetadata(2, 291L, entries));
        assertNotEquals(base, new GgufMetadata(3, 290L, entries));
        Map<String, Object> fewer = new LinkedHashMap<>(entries);
        fewer.remove(GgufMetadata.KEY_NAME);
        assertNotEquals(base, new GgufMetadata(3, 291L, fewer));
    }

    @Test
    public void toString_isCompactSummary() {
        assertThat(sample().toString(), is("GGUF v3 (291 tensors, 7 keys, arch=llama)"));
        GgufMetadata empty = new GgufMetadata(2, 0L, Collections.<String, Object>emptyMap());
        assertThat(empty.toString(), is("GGUF v2 (0 tensors, 0 keys, arch=?)"));
    }
}
