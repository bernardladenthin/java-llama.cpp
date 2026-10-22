// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import net.ladenthin.llama.value.GgufMetadata;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@ClaudeGenerated(
        purpose = "Verify the pure-Java GGUF header/metadata reader against in-memory generated "
                + "fixtures (no committed binaries, no native library): every metadata value "
                + "type, v2/v3 containers, big-endian auto-detection, the fail-loud paths "
                + "(magic mismatch, v1, unknown version/type, truncation, implausible lengths), "
                + "and that parsing never touches the tensor payload.")
public class GgufInspectorTest {

    @TempDir
    Path tempDir;

    /** Minimal GGUF byte-stream writer mirroring the container layout (header + KV table). */
    private static final class GgufWriter {
        private final ByteArrayOutputStream out = new ByteArrayOutputStream();
        private final ByteOrder order;

        GgufWriter(ByteOrder order) {
            this.order = order;
        }

        GgufWriter magic() {
            out.write('G');
            out.write('G');
            out.write('U');
            out.write('F');
            return this;
        }

        GgufWriter u32(long value) {
            out.write(ByteBuffer.allocate(4).order(order).putInt((int) value).array(), 0, 4);
            return this;
        }

        GgufWriter u64(long value) {
            out.write(ByteBuffer.allocate(8).order(order).putLong(value).array(), 0, 8);
            return this;
        }

        GgufWriter u8(int value) {
            out.write(value);
            return this;
        }

        GgufWriter str(String value) {
            byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
            u64(bytes.length);
            out.write(bytes, 0, bytes.length);
            return this;
        }

        GgufWriter raw(byte[] bytes) {
            out.write(bytes, 0, bytes.length);
            return this;
        }

        byte[] bytes() {
            return out.toByteArray();
        }
    }

    /** Builds a representative v3 file: header + a KV table covering the common value types. */
    private static byte[] sampleGguf(ByteOrder order, long version) {
        GgufWriter writer = new GgufWriter(order);
        writer.magic().u32(version);
        writer.u64(291); // tensor count
        writer.u64(8); // kv count
        writer.str("general.architecture").u32(8).str("llama");
        writer.str("general.name").u32(8).str("Test Model");
        writer.str("general.parameter_count").u32(10).u64(751_632_384L); // u64
        writer.str("general.file_type").u32(4).u32(15); // u32 (Q4_K_M)
        writer.str("llama.context_length").u32(4).u32(40_960);
        writer.str("llama.rope.freq_base").u32(6).u32(Float.floatToIntBits(1_000_000.0f)); // f32
        writer.str("tokenizer.chat_template").u32(8).str("{{ messages }}");
        // array of i32
        writer.str("tokenizer.ggml.token_type")
                .u32(9)
                .u32(5)
                .u64(3)
                .u32(1)
                .u32(1)
                .u32(2);
        // Tensor-info section would follow; parsing must stop before it, so arbitrary
        // trailing bytes must not disturb the result.
        writer.raw(new byte[] {(byte) 0xDE, (byte) 0xAD, (byte) 0xBE, (byte) 0xEF});
        return writer.bytes();
    }

    @Test
    public void readsHeaderAndAllValueTypes() throws IOException {
        GgufMetadata meta = GgufInspector.read(new ByteArrayInputStream(sampleGguf(ByteOrder.LITTLE_ENDIAN, 3)));

        assertThat(meta.getVersion(), is(3));
        assertThat(meta.getTensorCount(), is(291L));
        assertThat(meta.getEntries().size(), is(8));
        assertThat(meta.getArchitecture().orElse(""), is("llama"));
        assertThat(meta.getModelName().orElse(""), is("Test Model"));
        assertThat(meta.getParameterCount().orElse(0), is(751_632_384L));
        assertThat(meta.getFileType().orElse(0), is(15L));
        assertThat(meta.getContextLength().orElse(0), is(40_960L));
        assertThat(meta.getChatTemplate().orElse(""), is("{{ messages }}"));
        assertThat(meta.getValue("llama.rope.freq_base").orElse(null), is((Object) Double.valueOf(1_000_000.0)));
        assertThat(meta.getValue("tokenizer.ggml.token_type").orElse(null), is((Object)
                Arrays.<Object>asList(1L, 1L, 2L)));
    }

    @Test
    public void readsVersion2Container() throws IOException {
        GgufMetadata meta = GgufInspector.read(new ByteArrayInputStream(sampleGguf(ByteOrder.LITTLE_ENDIAN, 2)));

        assertThat(meta.getVersion(), is(2));
        assertThat(meta.getArchitecture().orElse(""), is("llama"));
    }

    @Test
    public void autoDetectsBigEndianContainer() throws IOException {
        GgufMetadata meta = GgufInspector.read(new ByteArrayInputStream(sampleGguf(ByteOrder.BIG_ENDIAN, 3)));

        assertThat(meta.getVersion(), is(3));
        assertThat(meta.getTensorCount(), is(291L));
        assertThat(meta.getArchitecture().orElse(""), is("llama"));
        assertThat(meta.getContextLength().orElse(0), is(40_960L));
    }

    @Test
    public void readsFromPath(@TempDir Path dir) throws IOException {
        Path file = dir.resolve("sample.gguf");
        Files.write(file, sampleGguf(ByteOrder.LITTLE_ENDIAN, 3));

        GgufMetadata meta = GgufInspector.read(file);

        assertThat(meta.getModelName().orElse(""), is("Test Model"));
    }

    @Test
    public void rejectsNonGgufMagic() {
        byte[] bytes = new GgufWriter(ByteOrder.LITTLE_ENDIAN)
                .raw("GGML".getBytes(StandardCharsets.UTF_8))
                .u32(3)
                .bytes();

        IOException thrown = assertThrows(IOException.class, () -> GgufInspector.read(new ByteArrayInputStream(bytes)));

        assertThat(thrown.getMessage(), containsString("magic"));
    }

    @Test
    public void rejectsVersion1() {
        byte[] bytes = new GgufWriter(ByteOrder.LITTLE_ENDIAN).magic().u32(1).bytes();

        IOException thrown = assertThrows(IOException.class, () -> GgufInspector.read(new ByteArrayInputStream(bytes)));

        assertThat(thrown.getMessage(), containsString("v1"));
    }

    @Test
    public void rejectsUnknownVersion() {
        byte[] bytes = new GgufWriter(ByteOrder.LITTLE_ENDIAN).magic().u32(4).bytes();

        IOException thrown = assertThrows(IOException.class, () -> GgufInspector.read(new ByteArrayInputStream(bytes)));

        assertThat(thrown.getMessage(), containsString("version"));
    }

    @Test
    public void rejectsUnknownValueTypeId() {
        byte[] bytes = new GgufWriter(ByteOrder.LITTLE_ENDIAN)
                .magic()
                .u32(3)
                .u64(0)
                .u64(1)
                .str("key")
                .u32(99)
                .bytes();

        IOException thrown = assertThrows(IOException.class, () -> GgufInspector.read(new ByteArrayInputStream(bytes)));

        assertThat(thrown.getMessage(), containsString("type id"));
    }

    @Test
    public void rejectsImplausibleStringLength() {
        byte[] bytes = new GgufWriter(ByteOrder.LITTLE_ENDIAN)
                .magic()
                .u32(3)
                .u64(0)
                .u64(1)
                .u64(Long.MAX_VALUE) // key length
                .bytes();

        IOException thrown = assertThrows(IOException.class, () -> GgufInspector.read(new ByteArrayInputStream(bytes)));

        assertThat(thrown.getMessage(), containsString("string length"));
    }

    @Test
    public void failsLoudOnTruncatedFile() {
        byte[] full = sampleGguf(ByteOrder.LITTLE_ENDIAN, 3);
        byte[] truncated = Arrays.copyOf(full, 40);

        assertThrows(IOException.class, () -> GgufInspector.read(new ByteArrayInputStream(truncated)));
    }

    /** A file with one key whose value is {@code depth} arrays nested in each other (one element each). */
    private static byte[] nestedArrayGguf(int depth) {
        GgufWriter writer = new GgufWriter(ByteOrder.LITTLE_ENDIAN);
        writer.magic().u32(3).u64(0).u64(1);
        writer.str("nested").u32(9); // the value is an array ...
        for (int level = 1; level < depth; level++) {
            writer.u32(9).u64(1); // ... of one array ...
        }
        writer.u32(5).u64(1).u32(42); // ... of one int32
        return writer.bytes();
    }

    @Test
    public void nestedArrayIsRejectedLikeUpstream() {
        // ggml's gguf reader rejects an array whose element type is an array ("invalid GGUF type"),
        // so llama.cpp cannot load such a file; the inspector must not report it as valid metadata.
        byte[] nested = nestedArrayGguf(2);

        assertThrows(IOException.class, () -> GgufInspector.read(new ByteArrayInputStream(nested)));
    }

    @Test
    public void deeplyNestedArraysFailWithIOExceptionNotStackOverflow() {
        // Each nesting level costs 12 bytes but one recursion frame; a crafted file ended the read in a
        // StackOverflowError instead of the IOException the method declares for an invalid file.
        byte[] nested = nestedArrayGguf(100_000);

        assertThrows(IOException.class, () -> GgufInspector.read(new ByteArrayInputStream(nested)));
    }

    @Test
    public void readsRealModelFileWhenPresent() throws IOException {
        // Real-file sanity (CI: the shared GGUF cache; locally self-skips without models/).
        Path model = java.nio.file.Paths.get(TestConstants.REASONING_MODEL_PATH);
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(model), "reasoning model not present");

        GgufMetadata meta = GgufInspector.read(model);

        assertThat(meta.getVersion() >= 2, is(true));
        assertThat(meta.getTensorCount() > 0, is(true));
        assertThat(meta.getArchitecture().isPresent(), is(true));
        assertThat(meta.getContextLength().orElse(0) > 0, is(true));
    }

    @Test
    public void arrayValuesAreUnmodifiable() throws IOException {
        GgufMetadata meta = GgufInspector.read(new ByteArrayInputStream(sampleGguf(ByteOrder.LITTLE_ENDIAN, 3)));
        @SuppressWarnings("unchecked")
        List<Object> tokenTypes =
                (List<Object>) meta.getValue("tokenizer.ggml.token_type").orElseThrow(AssertionError::new);

        assertThrows(UnsupportedOperationException.class, () -> tokenTypes.add(0L));
    }
}
