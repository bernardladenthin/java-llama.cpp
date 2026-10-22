// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import net.ladenthin.llama.value.GgufMetadata;

/**
 * Pure-Java GGUF header/metadata reader — inspects a model file <b>without loading it</b>:
 * no native library, no tensor data read, no RAM committed beyond the key/value table
 * (typically a few MB even for models whose tensor payload is many GB). Useful for model
 * pickers, download validators, and tooling that must describe a GGUF before deciding to
 * load it (e.g. checking {@code general.architecture} or the trained context length).
 *
 * <p>Supports GGUF container versions 2 and 3, in both little-endian (the common case) and
 * big-endian byte order (s390x-converted files; the byte order is auto-detected from the
 * version field). GGUF v1 (pre-2023, 32-bit lengths) is rejected with a clear error.
 * Only the header and key/value table are read; parsing stops before the tensor-info
 * section, so inspection cost is independent of model size.</p>
 *
 * <p>For metadata of an already-<em>loaded</em> model use
 * {@link LlamaModel#getModelMeta()} instead.</p>
 *
 * <pre>{@code
 * GgufMetadata meta = GgufInspector.read(Paths.get("models/Qwen3-0.6B-Q4_K_M.gguf"));
 * String arch = meta.getArchitecture().orElse("unknown");
 * long ctx    = meta.getContextLength().orElse(0);
 * }</pre>
 */
public final class GgufInspector {

    /** The 4-byte GGUF magic, {@code 'G' 'G' 'U' 'F'}. */
    private static final byte[] MAGIC = {'G', 'G', 'U', 'F'};

    /** Sanity cap on the declared key/value count (a real table has dozens of entries). */
    private static final long MAX_KV_COUNT = 1L << 20;

    /** Sanity cap on a single string length (largest real strings are chat templates / tokens). */
    private static final long MAX_STRING_BYTES = 1L << 27;

    /** Sanity cap on a single array element count (largest real arrays are ~150k-token vocabs). */
    private static final long MAX_ARRAY_ELEMENTS = 1L << 26;

    // GGUF metadata value type ids (gguf.h: enum gguf_type).
    private static final int TYPE_UINT8 = 0;
    private static final int TYPE_INT8 = 1;
    private static final int TYPE_UINT16 = 2;
    private static final int TYPE_INT16 = 3;
    private static final int TYPE_UINT32 = 4;
    private static final int TYPE_INT32 = 5;
    private static final int TYPE_FLOAT32 = 6;
    private static final int TYPE_BOOL = 7;
    private static final int TYPE_STRING = 8;
    private static final int TYPE_ARRAY = 9;
    private static final int TYPE_UINT64 = 10;
    private static final int TYPE_INT64 = 11;
    private static final int TYPE_FLOAT64 = 12;

    private GgufInspector() {}

    /**
     * Read the GGUF header and key/value table from a file.
     *
     * @param ggufFile path to the GGUF file
     * @return the decoded metadata
     * @throws IOException if the file cannot be read, is not a GGUF file, uses the
     *     unsupported v1 container, or is structurally invalid/truncated
     */
    public static GgufMetadata read(Path ggufFile) throws IOException {
        try (InputStream in = new BufferedInputStream(Files.newInputStream(ggufFile))) {
            return read(in);
        }
    }

    /**
     * Read the GGUF header and key/value table from a stream. Reads exactly the header +
     * key/value section; the stream is not consumed past it (tensor infos and tensor data
     * stay unread).
     *
     * @param in the stream positioned at the start of the GGUF file
     * @return the decoded metadata
     * @throws IOException if the stream is not a GGUF file, uses the unsupported v1
     *     container, or is structurally invalid/truncated
     */
    public static GgufMetadata read(InputStream in) throws IOException {
        DataInputStream data = new DataInputStream(in);
        byte[] magic = new byte[MAGIC.length];
        data.readFully(magic);
        for (int i = 0; i < MAGIC.length; i++) {
            if (magic[i] != MAGIC[i]) {
                throw new IOException("Not a GGUF file (magic mismatch: read 0x"
                        + String.format("%02X%02X%02X%02X", magic[0], magic[1], magic[2], magic[3])
                        + ", expected 'GGUF')");
            }
        }

        // The version field doubles as the byte-order detector: read little-endian first;
        // a big-endian (s390x-converted) file then shows the version in the high bytes.
        Reader reader = new Reader(data, ByteOrder.LITTLE_ENDIAN);
        long versionLe = reader.u32();
        long version = versionLe;
        if (versionLe > 0xFFFFL && Long.reverseBytes(versionLe << 32) <= 0xFFFFL) {
            version = Long.reverseBytes(versionLe << 32);
            reader = new Reader(data, ByteOrder.BIG_ENDIAN);
        }
        if (version == 1) {
            throw new IOException(
                    "GGUF v" + version + " is not supported (pre-2023 32-bit container); re-convert the model");
        }
        if (version != 2 && version != 3) {
            throw new IOException("Unsupported GGUF version: " + version);
        }

        long tensorCount = reader.u64();
        long kvCount = reader.u64();
        if (kvCount < 0 || kvCount > MAX_KV_COUNT) {
            throw new IOException("Implausible GGUF key/value count: " + kvCount);
        }

        Map<String, Object> entries = new LinkedHashMap<String, Object>();
        for (long i = 0; i < kvCount; i++) {
            String key = reader.string();
            int type = (int) reader.u32();
            entries.put(key, readValue(reader, type));
        }
        return new GgufMetadata((int) version, tensorCount, entries);
    }

    private static Object readValue(Reader reader, int type) throws IOException {
        switch (type) {
            case TYPE_UINT8:
                return Long.valueOf(reader.u8());
            case TYPE_INT8:
                return Long.valueOf((byte) reader.u8());
            case TYPE_UINT16:
                return Long.valueOf(reader.u16());
            case TYPE_INT16:
                return Long.valueOf((short) reader.u16());
            case TYPE_UINT32:
                return Long.valueOf(reader.u32());
            case TYPE_INT32:
                return Long.valueOf(reader.i32());
            case TYPE_FLOAT32:
                return Double.valueOf(Float.intBitsToFloat(reader.i32()));
            case TYPE_BOOL:
                return Boolean.valueOf(reader.u8() != 0);
            case TYPE_STRING:
                return reader.string();
            case TYPE_ARRAY:
                return readArray(reader);
            case TYPE_UINT64:
            case TYPE_INT64:
                // u64 values above Long.MAX_VALUE wrap negative; real metadata never gets there.
                return Long.valueOf(reader.u64());
            case TYPE_FLOAT64:
                return Double.valueOf(Double.longBitsToDouble(reader.u64()));
            default:
                throw new IOException("Unknown GGUF metadata value type id: " + type);
        }
    }

    private static Object readArray(Reader reader) throws IOException {
        int elementType = (int) reader.u32();
        // ggml's gguf reader rejects arrays of arrays ("invalid GGUF type"), so llama.cpp cannot load such
        // a file; accepting them here also let a crafted nesting recurse into a StackOverflowError.
        if (elementType == TYPE_ARRAY) {
            throw new IOException(
                    "Nested GGUF arrays are not supported (llama.cpp rejects them): element type id " + elementType);
        }
        long count = reader.u64();
        if (count < 0 || count > MAX_ARRAY_ELEMENTS) {
            throw new IOException("Implausible GGUF array element count: " + count);
        }
        List<Object> values = new ArrayList<Object>((int) Math.min(count, 1 << 16));
        for (long i = 0; i < count; i++) {
            values.add(readValue(reader, elementType));
        }
        return Collections.unmodifiableList(values);
    }

    /** Byte-order-aware primitive reader over the underlying stream. */
    private static final class Reader {
        private final DataInputStream in;
        private final ByteOrder order;
        private final byte[] scratch = new byte[8];

        Reader(DataInputStream in, ByteOrder order) {
            this.in = in;
            this.order = order;
        }

        int u8() throws IOException {
            return in.readUnsignedByte();
        }

        int u16() throws IOException {
            in.readFully(scratch, 0, 2);
            return ByteBuffer.wrap(scratch, 0, 2).order(order).getShort() & 0xFFFF;
        }

        long u32() throws IOException {
            return i32() & 0xFFFFFFFFL;
        }

        int i32() throws IOException {
            in.readFully(scratch, 0, 4);
            return ByteBuffer.wrap(scratch, 0, 4).order(order).getInt();
        }

        long u64() throws IOException {
            in.readFully(scratch, 0, 8);
            return ByteBuffer.wrap(scratch, 0, 8).order(order).getLong();
        }

        String string() throws IOException {
            long length = u64();
            if (length < 0 || length > MAX_STRING_BYTES) {
                throw new IOException("Implausible GGUF string length: " + length);
            }
            byte[] bytes = new byte[(int) length];
            in.readFully(bytes);
            return new String(bytes, StandardCharsets.UTF_8);
        }
    }
}
