// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import java.util.Objects;
import net.ladenthin.llama.args.QuantizationType;
import net.ladenthin.llama.loader.LlamaLoader;
import org.jspecify.annotations.Nullable;

/**
 * In-JVM GGUF model quantization over llama.cpp's {@code llama_model_quantize} — the Java
 * counterpart of the {@code llama-quantize} CLI tool. Converts a GGUF file to another
 * {@linkplain QuantizationType quantization scheme} without leaving the JVM or shelling out.
 *
 * <pre>{@code
 * LlamaQuantizer.quantize("model-f16.gguf", "model-q4_k_m.gguf", QuantizationType.Q4_K_M);
 * }</pre>
 *
 * <p>Quantizing an <em>already-quantized</em> input (re-quantization) is refused by llama.cpp by
 * default because it degrades quality; opt in explicitly via
 * {@link #quantize(String, String, QuantizationType, int, boolean)} with
 * {@code allowRequantize = true}.</p>
 *
 * <p>An importance matrix ({@code llama-imatrix}'s output) is passed with
 * {@link #quantize(String, String, QuantizationType, int, boolean, String)}; the IQ1 and IQ2
 * types, IQ3_XS, IQ3_XXS and Q2_K_S cannot be written without one.</p>
 *
 * <p>Quantization is CPU-bound and can take minutes for large models; it initializes the shared
 * llama backend (a no-op if a {@link LlamaModel} is already loaded) and may safely run while
 * models are loaded in the same JVM.</p>
 */
public final class LlamaQuantizer {

    static {
        LlamaLoader.initialize();
    }

    private LlamaQuantizer() {}

    /**
     * Quantize {@code inputPath} to {@code outputPath} with default settings (all available
     * hardware threads, no re-quantization of already-quantized inputs).
     *
     * @param inputPath  the source GGUF (typically F32/F16/BF16)
     * @param outputPath the destination GGUF to write
     * @param type       the target quantization scheme
     * @throws net.ladenthin.llama.exception.LlamaException if quantization fails (missing input,
     *         unwritable output, re-quantization without opt-in, unsupported tensor layout, …)
     */
    public static void quantize(String inputPath, String outputPath, QuantizationType type) {
        quantize(inputPath, outputPath, type, 0, false);
    }

    /**
     * Quantize {@code inputPath} to {@code outputPath} with explicit thread count and
     * re-quantization opt-in.
     *
     * @param inputPath       the source GGUF
     * @param outputPath      the destination GGUF to write
     * @param type            the target quantization scheme
     * @param threads         quantization threads; {@code <= 0} uses all hardware threads
     * @param allowRequantize permit quantizing tensors that are not F32/F16 (re-quantizing an
     *                        already-quantized model — lossy on top of lossy, quality degrades)
     * @throws net.ladenthin.llama.exception.LlamaException if quantization fails
     */
    public static void quantize(
            String inputPath, String outputPath, QuantizationType type, int threads, boolean allowRequantize) {
        Objects.requireNonNull(inputPath, "inputPath");
        Objects.requireNonNull(outputPath, "outputPath");
        quantize(inputPath, outputPath, type, threads, allowRequantize, null);
    }

    /**
     * Quantize a GGUF model file with an importance matrix: the file {@code llama-imatrix} computed
     * for this model from a calibration text. It tells the quantizer which weights matter, which
     * improves every type below Q8 and is required by the smallest ones
     * ({@link QuantizationType#requiresImportanceMatrix()}). The written model records the file
     * under {@code quantize.imatrix.*}, as {@code llama-quantize} does.
     *
     * @param inputPath path to the source GGUF file
     * @param outputPath path the quantized GGUF file is written to
     * @param type the target quantization scheme
     * @param threads number of threads to use; {@code <= 0} lets llama.cpp pick
     *     ({@code std::thread::hardware_concurrency()})
     * @param allowRequantize whether to permit quantizing an already-quantized input
     * @param importanceMatrixPath path of the importance matrix (GGUF or the legacy format), or
     *     {@code null} for none
     * @throws IllegalArgumentException if {@code type} requires an importance matrix and none is given
     * @throws net.ladenthin.llama.exception.LlamaException if quantization fails
     */
    public static void quantize(
            String inputPath,
            String outputPath,
            QuantizationType type,
            int threads,
            boolean allowRequantize,
            @Nullable String importanceMatrixPath) {
        Objects.requireNonNull(inputPath, "inputPath");
        Objects.requireNonNull(outputPath, "outputPath");
        Objects.requireNonNull(type, "type");
        if (importanceMatrixPath == null && type.requiresImportanceMatrix()) {
            throw new IllegalArgumentException(type + " cannot be quantized to without an importance matrix");
        }
        quantizeNative(inputPath, outputPath, type.getFtypeValue(), threads, allowRequantize, importanceMatrixPath);
    }

    private static native void quantizeNative(
            String inputPath,
            String outputPath,
            int ftype,
            int threads,
            boolean allowRequantize,
            @Nullable String importanceMatrixPath);
}
