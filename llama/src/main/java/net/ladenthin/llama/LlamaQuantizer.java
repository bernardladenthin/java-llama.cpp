// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import java.util.Objects;
import net.ladenthin.llama.args.QuantizationType;
import net.ladenthin.llama.callback.LoadProgressCallback;
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

    /**
     * Compute the importance matrix of a model over a calibration text, as the {@code llama-imatrix}
     * tool does, with chunks of 512 tokens over the whole text on the CPU.
     *
     * @param modelPath path to the model, best the unquantized one
     * @param textPath path to the calibration text (plain UTF-8 text)
     * @param outputPath path the importance matrix is written to (GGUF)
     * @return the number of chunks the matrix was computed over
     * @throws net.ladenthin.llama.exception.LlamaException if the model or the text cannot be read,
     *     or the text is shorter than one chunk
     * @see #computeImportanceMatrix(String, String, String, int, int, int, int)
     */
    public static int computeImportanceMatrix(String modelPath, String textPath, String outputPath) {
        return computeImportanceMatrix(modelPath, textPath, outputPath, 0, 0, 0, 0);
    }

    /**
     * Compute the importance matrix of a model over a calibration text: the model reads the text
     * chunk by chunk, and for every weight matrix of its layers the squared activations entering it
     * are summed per column. Pass the result to
     * {@link #quantize(String, String, QuantizationType, int, boolean, String)}; the file is the
     * one {@code llama-imatrix} writes, so either tool's output works with either quantizer.
     *
     * <p>The run loads the model itself and frees it again; it takes about as long as evaluating
     * the text once. A few hundred chunks of text close to what the model will be used for is the
     * usual amount.</p>
     *
     * @param modelPath path to the model, best the unquantized one
     * @param textPath path to the calibration text (plain UTF-8 text)
     * @param outputPath path the importance matrix is written to (GGUF)
     * @param chunkSize tokens per chunk, at least 16; {@code 0} for 512
     * @param maxChunks the most chunks to evaluate; {@code 0} for every whole chunk of the text
     * @param threads number of threads; {@code <= 0} lets llama.cpp pick
     * @param gpuLayers layers to offload to a GPU; {@code 0} for none
     * @return the number of chunks the matrix was computed over
     * @throws net.ladenthin.llama.exception.LlamaException if the model or the text cannot be read,
     *     the chunk size is below 16, or the text is shorter than one chunk
     */
    public static int computeImportanceMatrix(
            String modelPath,
            String textPath,
            String outputPath,
            int chunkSize,
            int maxChunks,
            int threads,
            int gpuLayers) {
        return computeImportanceMatrix(modelPath, textPath, outputPath, chunkSize, maxChunks, threads, gpuLayers, null);
    }

    /**
     * Compute the importance matrix of a model over a calibration text, reporting progress: the
     * callback is called on the calling thread after every chunk with the fraction of the chunks
     * done, and returning {@code false} cancels the run, which then throws and writes nothing.
     *
     * @param modelPath path to the model, best the unquantized one
     * @param textPath path to the calibration text (plain UTF-8 text)
     * @param outputPath path the importance matrix is written to (GGUF)
     * @param chunkSize tokens per chunk, at least 16; {@code 0} for 512
     * @param maxChunks the most chunks to evaluate; {@code 0} for every whole chunk of the text
     * @param threads number of threads; {@code <= 0} lets llama.cpp pick
     * @param gpuLayers layers to offload to a GPU; {@code 0} for none
     * @param progress called after every chunk, or {@code null}
     * @return the number of chunks the matrix was computed over
     * @throws net.ladenthin.llama.exception.LlamaException if the model or the text cannot be read,
     *     the chunk size is below 16, the text is shorter than one chunk, or the callback cancelled
     */
    public static int computeImportanceMatrix(
            String modelPath,
            String textPath,
            String outputPath,
            int chunkSize,
            int maxChunks,
            int threads,
            int gpuLayers,
            @Nullable LoadProgressCallback progress) {
        Objects.requireNonNull(modelPath, "modelPath");
        Objects.requireNonNull(textPath, "textPath");
        Objects.requireNonNull(outputPath, "outputPath");
        return computeImportanceMatrixNative(
                modelPath, textPath, outputPath, chunkSize, maxChunks, threads, gpuLayers, progress);
    }

    private static native int computeImportanceMatrixNative(
            String modelPath,
            String textPath,
            String outputPath,
            int chunkSize,
            int maxChunks,
            int threads,
            int gpuLayers,
            @Nullable LoadProgressCallback progress);

    private static native void quantizeNative(
            String inputPath,
            String outputPath,
            int ftype,
            int threads,
            boolean allowRequantize,
            @Nullable String importanceMatrixPath);
}
