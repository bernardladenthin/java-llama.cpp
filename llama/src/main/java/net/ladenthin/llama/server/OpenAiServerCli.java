// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import net.ladenthin.llama.args.CacheType;
import net.ladenthin.llama.parameters.ModelParameters;
import org.jspecify.annotations.Nullable;

/**
 * Command-line argument parser for {@link OpenAiCompatServer}'s standalone launcher. Pure and free of
 * any native dependency, so it can be unit-tested in isolation (no socket, no model).
 *
 * <p>{@link #parse(String[])} returns an immutable {@link Options} or throws
 * {@link IllegalArgumentException} (whose message embeds the {@link #usage()} text) for unknown flags,
 * missing values or a missing required {@code --model}. {@code -h}/{@code --help} is detected separately
 * via {@link #isHelpRequested(String[])} so callers can print help without it being treated as an error.
 *
 * <p>Flags mirror llama.cpp's own server where they overlap ({@code -m}, {@code -p}, {@code -c},
 * {@code -b}, {@code -ub}, {@code -ngl}, {@code -t}, {@code -tb}, {@code -ctk}, {@code -ctv},
 * {@code --jinja}, {@code --chat-template-kwargs}, {@code -mmdev}); a few legacy spellings are accepted as aliases so
 * earlier documented invocations keep working. The {@code --chat-template-kwargs} JSON is parsed here
 * (the only JSON this otherwise dependency-light parser touches) so a malformed object fails fast with
 * usage text rather than at native model load.
 */
public final class OpenAiServerCli {

    /** Default bind interface (loopback only; pass {@code --host 0.0.0.0} to expose on the LAN). */
    public static final String DEFAULT_HOST = OpenAiServerConfig.DEFAULT_HOST;

    /** Default TCP port. */
    public static final int DEFAULT_PORT = OpenAiServerConfig.DEFAULT_PORT;

    private OpenAiServerCli() {}

    /**
     * Whether the arguments request the help text.
     *
     * @param args the raw command-line arguments
     * @return {@code true} if {@code -h} or {@code --help} is present
     */
    public static boolean isHelpRequested(String... args) {
        for (final String arg : args) {
            if ("-h".equals(arg) || "--help".equals(arg)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Parse the command-line arguments into validated {@link Options}.
     *
     * @param args the raw command-line arguments
     * @return the parsed options
     * @throws IllegalArgumentException if an argument is unknown, a value is missing or malformed,
     *                                  or the required {@code --model} is absent
     */
    public static Options parse(String... args) {
        String host = DEFAULT_HOST;
        int port = DEFAULT_PORT;
        @Nullable String modelPath = null;
        @Nullable String modelId = null;
        @Nullable String apiKey = null;
        @Nullable String mmproj = null;
        @Nullable String mmprojDevice = null;
        int ctxSize = 0;
        int gpuLayers = 0;
        int threads = 0;
        int threadsBatch = 0;
        int parallel = 0;
        int batchSize = 0;
        int ubatchSize = 0;
        @Nullable CacheType cacheTypeK = null;
        @Nullable CacheType cacheTypeV = null;
        boolean jinja = false;
        @Nullable Map<String, String> chatTemplateKwargs = null;
        boolean embedding = false;
        boolean reranking = false;

        for (int i = 0; i < args.length; i++) {
            final String arg = args[i];
            switch (arg) {
                case "-m":
                case "--model":
                    modelPath = nextValue(args, ++i, arg);
                    break;
                case "--host":
                    host = nextValue(args, ++i, arg);
                    break;
                case "-p":
                case "--port":
                    port = intValue(args, ++i, arg);
                    break;
                case "-c":
                case "--ctx-size":
                case "--ctx":
                    ctxSize = intValue(args, ++i, arg);
                    break;
                case "-ngl":
                case "--n-gpu-layers":
                case "--gpu-layers":
                    gpuLayers = intValue(args, ++i, arg);
                    break;
                case "-t":
                case "--threads":
                    threads = intValue(args, ++i, arg);
                    break;
                case "-tb":
                case "--threads-batch":
                    threadsBatch = intValue(args, ++i, arg);
                    break;
                case "-b":
                case "--batch-size":
                    batchSize = intValue(args, ++i, arg);
                    break;
                case "-ub":
                case "--ubatch-size":
                    ubatchSize = intValue(args, ++i, arg);
                    break;
                case "-ctk":
                case "--cache-type-k":
                    cacheTypeK = cacheTypeValue(args, ++i, arg);
                    break;
                case "-ctv":
                case "--cache-type-v":
                    cacheTypeV = cacheTypeValue(args, ++i, arg);
                    break;
                case "--jinja":
                    jinja = true;
                    break;
                case "--chat-template-kwargs":
                    chatTemplateKwargs = parseChatTemplateKwargs(nextValue(args, ++i, arg), arg);
                    break;
                case "--parallel":
                    parallel = intValue(args, ++i, arg);
                    break;
                case "--model-id":
                case "--model-alias":
                    modelId = nextValue(args, ++i, arg);
                    break;
                case "--api-key":
                    apiKey = nextValue(args, ++i, arg);
                    break;
                case "--mmproj":
                    mmproj = nextValue(args, ++i, arg);
                    break;
                case "-mmdev":
                case "--mmproj-device":
                    mmprojDevice = nextValue(args, ++i, arg);
                    break;
                case "--embedding":
                case "--embeddings":
                    embedding = true;
                    break;
                case "--reranking":
                case "--rerank":
                    reranking = true;
                    break;
                case "-h":
                case "--help":
                    // Detected by isHelpRequested(); accepted here so parse() still succeeds.
                    break;
                default:
                    throw error("Unknown argument: " + arg);
            }
        }

        if (modelPath == null) {
            throw error("Missing required argument: -m/--model <path-to-gguf>");
        }
        return new Options(
                host,
                port,
                modelPath,
                modelId,
                apiKey,
                mmproj,
                mmprojDevice,
                ctxSize,
                gpuLayers,
                threads,
                threadsBatch,
                parallel,
                batchSize,
                ubatchSize,
                cacheTypeK,
                cacheTypeV,
                jinja,
                chatTemplateKwargs,
                embedding,
                reranking);
    }

    /**
     * The human-readable usage / help text.
     *
     * @return the usage text
     */
    public static String usage() {
        return String.join(
                System.lineSeparator(),
                "OpenAiCompatServer - OpenAI-compatible HTTP server for java-llama.cpp",
                "",
                "Usage:",
                "  java -jar llama-<version>-jar-with-dependencies.jar --model <path.gguf> [options]",
                "",
                "Required:",
                "  -m,  --model <path>        Path to the GGUF model file",
                "",
                "Options:",
                "  --host <host>              Interface to bind (default: " + DEFAULT_HOST + ")",
                "  -p,  --port <port>         TCP port to listen on (default: " + DEFAULT_PORT + ")",
                "  -c,  --ctx-size <n>        Context window size (default: llama.cpp default)",
                "  -b,  --batch-size <n>      Logical (prompt) batch size (default: llama.cpp default)",
                "  -ub, --ubatch-size <n>     Physical (micro) batch size (default: llama.cpp default)",
                "  -ngl,--n-gpu-layers <n>    Layers to offload to GPU (default: 0 = CPU only)",
                "  -t,  --threads <n>         Inference thread count (default: llama.cpp default)",
                "  -tb, --threads-batch <n>   Thread count for batch/prompt processing (default: same as -t)",
                "  -ctk,--cache-type-k <t>    KV cache K quantization: " + cacheTypeChoices() + " (default: f16)",
                "  -ctv,--cache-type-v <t>    KV cache V quantization: " + cacheTypeChoices() + " (default: f16)",
                "  --jinja                    Use the model's Jinja chat template",
                "  --chat-template-kwargs <j> JSON object of chat-template variables (requires --jinja),",
                "                             e.g. {\"reasoning_effort\":\"low\"}",
                "  --parallel <n>             Parallel inference slots (default: llama.cpp default)",
                "  --model-id <name>          Model id reported by /v1/models (default: file name)",
                "  --api-key <key>            Require an 'Authorization: Bearer <key>' header",
                "  --mmproj <path>            Multimodal projector for vision models (enables image input)",
                "  -mmdev, --mmproj-device <d>  Device for the multimodal projector ('none' = keep on CPU)",
                "  --embedding                Load in embedding mode (enables POST /v1/embeddings)",
                "  --reranking                Load in reranking mode (enables POST /v1/rerank)",
                "  -h,  --help                Show this help and exit",
                "",
                "Endpoints:",
                "  POST /v1/chat/completions  (streaming via SSE + non-streaming)",
                "  POST /v1/completions",
                "  POST /v1/embeddings        (requires --embedding)",
                "  POST /v1/rerank            (requires --reranking)",
                "  POST /infill               (fill-in-the-middle / autocomplete)",
                "  GET  /v1/models",
                "  GET  /health");
    }

    private static String nextValue(String[] args, int valueIndex, String flag) {
        if (valueIndex >= args.length) {
            throw error("Missing value for " + flag);
        }
        return args[valueIndex];
    }

    private static int intValue(String[] args, int valueIndex, String flag) {
        final String raw = nextValue(args, valueIndex, flag);
        try {
            return Integer.parseInt(raw.trim());
        } catch (NumberFormatException e) {
            throw error(flag + " expects an integer, got: " + raw, e);
        }
    }

    /** Reusable parser for the {@code --chat-template-kwargs} JSON object; no state, thread-safe. */
    private static final ObjectMapper CHAT_TEMPLATE_KWARGS_MAPPER = new ObjectMapper();

    private static CacheType cacheTypeValue(String[] args, int valueIndex, String flag) {
        final String raw = nextValue(args, valueIndex, flag).trim();
        for (final CacheType type : CacheType.values()) {
            if (type.getArgValue().equalsIgnoreCase(raw)) {
                return type;
            }
        }
        throw error(flag + " expects one of " + cacheTypeChoices() + ", got: " + raw);
    }

    private static String cacheTypeChoices() {
        final StringBuilder sb = new StringBuilder();
        for (final CacheType type : CacheType.values()) {
            if (sb.length() > 0) {
                sb.append(", ");
            }
            sb.append(type.getArgValue());
        }
        return sb.toString();
    }

    /**
     * Parse a {@code --chat-template-kwargs} JSON object into the raw-per-value map that
     * {@link ModelParameters#setChatTemplateKwargs(Map)} expects: each entry's value is kept as its
     * raw JSON text (a string stays quoted, a boolean/number stays bare), so the object is
     * reconstructed verbatim for the native flag. Insertion order is preserved.
     */
    private static Map<String, String> parseChatTemplateKwargs(String json, String flag) {
        final JsonNode root;
        try {
            root = CHAT_TEMPLATE_KWARGS_MAPPER.readTree(json);
        } catch (JsonProcessingException e) {
            throw error(flag + " expects a JSON object (e.g. {\"reasoning_effort\":\"low\"}), got: " + json, e);
        }
        if (root == null || !root.isObject()) {
            throw error(flag + " expects a JSON object (e.g. {\"reasoning_effort\":\"low\"}), got: " + json);
        }
        final Map<String, String> kwargs = new LinkedHashMap<>();
        for (final Map.Entry<String, JsonNode> field : root.properties()) {
            kwargs.put(field.getKey(), field.getValue().toString());
        }
        return Collections.unmodifiableMap(kwargs);
    }

    private static IllegalArgumentException error(String message) {
        return error(message, null);
    }

    private static IllegalArgumentException error(String message, @Nullable Throwable cause) {
        return new IllegalArgumentException(message + System.lineSeparator() + System.lineSeparator() + usage(), cause);
    }

    /**
     * Immutable, parsed launcher options. The integer tuning knobs — {@code ctxSize},
     * {@code threads}, {@code threadsBatch}, {@code parallel}, {@code batchSize} and
     * {@code ubatchSize} — use {@code 0} as a sentinel meaning "leave the llama.cpp default", and are
     * only applied to {@link ModelParameters} when positive. {@code cacheTypeK}/{@code cacheTypeV}
     * and {@code chatTemplateKwargs} use {@code null} as the same "leave the default" sentinel.
     * {@code gpuLayers} is always applied (its own default of {@code 0} already means CPU-only).
     */
    public static final class Options {

        private final String host;
        private final int port;
        private final String modelPath;
        private final @Nullable String modelId;
        private final @Nullable String apiKey;
        private final @Nullable String mmproj;
        private final @Nullable String mmprojDevice;
        private final int ctxSize;
        private final int gpuLayers;
        private final int threads;
        private final int threadsBatch;
        private final int parallel;
        private final int batchSize;
        private final int ubatchSize;
        private final @Nullable CacheType cacheTypeK;
        private final @Nullable CacheType cacheTypeV;
        private final boolean jinja;
        private final @Nullable Map<String, String> chatTemplateKwargs;
        private final boolean embedding;
        private final boolean reranking;

        private Options(
                String host,
                int port,
                String modelPath,
                @Nullable String modelId,
                @Nullable String apiKey,
                @Nullable String mmproj,
                @Nullable String mmprojDevice,
                int ctxSize,
                int gpuLayers,
                int threads,
                int threadsBatch,
                int parallel,
                int batchSize,
                int ubatchSize,
                @Nullable CacheType cacheTypeK,
                @Nullable CacheType cacheTypeV,
                boolean jinja,
                @Nullable Map<String, String> chatTemplateKwargs,
                boolean embedding,
                boolean reranking) {
            this.host = host;
            this.port = port;
            this.modelPath = modelPath;
            this.modelId = modelId;
            this.apiKey = apiKey;
            this.mmproj = mmproj;
            this.mmprojDevice = mmprojDevice;
            this.ctxSize = ctxSize;
            this.gpuLayers = gpuLayers;
            this.threads = threads;
            this.threadsBatch = threadsBatch;
            this.parallel = parallel;
            this.batchSize = batchSize;
            this.ubatchSize = ubatchSize;
            this.cacheTypeK = cacheTypeK;
            this.cacheTypeV = cacheTypeV;
            this.jinja = jinja;
            this.chatTemplateKwargs = chatTemplateKwargs;
            this.embedding = embedding;
            this.reranking = reranking;
        }

        /**
         * The interface to bind.
         *
         * @return the bind host
         */
        public String getHost() {
            return host;
        }

        /**
         * The TCP port to listen on.
         *
         * @return the port
         */
        public int getPort() {
            return port;
        }

        /**
         * The path to the GGUF model file to load.
         *
         * @return the model path
         */
        public String getModelPath() {
            return modelPath;
        }

        /**
         * The advertised model id, resolved from {@code --model-id} or derived from the model file name.
         *
         * @return the model id reported by {@code GET /v1/models}
         */
        public String getModelId() {
            if (modelId != null) {
                return modelId;
            }
            final Path name = Paths.get(modelPath).getFileName();
            return name != null ? name.toString() : OpenAiServerConfig.DEFAULT_MODEL_ID;
        }

        /**
         * The optional bearer API key.
         *
         * @return the API key, or {@code null} when authentication is disabled
         */
        public @Nullable String getApiKey() {
            return apiKey;
        }

        /**
         * The optional multimodal projector path for vision models.
         *
         * @return the mmproj path, or {@code null} when no vision projector is configured
         */
        public @Nullable String getMmproj() {
            return mmproj;
        }

        /**
         * The device the multimodal projector should run on. Independent of the main model's
         * placement; the literal {@code "none"} keeps the projector on the CPU.
         *
         * @return the projector device name, or {@code null} to let llama.cpp choose
         */
        public @Nullable String getMmprojDevice() {
            return mmprojDevice;
        }

        /**
         * The context window size, or {@code 0} for the llama.cpp default.
         *
         * @return the context size
         */
        public int getCtxSize() {
            return ctxSize;
        }

        /**
         * The number of layers to offload to the GPU ({@code 0} = CPU-only).
         *
         * @return the GPU layer count
         */
        public int getGpuLayers() {
            return gpuLayers;
        }

        /**
         * The inference thread count, or {@code 0} for the llama.cpp default.
         *
         * @return the thread count
         */
        public int getThreads() {
            return threads;
        }

        /**
         * The number of parallel inference slots, or {@code 0} for the llama.cpp default.
         *
         * @return the parallel slot count
         */
        public int getParallel() {
            return parallel;
        }

        /**
         * The batch/prompt-processing thread count, or {@code 0} for the llama.cpp default (same as
         * {@link #getThreads()}).
         *
         * @return the batch thread count
         */
        public int getThreadsBatch() {
            return threadsBatch;
        }

        /**
         * The logical (prompt) batch size, or {@code 0} for the llama.cpp default.
         *
         * @return the batch size
         */
        public int getBatchSize() {
            return batchSize;
        }

        /**
         * The physical (micro) batch size, or {@code 0} for the llama.cpp default.
         *
         * @return the micro-batch size
         */
        public int getUbatchSize() {
            return ubatchSize;
        }

        /**
         * The KV cache K quantization type, or {@code null} for the llama.cpp default.
         *
         * @return the K cache type, or {@code null} when unset
         */
        public @Nullable CacheType getCacheTypeK() {
            return cacheTypeK;
        }

        /**
         * The KV cache V quantization type, or {@code null} for the llama.cpp default.
         *
         * @return the V cache type, or {@code null} when unset
         */
        public @Nullable CacheType getCacheTypeV() {
            return cacheTypeV;
        }

        /**
         * Whether the model's Jinja chat template is enabled.
         *
         * @return {@code true} if {@code --jinja} was requested
         */
        public boolean isJinja() {
            return jinja;
        }

        /**
         * The parsed {@code --chat-template-kwargs} as a raw-per-value map (see
         * {@link ModelParameters#setChatTemplateKwargs(Map)}), or {@code null} when unset. The map is
         * unmodifiable.
         *
         * @return the chat-template variables, or {@code null} when unset
         */
        public @Nullable Map<String, String> getChatTemplateKwargs() {
            return chatTemplateKwargs;
        }

        /**
         * Whether to load the model in embedding mode.
         *
         * @return {@code true} if embedding mode is requested
         */
        public boolean isEmbedding() {
            return embedding;
        }

        /**
         * Whether to load the model in reranking mode.
         *
         * @return {@code true} if reranking mode is requested
         */
        public boolean isReranking() {
            return reranking;
        }

        /**
         * Build the {@link ModelParameters} for loading the model described by these options.
         *
         * @return the model parameters
         */
        public ModelParameters toModelParameters() {
            final ModelParameters params =
                    new ModelParameters().setModel(modelPath).setGpuLayers(gpuLayers);
            if (mmprojDevice != null) {
                params.setMmprojDevice(mmprojDevice);
            }
            if (mmproj != null) {
                params.setMmproj(mmproj);
            }
            if (ctxSize > 0) {
                params.setCtxSize(ctxSize);
            }
            if (threads > 0) {
                params.setThreads(threads);
            }
            if (threadsBatch > 0) {
                params.setThreadsBatch(threadsBatch);
            }
            if (parallel > 0) {
                params.setParallel(parallel);
            }
            if (batchSize > 0) {
                params.setBatchSize(batchSize);
            }
            if (ubatchSize > 0) {
                params.setUbatchSize(ubatchSize);
            }
            if (cacheTypeK != null) {
                params.setCacheTypeK(cacheTypeK);
            }
            if (cacheTypeV != null) {
                params.setCacheTypeV(cacheTypeV);
            }
            if (jinja) {
                params.enableJinja();
            }
            if (chatTemplateKwargs != null) {
                params.setChatTemplateKwargs(chatTemplateKwargs);
            }
            if (embedding) {
                params.enableEmbedding();
            }
            if (reranking) {
                params.enableReranking();
            }
            return params;
        }

        /**
         * Build the {@link OpenAiServerConfig} describing the server side of these options. When a
         * context size is given, the advertised input/output token budgets are derived from it.
         *
         * @return the server configuration
         */
        public OpenAiServerConfig toServerConfig() {
            return toServerConfig(mmproj != null, "");
        }

        /**
         * Build the server configuration with a capability value obtained from the loaded model.
         * This overload lets the standalone launcher avoid advertising vision merely because an
         * {@code --mmproj} path was supplied.
         *
         * @param supportsVision whether the loaded model reports usable vision input
         * @return the server configuration
         */
        public OpenAiServerConfig toServerConfig(boolean supportsVision) {
            return toServerConfig(supportsVision, "");
        }

        /**
         * Build the server configuration with capability + metadata values obtained from the loaded
         * model. This overload lets the standalone launcher advertise the model's quantization file
         * type in {@code /v1/models} alongside the vision capability.
         *
         * @param supportsVision whether the loaded model reports usable vision input
         * @param modelFtype the model's file-type (quantization) label, or {@code ""} if unknown
         * @return the server configuration
         */
        public OpenAiServerConfig toServerConfig(boolean supportsVision, String modelFtype) {
            final OpenAiServerConfig.Builder builder = OpenAiServerConfig.builder()
                    .host(host)
                    .port(port)
                    .modelId(getModelId())
                    .supportsVision(supportsVision)
                    .modelFtype(modelFtype);
            if (apiKey != null) {
                builder.apiKey(apiKey);
            }
            if (ctxSize > 0) {
                builder.maxOutputTokens(
                        Math.min(OpenAiServerConfig.DEFAULT_MAX_OUTPUT_TOKENS, Math.max(1, ctxSize / 2)));
                builder.maxInputTokens(Math.max(1, ctxSize - OpenAiServerConfig.DEFAULT_MAX_OUTPUT_TOKENS));
            }
            return builder.build();
        }
    }
}
