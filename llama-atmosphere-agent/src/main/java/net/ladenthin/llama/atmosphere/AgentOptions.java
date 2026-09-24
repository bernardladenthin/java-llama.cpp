// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.nio.file.Path;
import java.nio.file.Paths;
import org.jspecify.annotations.Nullable;

/**
 * Command-line options of {@link LocalAgent}, parsed without any framework so the parsing is a plain
 * unit-testable function.
 *
 * <p>Exactly one of {@code --base-url} (connect to a running OpenAI-compatible server, typically
 * java-llama.cpp's {@code NativeServer} or {@code OpenAiCompatServer}) and {@code --model} (load the
 * GGUF in this JVM and serve it to the agent over a loopback {@code OpenAiCompatServer}) must be
 * given.
 */
public final class AgentOptions {

    /** Bearer token sent by the agent; a local server that runs without {@code --api-key} ignores it. */
    public static final String DEFAULT_API_KEY = "sk-local";

    /** Model id carried in every request; single-model servers ignore it, a router selects by it. */
    public static final String DEFAULT_MODEL_ID = "local-model";

    /** Low temperature: coding agents want reproducible tool calls, not creative prose. */
    public static final double DEFAULT_TEMPERATURE = 0.2;

    /** Per-turn generation budget ({@code max_tokens}). */
    public static final int DEFAULT_MAX_TOKENS = 2048;

    /** Upper bound on model→tool→model rounds per user turn. */
    public static final int DEFAULT_MAX_TOOL_ROUNDS = 25;

    /** Context size for the in-process model ({@code --model}). */
    public static final int DEFAULT_CTX_SIZE = 8192;

    /** Whether the history is summarized on its own before it overflows the context. */
    public static final boolean DEFAULT_AUTO_COMPACT = true;

    /**
     * How full the context may get before that happens, in percent.
     *
     * <p>Lower than the ~85 % a hosted agent uses, and deliberately so: this number is usually an
     * estimate from the text length (llama.cpp reports its own count only to clients that ask for it),
     * and the reply still has to fit next to the prompt.
     */
    public static final int DEFAULT_COMPACT_AT = 70;

    /**
     * Log verbosity threshold of the in-process model ({@code --model}): llama.cpp's {@code -lv}
     * scale, {@code 0} output only, {@code 1} errors, {@code 2} warnings, {@code 3} info, {@code 4}
     * trace, {@code 5} debug. The default keeps warnings and errors but drops the per-request
     * {@code slot …} / {@code srv …} INFO lines, which otherwise interleave with the streamed answer
     * on the console (llama.cpp writes them to stderr).
     */
    public static final int DEFAULT_LOG_VERBOSITY = 2;

    private final @Nullable String baseUrl;
    private final @Nullable String modelPath;
    private final int gpuLayers;
    private final int ctxSize;
    private final int logVerbosity;
    private final boolean verbose;
    private final String apiKey;
    private final String modelId;
    private final Path workspace;
    private final boolean allowShell;
    private final boolean plain;
    private final boolean auto;
    private final boolean autoCompact;
    private final int compactAt;
    private final double temperature;
    private final int maxTokens;
    private final int maxToolRounds;
    private final @Nullable String systemPrompt;
    private final @Nullable String prompt;
    private final boolean help;

    private AgentOptions(Builder b) {
        this.baseUrl = b.baseUrl;
        this.modelPath = b.modelPath;
        this.gpuLayers = b.gpuLayers;
        this.ctxSize = b.ctxSize;
        this.logVerbosity = b.logVerbosity;
        this.verbose = b.verbose;
        this.apiKey = b.apiKey;
        this.modelId = b.modelId;
        this.workspace = b.workspace;
        this.allowShell = b.allowShell;
        this.plain = b.plain;
        this.auto = b.auto;
        this.autoCompact = b.autoCompact;
        this.compactAt = b.compactAt;
        this.temperature = b.temperature;
        this.maxTokens = b.maxTokens;
        this.maxToolRounds = b.maxToolRounds;
        this.systemPrompt = b.systemPrompt;
        this.prompt = b.prompt;
        this.help = b.help;
    }

    /**
     * Parse the command line.
     *
     * @param args the raw arguments
     * @return the parsed options
     * @throws IllegalArgumentException on an unknown flag, a missing value, or when neither/both of
     *     {@code --base-url} and {@code --model} are given
     */
    public static AgentOptions parse(String[] args) {
        Builder b = new Builder();
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            switch (a) {
                case "-h", "--help" -> b.help = true;
                case "--allow-shell" -> b.allowShell = true;
                case "--plain" -> b.plain = true;
                case "--auto" -> b.auto = true;
                case "--auto-compact" -> b.autoCompact = booleanValue(args, ++i, a);
                case "--compact-at" -> b.compactAt = percentValue(args, ++i, a);
                case "--base-url" -> b.baseUrl = stripTrailingSlash(value(args, ++i, a));
                case "--model" -> b.modelPath = value(args, ++i, a);
                case "--ngl", "--gpu-layers" -> b.gpuLayers = intValue(args, ++i, a);
                case "--ctx-size" -> b.ctxSize = intValue(args, ++i, a);
                case "--log-verbosity" -> b.logVerbosity = intValue(args, ++i, a);
                case "--verbose", "-v" -> b.verbose = true;
                case "--api-key" -> b.apiKey = value(args, ++i, a);
                case "--model-id" -> b.modelId = value(args, ++i, a);
                case "--workspace" ->
                    b.workspace =
                            Paths.get(value(args, ++i, a)).toAbsolutePath().normalize();
                case "--temperature" -> b.temperature = Double.parseDouble(value(args, ++i, a));
                case "--max-tokens" -> b.maxTokens = intValue(args, ++i, a);
                case "--max-tool-rounds" -> b.maxToolRounds = intValue(args, ++i, a);
                case "--system" -> b.systemPrompt = value(args, ++i, a);
                case "--prompt", "-p" -> b.prompt = value(args, ++i, a);
                default -> throw new IllegalArgumentException("Unknown argument: " + a);
            }
        }
        if (!b.help) {
            if ((b.baseUrl == null) == (b.modelPath == null)) {
                throw new IllegalArgumentException("Exactly one of --base-url <url> or --model <gguf> is required");
            }
        }
        return new AgentOptions(b);
    }

    private static String value(String[] args, int index, String flag) {
        if (index >= args.length) {
            throw new IllegalArgumentException("Missing value for " + flag);
        }
        return args[index];
    }

    private static boolean booleanValue(String[] args, int index, String flag) {
        String raw = value(args, index, flag).trim();
        if ("true".equalsIgnoreCase(raw) || "yes".equalsIgnoreCase(raw) || "on".equalsIgnoreCase(raw)) {
            return true;
        }
        if ("false".equalsIgnoreCase(raw) || "no".equalsIgnoreCase(raw) || "off".equalsIgnoreCase(raw)) {
            return false;
        }
        throw new IllegalArgumentException("Expected true or false for " + flag + ", got: " + raw);
    }

    private static int percentValue(String[] args, int index, String flag) {
        int percent = intValue(args, index, flag);
        if (percent < 10 || percent > 95) {
            throw new IllegalArgumentException(flag + " must be between 10 and 95, got: " + percent);
        }
        return percent;
    }

    private static int intValue(String[] args, int index, String flag) {
        String raw = value(args, index, flag);
        try {
            return Integer.parseInt(raw);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Expected an integer for " + flag + ", got: " + raw, e);
        }
    }

    private static String stripTrailingSlash(String url) {
        return url.endsWith("/") ? url.substring(0, url.length() - 1) : url;
    }

    /**
     * The usage text.
     *
     * @return one line per option
     */
    public static String usage() {
        return String.join(
                System.lineSeparator(),
                "Usage: LocalAgent (--base-url <url> | --model <file.gguf>) [options]",
                "",
                "Endpoint (exactly one):",
                "  --base-url <url>        OpenAI-compatible base URL of a running server,",
                "                          e.g. http://127.0.0.1:8080/v1 (java-llama.cpp NativeServer",
                "                          started with --jinja, OpenAiCompatServer, or llama-server)",
                "  --model <file.gguf>     load this GGUF in-process and serve it to the agent",
                "  --ngl <n>               GPU layers for --model (default 0 = CPU only)",
                "  --ctx-size <n>          context size for --model (default " + DEFAULT_CTX_SIZE + ")",
                "  --log-verbosity <n>     llama.cpp log threshold for --model: 1 errors, 2 warnings,",
                "                          3 info, 4 trace, 5 debug (default " + DEFAULT_LOG_VERBOSITY + ")",
                "  --verbose, -v           log everything for --model (same as llama-server -v)",
                "",
                "Agent:",
                "  --workspace <dir>       directory the file tools are confined to (default: cwd)",
                "  --allow-shell           add the run_command tool (runs any command line, starting in the workspace)",
                "  --plain                 line-oriented console: no pinned block, no cursor control",
                "  --auto                  run tools without asking (default: ask before writes and commands)",
                "  --auto-compact <bool>   summarize the history before it overflows the context (default "
                        + DEFAULT_AUTO_COMPACT + ")",
                "  --compact-at <percent>  how full the context may get first (default " + DEFAULT_COMPACT_AT + ")",
                "  --system <text>         replace the default system prompt",
                "  --prompt <text>, -p     run one turn and exit (default: interactive; /exit to quit)",
                "  --temperature <t>       sampling temperature (default " + DEFAULT_TEMPERATURE + ")",
                "  --max-tokens <n>        max_tokens per model call (default " + DEFAULT_MAX_TOKENS + ")",
                "  --max-tool-rounds <n>   tool rounds per turn (default " + DEFAULT_MAX_TOOL_ROUNDS + ")",
                "  --api-key <key>         bearer token (default " + DEFAULT_API_KEY + ")",
                "  --model-id <id>         model id in requests (default " + DEFAULT_MODEL_ID + ")",
                "  -h, --help              this text");
    }

    /**
     * External endpoint, or {@code null} in in-process mode.
     *
     * @return the base URL without a trailing slash
     */
    public @Nullable String getBaseUrl() {
        return baseUrl;
    }

    /**
     * GGUF to load in-process, or {@code null} when connecting to an external server.
     *
     * @return the model path
     */
    public @Nullable String getModelPath() {
        return modelPath;
    }

    /**
     * GPU layers for the in-process model.
     *
     * @return the layer count, {@code 0} for CPU only
     */
    public int getGpuLayers() {
        return gpuLayers;
    }

    /**
     * Context size for the in-process model.
     *
     * @return the context size in tokens
     */
    public int getCtxSize() {
        return ctxSize;
    }

    /**
     * Log verbosity threshold for the in-process model.
     *
     * @return the {@code -lv} threshold; ignored when {@link #isVerbose()} is set
     */
    public int getLogVerbosity() {
        return logVerbosity;
    }

    /**
     * Whether {@code --verbose} was given.
     *
     * @return {@code true} to log every message of the in-process model
     */
    public boolean isVerbose() {
        return verbose;
    }

    /**
     * Bearer token.
     *
     * @return the API key
     */
    public String getApiKey() {
        return apiKey;
    }

    /**
     * Model id sent in requests.
     *
     * @return the model id
     */
    public String getModelId() {
        return modelId;
    }

    /**
     * Root directory of the file tools.
     *
     * @return the absolute, normalized workspace path
     */
    public Path getWorkspace() {
        return workspace;
    }

    /**
     * Whether the history is summarized before it overflows the context.
     *
     * @return {@code true} when auto-compaction is on
     */
    public boolean isAutoCompact() {
        return autoCompact;
    }

    /**
     * How full the context may get before the history is summarized.
     *
     * @return the threshold in percent
     */
    public int getCompactAt() {
        return compactAt;
    }

    /**
     * Whether tool calls run without asking.
     *
     * @return {@code true} when {@code --auto} was given, i.e. the session starts in
     *     {@link ApprovalMode#AUTO}
     */
    public boolean isAuto() {
        return auto;
    }

    /**
     * Whether the {@code run_command} tool is registered.
     *
     * @return {@code true} when shell access was opted into
     */
    public boolean isAllowShell() {
        return allowShell;
    }

    /**
     * Whether to use the line-oriented console even when a full terminal is available.
     *
     * <p>The rich console positions the cursor: it pins a block to the bottom of the window and keeps
     * the input line there while output scrolls above it. That needs a terminal that reports its size
     * and understands the sequences, which is the normal case over SSH as well — but not in a plain
     * pipe, a CI log, a `dumb` terminal, an editor's run window or a serial console, and not when the
     * session is being recorded as text. This flag chooses the console that only ever appends lines,
     * which is also what the agent falls back to on its own when there is no usable terminal.
     *
     * @return {@code true} when {@code --plain} was passed
     */
    public boolean isPlain() {
        return plain;
    }

    /**
     * Sampling temperature.
     *
     * @return the temperature
     */
    public double getTemperature() {
        return temperature;
    }

    /**
     * Generation budget per model call.
     *
     * @return {@code max_tokens}
     */
    public int getMaxTokens() {
        return maxTokens;
    }

    /**
     * Tool-round cap per user turn.
     *
     * @return the maximum number of tool rounds
     */
    public int getMaxToolRounds() {
        return maxToolRounds;
    }

    /**
     * System prompt override.
     *
     * @return the prompt, or {@code null} for the built-in default
     */
    public @Nullable String getSystemPrompt() {
        return systemPrompt;
    }

    /**
     * One-shot prompt.
     *
     * @return the prompt, or {@code null} for interactive mode
     */
    public @Nullable String getPrompt() {
        return prompt;
    }

    /**
     * Whether {@code --help} was given.
     *
     * @return {@code true} to print usage and exit
     */
    public boolean isHelp() {
        return help;
    }

    @Override
    public String toString() {
        return "AgentOptions{baseUrl=" + baseUrl + ", modelPath=" + modelPath + ", gpuLayers=" + gpuLayers
                + ", ctxSize=" + ctxSize + ", logVerbosity=" + (verbose ? "verbose" : logVerbosity)
                + ", modelId=" + modelId + ", workspace=" + workspace
                + ", allowShell=" + allowShell + ", plain=" + plain + ", auto=" + auto + ", autoCompact=" + autoCompact
                + ", temperature="
                + temperature + ", maxTokens="
                + maxTokens
                + ", maxToolRounds=" + maxToolRounds + ", prompt=" + (prompt == null ? "<interactive>" : "<set>")
                + "}";
    }

    private static final class Builder {
        @Nullable
        String baseUrl;

        @Nullable
        String modelPath;

        int gpuLayers = 0;
        int ctxSize = DEFAULT_CTX_SIZE;
        int logVerbosity = DEFAULT_LOG_VERBOSITY;
        boolean verbose;
        String apiKey = DEFAULT_API_KEY;
        String modelId = DEFAULT_MODEL_ID;
        Path workspace = Paths.get("").toAbsolutePath().normalize();
        boolean allowShell;
        boolean plain;
        boolean auto;
        boolean autoCompact = DEFAULT_AUTO_COMPACT;
        int compactAt = DEFAULT_COMPACT_AT;
        double temperature = DEFAULT_TEMPERATURE;
        int maxTokens = DEFAULT_MAX_TOKENS;
        int maxToolRounds = DEFAULT_MAX_TOOL_ROUNDS;

        @Nullable
        String systemPrompt;

        @Nullable
        String prompt;

        boolean help;
    }
}
