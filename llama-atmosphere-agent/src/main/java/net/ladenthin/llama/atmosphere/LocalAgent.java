// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.parameters.ModelParameters;
import net.ladenthin.llama.server.OpenAiCompatServer;
import net.ladenthin.llama.server.OpenAiServerConfig;
import org.atmosphere.ai.fs.AgentFileSystem;
import org.atmosphere.ai.fs.FileSystemTools;
import org.atmosphere.ai.fs.WorkspaceAgentFileSystem;
import org.atmosphere.ai.llm.ChatMessage;
import org.atmosphere.ai.tool.ToolDefinition;
import org.jspecify.annotations.Nullable;

/**
 * A local, general-purpose terminal agent in the spirit of Claude Code / OpenCode, built from two parts that
 * already exist: <b>Atmosphere</b>'s built-in OpenAI-compatible agent runtime (streaming, tool loop,
 * workspace file tools) and <b>java-llama.cpp</b>'s OpenAI-compatible server.
 *
 * <p>Two ways to reach a model:
 *
 * <ul>
 *   <li>{@code --base-url http://127.0.0.1:8080/v1} — a server you started yourself (java-llama.cpp's
 *       fat jar {@code NativeServer} with {@code --jinja}, its {@code OpenAiCompatServer}, or upstream
 *       {@code llama-server}), so you keep full control over model parameters.
 *   <li>{@code --model model.gguf} — loads the GGUF in this JVM and serves it to the agent over a
 *       loopback {@link OpenAiCompatServer}: one process, one command.
 * </ul>
 *
 * <p>Run from the source tree: {@code mvn -q compile exec:java -Dexec.args="--base-url ... --workspace
 * /path --allow-shell"}. Exit code 0 on a completed turn, 1 when the turn errored, 2 on bad usage.
 */
public final class LocalAgent {

    /** Wall-clock bound on one user turn, including every tool round. */
    private static final Duration TURN_TIMEOUT = Duration.ofMinutes(30);

    private static final Duration SHELL_TIMEOUT = Duration.ofSeconds(120);
    private static final int SHELL_MAX_OUTPUT_CHARS = 20_000;

    private LocalAgent() {}

    /**
     * Entry point.
     *
     * @param args see {@link AgentOptions#usage()}
     * @throws Exception on an unrecoverable setup failure (model load, socket bind)
     */
    public static void main(String[] args) throws Exception {
        AgentOptions options;
        try {
            options = AgentOptions.parse(args);
        } catch (IllegalArgumentException e) {
            System.err.println(e.getMessage());
            System.err.println(AgentOptions.usage());
            System.exit(2);
            return;
        }
        if (options.isHelp()) {
            System.out.println(AgentOptions.usage());
            return;
        }
        System.exit(run(
                options,
                System.in == null ? null : new InputStreamReader(System.in, StandardCharsets.UTF_8),
                System.out,
                System.err));
    }

    /**
     * Run the agent with parsed options.
     *
     * @param options the options
     * @param input the interactive input (ignored in one-shot mode), or {@code null} for none
     * @param out the console the answer streams to
     * @param err diagnostics
     * @return the process exit code
     * @throws Exception on an unrecoverable setup failure
     */
    static int run(AgentOptions options, java.io.@Nullable Reader input, PrintStream out, PrintStream err)
            throws Exception {
        LlamaModel model = null;
        OpenAiCompatServer server = null;
        String baseUrl = options.getBaseUrl();
        try {
            if (options.getModelPath() != null) {
                err.println("Loading " + options.getModelPath() + " (gpu layers: " + options.getGpuLayers() + ", ctx: "
                        + options.getCtxSize() + ") ...");
                model = new LlamaModel(modelParameters(options));
                server = new OpenAiCompatServer(
                                model,
                                OpenAiServerConfig.builder()
                                        .host("127.0.0.1")
                                        .port(0)
                                        .apiKey(options.getApiKey())
                                        .modelId(options.getModelId())
                                        .build())
                        .start();
                baseUrl = "http://127.0.0.1:" + server.getPort() + "/v1";
            }
            if (baseUrl == null) {
                throw new IllegalStateException("no endpoint");
            }
            AgentFileSystem fileSystem =
                    new WorkspaceAgentFileSystem(options.getWorkspace(), AgentFileSystem.Limits.defaults());
            List<ToolDefinition> tools = new ArrayList<>(FileSystemTools.all());
            if (options.isAllowShell()) {
                tools.add(ShellTool.definition(options.getWorkspace(), SHELL_TIMEOUT, SHELL_MAX_OUTPUT_CHARS));
            }
            AgentRunner runner = new AgentRunner(
                    baseUrl,
                    options.getApiKey(),
                    options.getModelId(),
                    tools,
                    systemPrompt(options),
                    options.getTemperature(),
                    options.getMaxTokens(),
                    options.getMaxToolRounds());
            err.println("Endpoint " + baseUrl + " models=" + runner.models() + " workspace=" + options.getWorkspace()
                    + " tools=" + runner.toolNames());

            List<ChatMessage> history = new ArrayList<>();
            if (options.getPrompt() != null) {
                return turn(runner, fileSystem, options.getPrompt(), history, out) ? 0 : 1;
            }
            if (input == null) {
                err.println("No interactive input available; pass --prompt <text>.");
                return 2;
            }
            BufferedReader reader = new BufferedReader(input);
            err.println("Interactive mode: type a request, /clear to drop the history, /exit to quit.");
            while (true) {
                out.print("you> ");
                out.flush();
                String line = reader.readLine();
                if (line == null || line.trim().equals("/exit") || line.trim().equals("/quit")) {
                    return 0;
                }
                if (line.trim().isEmpty()) {
                    continue;
                }
                if (line.trim().equals("/clear")) {
                    history.clear();
                    err.println("(history cleared)");
                    continue;
                }
                turn(runner, fileSystem, line, history, out);
            }
        } finally {
            if (server != null) {
                server.close();
            }
            if (model != null) {
                model.close();
            }
        }
    }

    private static boolean turn(
            AgentRunner runner, AgentFileSystem fileSystem, String message, List<ChatMessage> history, PrintStream out)
            throws InterruptedException {
        ConsoleSession session = new ConsoleSession(out, fileSystem);
        runner.run(message, history, session);
        boolean finished = session.await(TURN_TIMEOUT);
        history.add(ChatMessage.user(message));
        if (!session.text().isEmpty()) {
            history.add(ChatMessage.assistant(session.text()));
        }
        return finished && session.failure() == null;
    }

    /**
     * The native parameters for {@code --model}.
     *
     * <p>Visible for tests: the log threshold is the one knob whose effect is only observable on a
     * console, so the test pins the flags that leave here instead.
     *
     * @param options the parsed options
     * @return the parameters the in-process {@link LlamaModel} is loaded with
     */
    static ModelParameters modelParameters(AgentOptions options) {
        ModelParameters parameters = new ModelParameters()
                .setModel(options.getModelPath())
                .setCtxSize(options.getCtxSize())
                .setGpuLayers(options.getGpuLayers())
                .setFit(false)
                // Jinja rendering is what lets the native parser apply the model's tool-call template.
                .enableJinja();
        // llama.cpp logs to stderr, which shares the console with the streamed answer on stdout; the
        // default threshold keeps warnings and errors and drops the per-request INFO lines.
        if (options.isVerbose()) {
            parameters.setVerbose();
        } else {
            parameters.setLogVerbosity(options.getLogVerbosity());
        }
        if (options.getGpuLayers() == 0) {
            parameters.setDevices("none");
        }
        return parameters;
    }

    /**
     * The default system prompt, or the {@code --system} override.
     *
     * <p>The default describes a general-purpose agent on this machine, not a coding agent confined to a
     * project: a small model reads a narrow role or tool description as a prohibition and then refuses
     * requests such as "list the docker images" even though {@code run_command} could do it. With
     * {@code --allow-shell} the prompt therefore states that any command line is allowed and that the
     * model should run a command rather than explain one; without it, the prompt says so honestly
     * instead of letting the model invent a limitation.
     *
     * @param options the options
     * @return the system prompt
     */
    static String systemPrompt(AgentOptions options) {
        if (options.getSystemPrompt() != null) {
            return options.getSystemPrompt();
        }
        String files = " The file tools ls, read_file, write_file, edit_file, glob, grep, delete and rename work"
                + " on the directory " + options.getWorkspace() + "; their paths are relative to it.";
        String shell = options.isAllowShell()
                ? " You have full shell access: run_command executes any command line through "
                        + ShellTool.shellName() + " on this machine, starting in that directory but not"
                        + " limited to it, e.g. docker, git, package managers, build tools or system"
                        + " information. When the user asks about this machine or wants something done,"
                        + " run the command instead of explaining how to do it."
                : " You cannot run shell commands in this session; if a request needs one, say so and"
                        + " suggest restarting the agent with --allow-shell.";
        return "You are a helpful general-purpose assistant running locally on the user's computer, with"
                + " tools to act on it." + files + shell
                + " Work step by step: read a file before you edit it, check the result after a change,"
                + " and finish with a short summary. Answer in the user's language.";
    }
}
