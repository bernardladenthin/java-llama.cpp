// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;

/**
 * Every text the agent sends to the model, and the texts it shows to the user, read from the resources
 * next to this class.
 *
 * <p>The wording lives in {@code src/main/resources/net/ladenthin/llama/atmosphere/*.txt} so it can be read
 * and edited as text; {@code {placeholders}} are filled in by the callers. It is shared by every front end
 * — the consoles, the browser and the editor protocol all send the model the same words.
 */
public final class Prompts {

    /** The default system prompt; placeholders {@code {workspace}} and {@code {shell_section}}. */
    public static final String SYSTEM_PROMPT = "system-prompt.txt";

    /** The {@code {shell_section}} with {@code --allow-shell}; placeholder {@code {shell}}. */
    public static final String SHELL_PROMPT = "system-prompt-shell.txt";

    /** The {@code {shell_section}} without {@code --allow-shell}. */
    public static final String NO_SHELL_PROMPT = "system-prompt-no-shell.txt";

    /** The {@code /help} overview. */
    public static final String HELP_TEXT = "help.txt";

    /** The per-step message of {@code /loop}; placeholders {@code {task}}, {@code {file}}, {@code {check_hint}}. */
    public static final String LOOP_PROMPT = "loop-prompt.txt";

    /** The skeleton written to {@link TaskLoop#LOOP_FILE}; placeholder {@code {task}}. */
    public static final String LOOP_FILE_TEMPLATE = "loop-file-template.md";

    /** The instructions {@code /compact} sends; placeholder {@code {focus}}. */
    public static final String COMPACT_PROMPT = "compact-prompt.txt";

    private Prompts() {}

    /**
     * The default system prompt, or the {@code --system} override.
     *
     * <p>The default describes a general-purpose agent on this machine, not a coding agent confined to a
     * project: a small model reads a narrow role or tool description as a prohibition and then refuses
     * requests such as "list the docker images" even though {@code run_command} could do it. With
     * {@code --allow-shell} the prompt therefore states that any command line is allowed and that the
     * model should run a command rather than explain one; without it, the prompt says so honestly
     * instead of letting the model invent a limitation. The text itself is in the resources
     * {@value #SYSTEM_PROMPT}, {@value #SHELL_PROMPT} and {@value #NO_SHELL_PROMPT} (see {@link #prompt}).
     *
     * @param options the options
     * @return the system prompt
     */
    public static String systemPrompt(AgentOptions options) {
        if (options.getSystemPrompt() != null) {
            return options.getSystemPrompt();
        }
        String shellSection = options.isAllowShell()
                ? prompt(SHELL_PROMPT).replace("{shell}", ShellTool.shellName())
                : prompt(NO_SHELL_PROMPT);
        return prompt(SYSTEM_PROMPT)
                .replace("{workspace}", options.getWorkspace().toString())
                .replace("{shell_section}", shellSection);
    }

    /**
     * A text from the resources next to this class, trimmed.
     *
     * @param name the file name, e.g. {@value #SYSTEM_PROMPT}
     * @return the file content without leading or trailing whitespace
     * @throws IllegalStateException when the resource is missing from the jar
     */
    public static String prompt(String name) {
        try (InputStream in = Prompts.class.getResourceAsStream(name)) {
            if (in == null) {
                throw new IllegalStateException("Prompt resource missing: " + name);
            }
            return new String(in.readAllBytes(), StandardCharsets.UTF_8).strip();
        } catch (IOException e) {
            throw new UncheckedIOException("Cannot read prompt resource " + name, e);
        }
    }
}
