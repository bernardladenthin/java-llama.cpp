// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.util.Map;
import org.atmosphere.ai.AiEvent;
import org.atmosphere.ai.StreamingSession;

/**
 * Shows one agent turn on a console: streamed text as it arrives, one line per tool call and tool result,
 * and the error a turn ended with.
 *
 * <p>Rendering only. What the session has to remember of a turn is kept by the {@link TurnRecorder} in
 * front of this, so the console, the browser and the editor protocol all record the same way and differ
 * only in how they show it.
 */
public final class ConsoleRenderer implements StreamingSession {

    private static final int RESULT_PREVIEW_CHARS = 400;

    /** How much of a call's arguments the console shows; the model still gets them in full. */
    private static final int ARGUMENT_PREVIEW_CHARS = 200;

    /** How much of a single argument value survives, so one big one cannot hide the others. */
    private static final int ARGUMENT_VALUE_PREVIEW_CHARS = 80;

    private final AgentTerminal terminal;
    private final Ansi ansi;
    private final MarkdownConsole markdown;

    /**
     * Render to {@code terminal}.
     *
     * @param terminal where streamed text and tool lines go
     */
    public ConsoleRenderer(AgentTerminal terminal) {
        this.terminal = terminal;
        this.ansi = terminal.ansi();
        this.markdown = new MarkdownConsole(terminal::line, ansi);
    }

    @Override
    public String sessionId() {
        return "console";
    }

    @Override
    public void send(String chunk) {
        markdown.append(chunk);
    }

    @Override
    public void sendMetadata(String key, Object value) {
        // model id, tool-call argument deltas: not shown on the console
    }

    @Override
    public void progress(String message) {
        // "Connecting to built-in..." and friends: not shown on the console
    }

    @Override
    public void complete() {
        markdown.flush();
    }

    @Override
    public void complete(String summary) {
        complete();
    }

    @Override
    public void error(Throwable t) {
        markdown.flush();
        terminal.line(ansi.red("[error] " + t));
    }

    @Override
    public boolean isClosed() {
        return false;
    }

    @Override
    public void emit(AiEvent event) {
        switch (event) {
            case AiEvent.ToolStart start -> {
                markdown.flush();
                terminal.line(ansi.green("●") + " " + ansi.bold(start.toolName()) + " "
                        + ansi.dim(describeArguments(start.arguments())));
            }
            case AiEvent.ToolResult result ->
                terminal.line(ansi.dim("  ↳ " + cut(String.valueOf(result.result()), RESULT_PREVIEW_CHARS)));
            case AiEvent.ToolError error ->
                terminal.line(ansi.red("  ↳ error: " + cut(String.valueOf(error.error()), RESULT_PREVIEW_CHARS)));
            default -> {
                // the console asks for approvals itself and shows nothing else of the event stream
            }
        }
    }

    /**
     * The arguments of a call, short enough for one console line.
     *
     * <p>Every value is cut <em>on its own</em> before the whole thing is. Cutting only the rendered
     * map would let one big argument push the others out of the line — a {@code write_file} call would
     * then show half of the file and not the name of the file, which is the one thing worth seeing.
     *
     * @param arguments what the model passed, usually a map
     * @return one line
     */
    static String describeArguments(Object arguments) {
        if (!(arguments instanceof Map<?, ?> map)) {
            return cut(String.valueOf(arguments), ARGUMENT_PREVIEW_CHARS);
        }
        StringBuilder rendered = new StringBuilder("{");
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            if (rendered.length() > 1) {
                rendered.append(", ");
            }
            rendered.append(entry.getKey())
                    .append('=')
                    .append(cut(String.valueOf(entry.getValue()), ARGUMENT_VALUE_PREVIEW_CHARS));
        }
        return cut(rendered.append('}').toString(), ARGUMENT_PREVIEW_CHARS);
    }

    /**
     * Fold a value onto one line and cut it.
     *
     * <p>Both halves matter. The cut keeps a whole file out of the scrollback, and the folding keeps
     * the pinned block intact: that block is sized in <em>lines</em>, so a single printed "line"
     * carrying twenty newlines moves the screen twenty rows further than the terminal accounted for,
     * and the block ends up drawn across the output. A {@code write_file} call whose arguments contain
     * the file did exactly that.
     *
     * @param value the raw text
     * @param max how many characters survive
     * @return one line, with a note about what was left out
     */
    static String cut(String value, int max) {
        String oneLine = value.replace("\r\n", " ").replace('\n', ' ').replace('\r', ' ');
        return oneLine.length() <= max ? oneLine : oneLine.substring(0, max) + " … (" + value.length() + " chars)";
    }
}
