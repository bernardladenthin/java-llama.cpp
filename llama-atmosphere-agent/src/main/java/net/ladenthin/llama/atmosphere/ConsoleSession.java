// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import org.atmosphere.ai.fs.AgentFileSystem;

/**
 * One agent turn shown on a console: a {@link TurnRecorder} in front of a {@link ConsoleRenderer}.
 *
 * <p>The recording half is the one every front end shares; this class only fixes the rendering half to
 * the console, which is what the terminal front ends and most tests want.
 */
public final class ConsoleSession extends TurnRecorder {

    /**
     * Create a session writing to {@code terminal}.
     *
     * @param terminal where streamed text and tool lines go
     * @param fileSystem the workspace-confined filesystem handed to the file tools
     */
    public ConsoleSession(AgentTerminal terminal, AgentFileSystem fileSystem) {
        super(fileSystem, new ConsoleRenderer(terminal));
    }
}
