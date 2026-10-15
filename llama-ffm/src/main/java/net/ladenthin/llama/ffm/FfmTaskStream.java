// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
package net.ladenthin.llama.ffm;

import java.lang.invoke.MethodHandle;
import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * A streaming task of {@link FfmLlamaModel}: one JSON text per {@link #next()} until the server reports
 * the last one, after which the task is released. {@link #close()} releases it early (cancelling the
 * generation); it is idempotent, so a try-with-resources around a loop that ran to the end is fine.
 */
public final class FfmTaskStream implements Iterator<String>, AutoCloseable {

    private final FfmLlamaModel model;
    private final int taskId;
    private final MethodHandle receive;
    private boolean finished;
    private boolean released;

    FfmTaskStream(FfmLlamaModel model, int taskId, MethodHandle receive) {
        this.model = model;
        this.taskId = taskId;
        this.receive = receive;
    }

    /**
     * The task id the server assigned, for diagnostics.
     *
     * @return the task id
     */
    public int taskId() {
        return taskId;
    }

    @Override
    public boolean hasNext() {
        return !finished;
    }

    @Override
    public String next() {
        if (finished) {
            throw new NoSuchElementException("the task has delivered its last result");
        }
        boolean[] stop = new boolean[1];
        String result = model.receive(receive, taskId, stop);
        if (stop[0]) {
            finished = true;
            close();
        }
        return result;
    }

    @Override
    public void close() {
        if (!released) {
            released = true;
            finished = true;
            model.releaseTask(taskId);
        }
    }
}
