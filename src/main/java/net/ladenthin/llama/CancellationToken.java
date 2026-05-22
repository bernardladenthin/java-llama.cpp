// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Cancellation handle for a blocking {@link LlamaModel} call. Pass an instance to
 * {@link LlamaModel#complete(InferenceParameters, CancellationToken)} and invoke
 * {@link #cancel()} from another thread to abort the inference loop.
 * <p>
 * A token may be reused across calls but is not thread-safe for concurrent
 * <em>publishing</em> &mdash; only one call at a time should bind it via the package-private
 * {@code bind} method. {@link #cancel()} and {@link #isCancelled()} are safe to call
 * concurrently with the inference loop.
 * </p>
 */
public final class CancellationToken {

    private static final int NO_TASK = -1;

    private final AtomicInteger taskId = new AtomicInteger(NO_TASK);
    private final AtomicReference<LlamaModel> bound = new AtomicReference<LlamaModel>();
    private volatile boolean cancelled;

    public CancellationToken() {
        // empty
    }

    /** Returns {@code true} once {@link #cancel()} has been called. */
    public boolean isCancelled() {
        return cancelled;
    }

    /**
     * Request cancellation. If the token is already bound to a running inference, the
     * underlying native task is cancelled immediately and the calling inference loop will
     * return on its next iteration. Idempotent.
     */
    public void cancel() {
        cancelled = true;
        LlamaModel m = bound.get();
        int id = taskId.get();
        if (m != null && id != NO_TASK) {
            m.cancelCompletion(id);
        }
    }

    /**
     * Bind this token to a running native task. Called by {@link LlamaModel} after the
     * task id has been allocated. If {@link #cancel()} was invoked before binding, the
     * native task is cancelled here.
     */
    void bind(LlamaModel model, int id) {
        bound.set(model);
        taskId.set(id);
        if (cancelled) {
            model.cancelCompletion(id);
        }
    }

    /** Clear binding after the call returns. Resets cancelled flag so the token can be reused. */
    void reset() {
        bound.set(null);
        taskId.set(NO_TASK);
        cancelled = false;
    }
}
