// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

/**
 * Cancellation handle for a blocking {@link LlamaModel} call. Pass an instance to
 * {@link LlamaModel#complete(InferenceParameters, CancellationToken)} and invoke
 * {@link #cancel()} from another thread to abort the inference loop.
 * <p>
 * Cancellation is cooperative: {@link #cancel()} only sets a flag, and the inference
 * loop checks that flag between generated tokens. Effective latency is therefore one
 * token interval (typically tens to a few hundred ms). The native task is <em>not</em>
 * unblocked mid-token because the underlying JNI reader cannot be safely freed while
 * another thread is blocked inside it.
 * </p>
 * <p>
 * A token may be reused across calls. {@link #cancel()} and {@link #isCancelled()} are
 * safe to invoke concurrently with the inference loop.
 * </p>
 */
public final class CancellationToken {

    private volatile boolean cancelled;

    /** Construct a fresh, not-cancelled token. */
    public CancellationToken() {
        // empty
    }

    /**
     * Cancellation flag accessor.
     * @return {@code true} once {@link #cancel()} has been called and before {@link #reset()}
     */
    public boolean isCancelled() {
        return cancelled;
    }

    /**
     * Request cancellation. Sets the flag observed by the inference loop; the loop will
     * return at its next token boundary. Idempotent and safe to call from any thread.
     */
    public void cancel() {
        cancelled = true;
    }

    /** Clear the cancelled flag so the token can be reused. Package-private. */
    void reset() {
        cancelled = false;
    }
}
