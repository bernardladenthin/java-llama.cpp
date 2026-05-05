package net.ladenthin.llama;

import net.ladenthin.llama.json.CompletionResponseParser;
import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * This iterator is used by {@link LlamaModel#generate(InferenceParameters)} and
 * {@link LlamaModel#generateChat(InferenceParameters)}. In addition to implementing {@link Iterator},
 * it allows to cancel ongoing inference (see {@link #cancel()}).
 *
 * <p>{@link LlamaIterator} implements {@link AutoCloseable}. When used via {@link LlamaIterable}
 * inside a try-with-resources block, {@link #close()} is called automatically on early exit
 * (e.g. {@code break}), preventing the native task slot from leaking.
 */
public final class LlamaIterator implements Iterator<LlamaOutput>, AutoCloseable {

    private final LlamaModel model;
    private final int taskId;
    private final CompletionResponseParser completionParser = new CompletionResponseParser();

    private boolean hasNext = true;

    LlamaIterator(LlamaModel model, InferenceParameters parameters) {
        this(model, parameters, false);
    }

    LlamaIterator(LlamaModel model, InferenceParameters parameters, boolean chat) {
        this.model = model;
        parameters.setStream(true);
        taskId = chat
                ? model.requestChatCompletion(parameters.toString())
                : model.requestCompletion(parameters.toString());
    }

    @Override
    public boolean hasNext() {
        return hasNext;
    }

    @Override
    public LlamaOutput next() {
        if (!hasNext) {
            throw new NoSuchElementException();
        }
        String json = model.receiveCompletionJson(taskId);
        LlamaOutput output = completionParser.parse(json);
        hasNext = !output.stop;
        if (output.stop) {
        	model.releaseTask(taskId);
        }
        return output;
    }

    /**
     * Cancel the ongoing generation process.
     */
    public void cancel() {
        model.cancelCompletion(taskId);
        hasNext = false;
    }

    /**
     * Cancels any in-progress generation if the iterator has not yet reached a stop token.
     * Safe to call multiple times — subsequent calls are no-ops.
     *
     * <p>Prefer using the enclosing {@link LlamaIterable} in a try-with-resources block rather
     * than calling this directly.
     */
    @Override
    public void close() {
        if (hasNext) {
            cancel();
        }
    }
}
