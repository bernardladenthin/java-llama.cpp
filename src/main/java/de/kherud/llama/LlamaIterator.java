package de.kherud.llama;

import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * This iterator is used by {@link LlamaModel#generate(InferenceParameters)} and
 * {@link LlamaModel#generateChat(InferenceParameters)}. In addition to implementing {@link Iterator},
 * it allows to cancel ongoing inference (see {@link #cancel()}).
 */
public final class LlamaIterator implements Iterator<LlamaOutput> {

    private final LlamaModel model;
    private final int taskId;

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
        byte[] bytes = model.receiveCompletionBytes(taskId);
        LlamaOutput output = LlamaOutput.fromBytes(bytes);
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
}
