package de.kherud.llama;

import org.jetbrains.annotations.NotNull;

/**
 * An {@link Iterable} wrapper around {@link LlamaIterator} returned by
 * {@link LlamaModel#generate(InferenceParameters)} and {@link LlamaModel#generateChat(InferenceParameters)}.
 *
 * <p>Implements {@link AutoCloseable} so that a try-with-resources block automatically cancels
 * any in-progress generation when the loop exits early (e.g. via {@code break}), preventing the
 * native task slot from leaking:
 *
 * <pre>{@code
 * try (LlamaIterable it = model.generate(params)) {
 *     for (LlamaOutput o : it) {
 *         if (done) break;   // close() cancels the native task automatically
 *     }
 * }
 * }</pre>
 *
 * <p>A plain for-each loop without try-with-resources continues to work; the {@link #close()}
 * method just will not be called on early exit in that case.
 */
public final class LlamaIterable implements Iterable<LlamaOutput>, AutoCloseable {

    private final LlamaIterator iterator;

    LlamaIterable(LlamaIterator iterator) {
        this.iterator = iterator;
    }

    @NotNull
    @Override
    public LlamaIterator iterator() {
        return iterator;
    }

    /**
     * Cancels any in-progress generation. Delegates to {@link LlamaIterator#close()}.
     * Safe to call multiple times.
     */
    @Override
    public void close() {
        iterator.close();
    }
}
