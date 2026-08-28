// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

/**
 * On-demand reading of tensors the model architecture marks as lazy-loadable, such as per-layer
 * embeddings.
 *
 * <p>The string constants are the exact values accepted by llama.cpp's {@code --tensor-read-lazy}
 * CLI argument (added in b10679), and map 1-to-1 to the {@code llama_lazy_mode} enum in
 * {@code include/llama.h}. Reading rows on demand keeps a large marked tensor out of resident
 * memory at the cost of disk reads during inference; it <strong>requires mmap</strong>, so it has
 * no effect when the model is loaded with mmap disabled.
 *
 * @see net.ladenthin.llama.parameters.ModelParameters#setTensorReadLazy(TensorReadLazyMode)
 */
public enum TensorReadLazyMode implements CliArg {

    /**
     * Always read a marked tensor up front and keep it resident.
     *
     * <p>CLI string: {@code "off"} — maps to {@code LLAMA_LAZY_MODE_OFF = 0}.
     */
    OFF("off"),

    /**
     * Read rows on demand, but only for marked tensors larger than 4 GiB.
     *
     * <p>CLI string: {@code "auto"} — maps to {@code LLAMA_LAZY_MODE_AUTO = 1}. This is
     * upstream's default, so passing it is equivalent to omitting the flag.
     */
    AUTO("auto"),

    /**
     * Read the rows of every marked tensor on demand, regardless of size.
     *
     * <p>CLI string: {@code "on"} — maps to {@code LLAMA_LAZY_MODE_ON = 2}.
     */
    ON("on");

    /**
     * The CLI string passed to {@code --tensor-read-lazy} in llama.cpp's {@code common/arg.cpp}.
     */
    private final String argValue;

    TensorReadLazyMode(String value) {
        this.argValue = value;
    }

    /**
     * Returns the CLI string accepted by llama.cpp's {@code --tensor-read-lazy} argument.
     *
     * @return the mode string ({@code "off"}, {@code "auto"} or {@code "on"})
     */
    @Override
    public String getArgValue() {
        return argValue;
    }
}
