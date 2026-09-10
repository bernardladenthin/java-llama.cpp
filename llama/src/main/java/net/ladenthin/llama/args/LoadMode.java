// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

/**
 * How llama.cpp brings the model weights into memory.
 *
 * <p>The string constants are the exact values accepted by llama.cpp's {@code -lm}/{@code --load-mode}
 * CLI argument, and map 1-to-1 to the {@code llama_load_mode} enum in {@code include/llama.h}.
 *
 * <p>This single option replaced the older independent switches. Upstream deprecated
 * {@code --mlock}, {@code --mmap}/{@code --no-mmap} and {@code -dio}/{@code --direct-io} at b10092
 * and <strong>deleted them at b10878</strong> — the whole deprecation window opened and closed
 * inside eight tags. Because llama.cpp's argument parser treats an unknown option as a hard error
 * rather than a warning, the deleted spellings do not degrade a model load, they prevent it. The
 * {@code ModelParameters} methods that used to emit them were removed together with the flags;
 * pass {@link #MLOCK} (was {@code --mlock}) or {@link #NONE} (was {@code --no-mmap}) to
 * {@link net.ladenthin.llama.parameters.ModelParameters#setLoadMode(LoadMode)} instead.
 *
 * @see net.ladenthin.llama.parameters.ModelParameters#setLoadMode(LoadMode)
 */
public enum LoadMode implements CliArg {

    /**
     * Let llama.cpp choose — mmap unless a device does not support it.
     *
     * <p>CLI string: {@code "auto"} — maps to {@code LLAMA_LOAD_MODE_AUTO = -1}. This is
     * upstream's default, so passing it is equivalent to omitting the flag.
     */
    AUTO("auto"),

    /**
     * No special loading mode: read the weights normally, without mmap and without mlock.
     *
     * <p>CLI string: {@code "none"} — maps to {@code LLAMA_LOAD_MODE_NONE = 0}. This is what the
     * removed {@code --no-mmap} mapped to in upstream's own deprecation shim.
     */
    NONE("none"),

    /**
     * Memory-map the model.
     *
     * <p>CLI string: {@code "mmap"} — maps to {@code LLAMA_LOAD_MODE_MMAP = 1}. Loading is fast and
     * pages are shared, but pages can be evicted again under memory pressure; combine with
     * {@link #MMAP_MLOCK} to prevent that.
     */
    MMAP("mmap"),

    /**
     * Force the system to keep the model in RAM rather than swapping or compressing it.
     *
     * <p>CLI string: {@code "mlock"} — maps to {@code LLAMA_LOAD_MODE_MLOCK = 2}. This is what the
     * removed {@code --mlock} mapped to in upstream's own deprecation shim.
     */
    MLOCK("mlock"),

    /**
     * Memory-map the model <em>and</em> lock it into RAM.
     *
     * <p>CLI string: {@code "mmap+mlock"} — maps to {@code LLAMA_LOAD_MODE_MMAP_MLOCK = 3}.
     */
    MMAP_MLOCK("mmap+mlock"),

    /**
     * Use direct I/O if the platform supports it, bypassing the page cache.
     *
     * <p>CLI string: {@code "dio"} — maps to {@code LLAMA_LOAD_MODE_DIRECT_IO = 4}.
     */
    DIRECT_IO("dio");

    /**
     * The CLI string passed to {@code --load-mode} in llama.cpp's {@code common/arg.cpp}.
     */
    private final String argValue;

    LoadMode(String value) {
        this.argValue = value;
    }

    /**
     * Returns the CLI string accepted by llama.cpp's {@code --load-mode} argument.
     *
     * @return the mode string ({@code "auto"}, {@code "none"}, {@code "mmap"}, {@code "mlock"},
     *     {@code "mmap+mlock"} or {@code "dio"})
     */
    @Override
    public String getArgValue() {
        return argValue;
    }
}
