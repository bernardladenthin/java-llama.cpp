// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

/**
 * Flash Attention mode for {@code --flash-attn}.
 *
 * <p>llama.cpp turned {@code --flash-attn} from a bare flag into a value-taking option in b10273:
 * the value is mandatory, and emitting the key alone makes the parser consume whatever argv token
 * follows it. That is why this is an enum rather than a boolean — see
 * {@link net.ladenthin.llama.parameters.ModelParameters#setFlashAttn(FlashAttn)}.</p>
 */
public enum FlashAttn implements CliArg {

    /** Force Flash Attention on; the model load fails if the backend cannot provide it. */
    ON("on"),
    /** Force Flash Attention off. */
    OFF("off"),
    /** Let llama.cpp decide per backend and model — upstream's own default. */
    AUTO("auto");

    private final String argValue;

    FlashAttn(String argValue) {
        this.argValue = argValue;
    }

    @Override
    public String getArgValue() {
        return argValue;
    }
}
