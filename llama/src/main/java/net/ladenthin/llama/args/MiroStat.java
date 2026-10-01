// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

/**
 * Mirostat sampling mode for {@code --mirostat}.
 *
 * <p>The arg values ({@code "0"}, {@code "1"}, {@code "2"}) are the integer strings
 * accepted by the CLI flag, matching llama.cpp's {@code MIROSTAT_*} constants.
 */
public enum MiroStat implements CliArg {

    /** Mirostat sampling disabled. */
    DISABLED("0"),
    /** Mirostat v1 sampling. */
    V1("1"),
    /** Mirostat v2 sampling. */
    V2("2");

    private final String argValue;

    MiroStat(String argValue) {
        this.argValue = argValue;
    }

    @Override
    public String getArgValue() {
        return argValue;
    }
}
