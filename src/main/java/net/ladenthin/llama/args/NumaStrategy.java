// SPDX-FileCopyrightText: 2023-2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2026 Konstantin Heurer
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

/**
 * NUMA optimization strategy for {@code --numa}.
 */
public enum NumaStrategy implements CliArg {

    DISTRIBUTE("distribute"),
    ISOLATE("isolate"),
    NUMACTL("numactl");

    private final String argValue;

    NumaStrategy(String argValue) {
        this.argValue = argValue;
    }

    @Override
    public String getArgValue() {
        return argValue;
    }
}
