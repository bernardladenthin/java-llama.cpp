// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

/**
 * NUMA optimization strategy for {@code --numa}.
 */
public enum NumaStrategy implements CliArg {

    /** Distribute execution evenly across all NUMA nodes. */
    DISTRIBUTE("distribute"),
    /** Pin execution to a single NUMA node. */
    ISOLATE("isolate"),
    /** Defer NUMA placement to {@code numactl}. */
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
