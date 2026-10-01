// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.value;

/**
 * This enum represents the native log levels of llama.cpp.
 */
public enum LogLevel {

    /** Verbose debug output. */
    DEBUG,
    /** Informational messages. */
    INFO,
    /** Recoverable problems. */
    WARN,
    /** Errors that prevent normal operation. */
    ERROR
}
