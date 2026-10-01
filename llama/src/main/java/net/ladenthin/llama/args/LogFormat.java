// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

/**
 * The log output format (defaults to JSON for all server-based outputs).
 */
public enum LogFormat {

    /** Structured JSON log records (one JSON object per line). */
    JSON,
    /** Human-readable plain-text log lines. */
    TEXT
}
