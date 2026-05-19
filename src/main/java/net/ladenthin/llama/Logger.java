// SPDX-FileCopyrightText: 2024 Charles Oliver Nutter <headius@headius.com>
// SPDX-FileCopyrightText: 2024 Gauthier Roebroeck <gauthier.roebroeck@gmail.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// Vendored 1:1 from xerial/sqlite-jdbc:
//   src/main/java/org/sqlite/util/Logger.java
// Only deviation from the upstream file:
//   1. package org.sqlite.util  ->  package net.ladenthin.llama

package net.ladenthin.llama;

import java.util.function.Supplier;

/** A simple internal Logger interface. */
public interface Logger {
    void trace(Supplier<String> message);

    void info(Supplier<String> message);

    void warn(Supplier<String> message);

    void error(Supplier<String> message, Throwable t);
}
