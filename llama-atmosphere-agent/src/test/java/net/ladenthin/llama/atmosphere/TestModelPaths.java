// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.jspecify.annotations.Nullable;

/**
 * Resolves model paths the way the core's {@code TestConstants.resolveModelPath} does: relative to
 * this project first, then to its parent (the reactor root, where CI restores the shared GGUF cache).
 * Test classes are not shared between modules, so the resolver is carried here as well.
 */
final class TestModelPaths {

    private TestModelPaths() {}

    /**
     * Resolves a configured fixture path against the working directory and then its parent.
     *
     * @param path the configured path, may be {@code null} or empty
     * @return an existing path, or {@code null} when {@code path} is null/empty, or the unresolved
     *     path itself when it exists in neither location (so skip messages name what was looked for)
     */
    static @Nullable Path resolve(@Nullable String path) {
        if (path == null || path.isEmpty()) {
            return null;
        }
        Path candidate = Paths.get(path);
        if (candidate.isAbsolute() || Files.exists(candidate)) {
            return candidate;
        }
        Path fromParent = Paths.get("..").resolve(candidate);
        if (Files.exists(fromParent)) {
            return fromParent.toAbsolutePath().normalize();
        }
        return candidate;
    }

    /**
     * Resolves the path held by a system property.
     *
     * @param property the system-property name
     * @return the resolved path, or {@code null} when the property is unset or empty
     */
    static @Nullable Path fromProperty(String property) {
        return resolve(System.getProperty(property));
    }
}
