// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.langchain4j;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Resolves the GGUF paths the model-backed integration tests are pointed at.
 *
 * <p>Surefire's working directory defaults to the module basedir ({@code <repo>/llama-langchain4j}),
 * while CI restores the shared GGUF cache to {@code <repo>/models} and passes the paths as bare
 * {@code models/<name>}. Resolving those against the module directory finds nothing, every test
 * self-skips on its {@code Assumptions.assumeTrue(exists)}, and the job still reports success. This
 * helper accepts either layout &mdash; module-relative first, then reactor-root &mdash; so the tests
 * actually run in CI without the workflow having to know where Surefire stands.
 *
 * <p>The core module carries the same resolver as {@code TestConstants.resolveModelPath}; it is
 * duplicated rather than shared because test classes are not published between modules.
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
    static Path resolve(String path) {
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
    static Path fromProperty(String property) {
        return resolve(System.getProperty(property));
    }
}
