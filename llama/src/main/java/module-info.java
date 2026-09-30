// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

/**
 * JPMS module descriptor for the java-llama.cpp JNI bindings.
 *
 * <p>Exports the public packages. The native libraries are not in this module: they ship as
 * separate natives jars, one directory {@code net/ladenthin/llama/<os>/<arch>/<backend>/} each,
 * which on the module path are automatic modules (each jar declares its own
 * {@code Automatic-Module-Name}). {@link net.ladenthin.llama.loader.LlamaLoader} finds them
 * through its {@link ClassLoader}, which sees the resources of every module and of the
 * classpath, so nothing needs to be {@code opens}'d. Nothing {@code requires} a natives module,
 * so a module-path launch resolves them only with {@code --add-modules} (or with the natives
 * jars on the classpath).</p>
 *
 * <p>JSpecify {@code @NullMarked} is declared at the module level here so that no source
 * file compiled at {@code --release 8} references the JSpecify annotation type directly.
 * Otherwise javac would emit an unsuppressible {@code unknown enum constant
 * ElementType.MODULE} classfile-read warning for each source compiled at release 8 that
 * resolves {@code @NullMarked} ({@code @NullMarked} carries
 * {@code @Target({MODULE, PACKAGE, TYPE})} and Java 8 does not know about
 * {@code ElementType.MODULE}). Confining the reference to {@code module-info.java} —
 * which compiles at {@code --release 9} — keeps that warning out of the build entirely.</p>
 *
 * <p>{@code requires static org.jspecify} is needed only at compile time of this
 * descriptor; JSpecify annotations carry {@code RetentionPolicy.CLASS} so module-path
 * consumers never need jspecify on their runtime path. Checker Framework qualifiers and
 * the Codehaus animal-sniffer annotation are likewise compile-time only. Jackson (tree model
 * only; a caller's own type given to {@code completeAsJson} must be open to Jackson) and SLF4J
 * are runtime dependencies and therefore required.</p>
 *
 * <p>This descriptor compiles at {@code --release 9}; the rest of the source compiles
 * at {@code --release 8}. Java 8 runtimes silently ignore {@code module-info.class} at
 * the JAR root.</p>
 */
@org.jspecify.annotations.NullMarked
module net.ladenthin.llama {
    requires static org.jspecify;

    // Lombok is `provided` scope: only used at compile time to generate equals/hashCode/toString.
    // `requires static` means the runtime does not need the lombok jar on the module path —
    // the @lombok.Generated annotation carried on generated members has CLASS retention.
    requires static lombok;

    // The OpenAI-compatible endpoint (net.ladenthin.llama.server) uses the JDK's built-in
    // com.sun.net.httpserver, so module-path consumers need to read jdk.httpserver. It is a
    // platform module (always present in the JDK), not an external dependency.
    requires jdk.httpserver;
    // Runtime dependencies of the classes (JSON tree model, logging facade).
    requires com.fasterxml.jackson.databind;
    requires org.slf4j;

    exports net.ladenthin.llama;
    exports net.ladenthin.llama.args;
    exports net.ladenthin.llama.callback;
    exports net.ladenthin.llama.exception;
    exports net.ladenthin.llama.json;
    exports net.ladenthin.llama.parameters;
    exports net.ladenthin.llama.server;
    exports net.ladenthin.llama.value;
// net.ladenthin.llama.loader is intentionally NOT exported: native-library loading,
// OS detection and process/system-property infrastructure are internal to the module.
}
