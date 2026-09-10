// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
package net.ladenthin.llama;

import static com.tngtech.archunit.core.domain.JavaCall.Predicates.target;
import static com.tngtech.archunit.core.domain.JavaClass.Predicates.assignableTo;
import static com.tngtech.archunit.core.domain.JavaClass.Predicates.resideInAPackage;
import static com.tngtech.archunit.core.domain.JavaClass.Predicates.resideInAnyPackage;
import static com.tngtech.archunit.core.domain.properties.HasName.Predicates.name;
import static com.tngtech.archunit.core.domain.properties.HasOwner.Predicates.With.owner;
import static com.tngtech.archunit.lang.syntax.ArchRuleDefinition.fields;
import static com.tngtech.archunit.lang.syntax.ArchRuleDefinition.noClasses;
import static com.tngtech.archunit.library.Architectures.layeredArchitecture;
import static com.tngtech.archunit.library.dependencies.SlicesRuleDefinition.slices;

import com.tngtech.archunit.base.DescribedPredicate;
import com.tngtech.archunit.core.importer.ImportOption;
import com.tngtech.archunit.junit.AnalyzeClasses;
import com.tngtech.archunit.junit.ArchTest;
import com.tngtech.archunit.lang.ArchRule;
import java.util.Random;
import org.slf4j.Logger;

@AnalyzeClasses(packages = "net.ladenthin.llama", importOptions = ImportOption.DoNotIncludeTests.class)
public class LlamaArchitectureTest {

    /**
     * Production code must not use java.util.logging directly; all logging goes through SLF4J.
     */
    @ArchTest
    static final ArchRule noJavaUtilLogging = noClasses()
            .that()
            .resideInAPackage("net.ladenthin.llama..")
            .should()
            .dependOnClassesThat()
            .resideInAPackage("java.util.logging..");

    /**
     * Test-framework classes must not appear in production code.
     */
    @ArchTest
    static final ArchRule noTestFrameworksInProduction = noClasses()
            .that()
            .resideInAPackage("net.ladenthin.llama..")
            .should()
            .dependOnClassesThat()
            .resideInAnyPackage("org.junit..", "net.jqwik..", "com.tngtech.archunit..");

    /**
     * Every SLF4J {@link Logger} field follows the {@code private static final} idiom.
     */
    @ArchTest
    static final ArchRule loggersArePrivateStaticFinal = fields().that()
            .haveRawType(Logger.class)
            .should()
            .bePrivate()
            .andShould()
            .beStatic()
            .andShould()
            .beFinal();

    /**
     * No package cycles between sub-packages. Catches design drift where a leaf
     * package starts importing from its parent or sibling.
     */
    @ArchTest
    static final ArchRule noPackageCycles =
            slices().matching("net.ladenthin.llama.(*)..").should().beFreeOfCycles();

    /**
     * The {@code args} sub-package is a true leaf: pure enums / constants
     * ({@code Sampler}, {@code PoolingType}, {@code ModelFlag}, …). It must not
     * import anything from elsewhere in the project. Subsumed by the
     * {@link #layeredArchitecture} rule below (args is in the Foundation layer),
     * but kept as a precise, fast-failing guard for this specific leaf.
     */
    @ArchTest
    static final ArchRule argsPackageIsALeaf = noClasses()
            .that()
            .resideInAPackage("net.ladenthin.llama.args..")
            .should()
            .dependOnClassesThat()
            .resideInAnyPackage(
                    "net.ladenthin.llama",
                    "net.ladenthin.llama.callback..",
                    "net.ladenthin.llama.exception..",
                    "net.ladenthin.llama.json..",
                    "net.ladenthin.llama.loader..",
                    "net.ladenthin.llama.parameters..",
                    "net.ladenthin.llama.value..");

    /**
     * Strict layered architecture — <b>one layer per package</b>. Each package's
     * {@code mayOnlyBeAccessedByLayers} lists the EXACT set of packages that reference it today
     * (verified against the compiled bytecode graph), so even intra-tier edges are governed: a
     * new dependency between any two packages fails the build unless this rule is updated to
     * intend it. Conceptual tiers (informational): {@code Server} &gt; {@code Api} (root) &gt;
     * {@code Loader} &gt; {@code Json}/{@code Parameters} &gt;
     * {@code Value}/{@code Callback}/{@code Exception}/{@code Args}. The {@code Server} layer is the
     * optional OpenAI-compatible HTTP / native-server entry point; it is the only layer permitted to
     * access the {@code Api} root, and it also reaches the {@code Loader} ({@code NativeServer}
     * triggers {@code LlamaLoader.initialize()} before starting the embedded native server) and the
     * {@code Args} enums ({@code OpenAiServerCli} maps {@code -ctk}/{@code -ctv} to {@code CacheType}).
     */
    @ArchTest
    static final ArchRule layeredArchitecture = layeredArchitecture()
            .consideringOnlyDependenciesInLayers()
            .layer("Api")
            .definedBy("net.ladenthin.llama")
            .layer("Loader")
            .definedBy("net.ladenthin.llama.loader..")
            .layer("Json")
            .definedBy("net.ladenthin.llama.json..")
            .layer("Parameters")
            .definedBy("net.ladenthin.llama.parameters..")
            .layer("Value")
            .definedBy("net.ladenthin.llama.value..")
            .layer("Callback")
            .definedBy("net.ladenthin.llama.callback..")
            .layer("Exception")
            .definedBy("net.ladenthin.llama.exception..")
            .layer("Args")
            .definedBy("net.ladenthin.llama.args..")
            .layer("Server")
            .definedBy("net.ladenthin.llama.server..")
            .whereLayer("Api")
            .mayOnlyBeAccessedByLayers("Server")
            .whereLayer("Loader")
            .mayOnlyBeAccessedByLayers("Api", "Server")
            // Server: RouterClient parses the router GET /models wire format via
            // json.RouterModelsResponseParser (a pure transform, tested model-free).
            .whereLayer("Json")
            .mayOnlyBeAccessedByLayers("Api", "Server")
            .whereLayer("Parameters")
            .mayOnlyBeAccessedByLayers("Api", "Loader", "Server")
            .whereLayer("Value")
            .mayOnlyBeAccessedByLayers("Api", "Json", "Parameters", "Server")
            .whereLayer("Callback")
            .mayOnlyBeAccessedByLayers("Api")
            .whereLayer("Exception")
            .mayOnlyBeAccessedByLayers("Api", "Loader")
            .whereLayer("Args")
            .mayOnlyBeAccessedByLayers("Api", "Loader", "Parameters", "Server")
            .whereLayer("Server")
            .mayNotBeAccessedByAnyLayer();

    /**
     * Production code must not import unsupported / internal JDK packages.
     * These are not part of the Java SE API and may change or disappear without notice.
     * {@code OSInfo} is vendored from xerial/sqlite-jdbc and was already audited;
     * if it ever pulls in sun.*, this rule fails and forces a re-audit.
     *
     * <p>Exception: {@code com.sun.net.httpserver} is a <em>supported</em>, documented JDK API
     * (the exported {@code jdk.httpserver} module), used by {@code net.ladenthin.llama.server} to
     * provide the OpenAI-compatible endpoint without adding a web-framework dependency. Despite the
     * {@code com.sun} prefix it is not an internal package, so it is allowed.
     */
    @ArchTest
    static final ArchRule noInternalJdkImports = noClasses()
            .that()
            .resideInAPackage("net.ladenthin.llama..")
            .should()
            .dependOnClassesThat(resideInAnyPackage("sun..", "com.sun..", "jdk.internal..")
                    .and(DescribedPredicate.not(resideInAPackage("com.sun.net.httpserver.."))));

    /**
     * Public mutable state forbidden: any non-static field declared
     * {@code public} must also be {@code final}. {@link LlamaOutput} is an
     * immutable value class with {@code public final} fields — that pattern
     * remains allowed because the fields ARE final.
     */
    @ArchTest
    static final ArchRule noPublicMutableFields =
            fields().that().arePublic().and().areNotStatic().should().beFinal();

    /**
     * Production code must not call {@link System#exit(int)}; throw an exception instead.
     */
    @ArchTest
    static final ArchRule noSystemExit = noClasses()
            .that()
            .resideInAPackage("net.ladenthin.llama..")
            .should()
            .callMethod(System.class, "exit", int.class)
            .allowEmptyShould(true);

    /**
     * Production code must not construct {@link java.util.Random}; {@code Random} is a non-cryptographic
     * PRNG (CWE-338). Use {@link java.security.SecureRandom} or {@link java.util.concurrent.ThreadLocalRandom}
     * depending on whether cryptographic strength or thread-local fast jitter is needed.
     */
    @ArchTest
    static final ArchRule noNewRandom = noClasses()
            .that()
            .resideInAPackage("net.ladenthin.llama..")
            .should()
            .callConstructor(Random.class)
            .orShould()
            .callConstructor(Random.class, long.class)
            .allowEmptyShould(true);

    /**
     * Production code must not call {@link Thread#sleep(long)} / {@link Thread#sleep(long, int)};
     * prefer {@link java.util.concurrent.BlockingQueue#poll(long, java.util.concurrent.TimeUnit)} or
     * {@link java.util.concurrent.locks.Condition#await(long, java.util.concurrent.TimeUnit)}.
     */
    @ArchTest
    static final ArchRule noThreadSleep = noClasses()
            .that()
            .resideInAPackage("net.ladenthin.llama..")
            .should()
            .callMethod(Thread.class, "sleep", long.class)
            .orShould()
            .callMethod(Thread.class, "sleep", long.class, int.class)
            .allowEmptyShould(true);

    /**
     * A parameter object's {@code toString()} must never reach a wire payload. It is the redacted
     * debug view ({@code InferenceParameters{keys=[...], values=redacted}}) and is deliberately not
     * valid JSON; {@code toJson()} is the serializer. The two used to be the same method, so every
     * call site that was correct before the split is a silent trap after it — and this one bit:
     * {@code LlamaIterator} kept calling {@code toString()} and sent the native parser an
     * unparseable body on every streaming generation, which no local run could see because every
     * test that exercises streaming is model-gated and self-skips without a GGUF.
     *
     * <p>The {@code parameters} package itself is scoped out, and not as a convenience: because
     * {@code JsonParameters} is package-private and its subclasses are public, javac emits a
     * synthetic bridge {@code toString()} in each subclass that does nothing but
     * {@code invokespecial} the supertype's. That call exists in the bytecode and in no source file,
     * so a rule covering the package would fail on a method nobody can edit.
     *
     * <p>Limitation worth knowing: this catches an explicit {@code toString()} call, not an implicit
     * one through string concatenation, which the compiler lowers to {@code StringBuilder.append} or
     * an {@code invokedynamic} string-concat factory and leaves no {@code toString()} call site to
     * match. Concatenating a parameter object into a request body would still slip through — but
     * that shape does not occur here, and the redacted form is unparseable precisely so that it
     * fails loudly at the parser rather than sending a different body.
     */
    @ArchTest
    static final ArchRule parameterToStringIsNeverAWirePayload = noClasses()
            .that()
            .resideInAPackage("net.ladenthin.llama..")
            .and()
            .resideOutsideOfPackage("net.ladenthin.llama.parameters..")
            .should()
            .callMethodWhere(target(name("toString"))
                    .and(target(owner(assignableTo("net.ladenthin.llama.parameters.JsonParameters")))))
            .because("toString() is the redacted debug view; the wire serializer is toJson()")
            .allowEmptyShould(true);

    /**
     * Per-module banned import: the foundation contracts ({@code args}, {@code callback},
     * {@code exception}) and the {@code loader} infrastructure must stay free of the Jackson
     * JSON library ({@code com.fasterxml.jackson..}). JSON marshalling is the job of
     * {@code value} / {@code json} / {@code parameters} (and the root {@code Api}, which drives
     * them); these layers carry only plain typed data and native-loading logic.
     */
    @ArchTest
    static final ArchRule jacksonBannedFromContractsAndLoader = noClasses()
            .that()
            .resideInAnyPackage(
                    "net.ladenthin.llama.args..",
                    "net.ladenthin.llama.callback..",
                    "net.ladenthin.llama.exception..",
                    "net.ladenthin.llama.loader..")
            .should()
            .dependOnClassesThat()
            .resideInAPackage("com.fasterxml.jackson..")
            .allowEmptyShould(true);
}
