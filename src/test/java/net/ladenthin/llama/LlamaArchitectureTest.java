// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
package net.ladenthin.llama;

import static com.tngtech.archunit.lang.syntax.ArchRuleDefinition.noClasses;

import com.tngtech.archunit.core.importer.ImportOption;
import com.tngtech.archunit.junit.AnalyzeClasses;
import com.tngtech.archunit.junit.ArchTest;
import com.tngtech.archunit.lang.ArchRule;

@AnalyzeClasses(packages = "net.ladenthin.llama", importOptions = ImportOption.DoNotIncludeTests.class)
public class LlamaArchitectureTest {

    /**
     * Production code must not use java.util.logging directly; all logging goes through SLF4J.
     */
    @ArchTest
    static final ArchRule noJavaUtilLogging = noClasses()
            .that().resideInAPackage("net.ladenthin.llama..")
            .should().dependOnClassesThat()
            .resideInAPackage("java.util.logging..");

    /**
     * Test-framework classes must not appear in production code.
     */
    @ArchTest
    static final ArchRule noTestFrameworksInProduction = noClasses()
            .that().resideInAPackage("net.ladenthin.llama..")
            .should().dependOnClassesThat()
            .resideInAnyPackage("org.junit..", "net.jqwik..", "com.tngtech.archunit..");
}
