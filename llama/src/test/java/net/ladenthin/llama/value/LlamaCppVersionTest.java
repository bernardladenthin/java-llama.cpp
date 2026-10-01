// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.value;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.matchesRegex;
import static org.hamcrest.Matchers.notNullValue;

import java.lang.reflect.Constructor;
import java.lang.reflect.Modifier;
import org.junit.jupiter.api.Test;

/**
 * Tests for the {@link LlamaCppVersion#LLAMA_CPP_VERSION} compile-time constant.
 *
 * <p>These are pure-Java assertions with no native dependency: they pin the shape of the pinned
 * llama.cpp tag ({@code b} followed by digits, e.g. {@code "b9959"}) rather than a literal value, so
 * a routine version bump does not break the test while a malformed edit (empty string, wrong prefix)
 * still does. The authoritative runtime value ({@code LlamaModel#getLlamaCppBuildInfo()}) is
 * exercised separately by the native-load smoke test.
 */
public class LlamaCppVersionTest {

    @Test
    public void testVersionConstantIsPresentAndWellFormed() {
        assertThat(LlamaCppVersion.LLAMA_CPP_VERSION, notNullValue());
        // The pinned tag is always "b" followed by the upstream build number.
        assertThat(LlamaCppVersion.LLAMA_CPP_VERSION, matchesRegex("^b\\d+$"));
    }

    @Test
    public void testClassIsNotInstantiable() throws Exception {
        Constructor<LlamaCppVersion> constructor = LlamaCppVersion.class.getDeclaredConstructor();
        assertThat(Modifier.isPrivate(constructor.getModifiers()), org.hamcrest.Matchers.is(true));
    }
}
