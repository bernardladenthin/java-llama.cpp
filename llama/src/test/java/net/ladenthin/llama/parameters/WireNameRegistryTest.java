// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.parameters;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;
import static org.hamcrest.Matchers.notNullValue;
import static org.hamcrest.Matchers.startsWith;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import net.ladenthin.llama.ClaudeGenerated;
import net.ladenthin.llama.args.CliArg;
import net.ladenthin.llama.args.ModelFlag;
import net.ladenthin.llama.args.ModelOption;
import org.junit.jupiter.api.Test;

/**
 * Properties of the wire-name registries that only became checkable once the names moved onto
 * enum constants.
 *
 * <p>While a name was a string literal buried among ~200 builder setters, none of this was
 * expressible: two setters could write the same key with the last call silently winning, a name
 * could be declared and never emitted by anything, and the only way to enumerate the set at all was
 * to scrape the Java source as text. The registries make each of those a one-line assertion over
 * {@code values()}.
 */
@ClaudeGenerated(
        purpose = "Pin the registry-level properties of ModelFlag / ModelOption / RequestField: "
                + "well-formed and unique wire names across both CLI registries, a declared contract "
                + "on every constant, and — by driving every public builder setter reflectively — "
                + "that each declared constant is actually reachable from the public API.")
public class WireNameRegistryTest {

    // -------------------------------------------------------------------------
    // Well-formedness and uniqueness
    // -------------------------------------------------------------------------

    @Test
    public void everyCliNameIsAWellFormedLongOption() {
        for (ModelFlag flag : ModelFlag.values()) {
            assertThat(flag.name(), flag.getCliFlag(), startsWith("--"));
            assertThat(flag.name(), flag.getContract(), is(notNullValue()));
        }
        for (ModelOption option : ModelOption.values()) {
            assertThat(option.name(), option.getCliOption(), startsWith("--"));
            assertThat(option.name(), option.getContract(), is(notNullValue()));
        }
    }

    /**
     * The two CLI registries share one namespace — they are merged into a single argv — so a name
     * may appear in exactly one of them. A duplicate would mean two builder methods writing the same
     * argv key, where the last call silently wins.
     */
    @Test
    public void cliNamesAreUniqueAcrossBothRegistries() {
        Map<String, String> seen = new HashMap<>();
        List<String> duplicates = new ArrayList<>();
        for (ModelFlag flag : ModelFlag.values()) {
            String previous = seen.put(flag.getCliFlag(), "ModelFlag." + flag.name());
            if (previous != null) {
                duplicates.add(flag.getCliFlag() + " (" + previous + " / ModelFlag." + flag.name() + ")");
            }
        }
        for (ModelOption option : ModelOption.values()) {
            String previous = seen.put(option.getCliOption(), "ModelOption." + option.name());
            if (previous != null) {
                duplicates.add(option.getCliOption() + " (" + previous + " / ModelOption." + option.name() + ")");
            }
        }
        assertThat(duplicates, is(empty()));
    }

    @Test
    public void requestKeysAreWellFormedAndUnique() {
        Set<String> seen = new HashSet<>();
        List<String> duplicates = new ArrayList<>();
        for (RequestField field : RequestField.values()) {
            assertThat(field.name(), field.getKey().trim(), is(field.getKey()));
            assertThat(field.name(), field.getKey().isEmpty(), is(false));
            assertThat(field.name(), field.getContract(), is(notNullValue()));
            if (!seen.add(field.getKey())) {
                duplicates.add(field.getKey());
            }
        }
        assertThat(duplicates, is(empty()));
    }

    // -------------------------------------------------------------------------
    // Reachability: a declared name that nothing emits is dead weight
    // -------------------------------------------------------------------------

    /**
     * Drives every public {@code ModelParameters} setter with a plausible argument and collects the
     * argv keys they write, then asserts every declared constant showed up.
     *
     * <p>This catches the inverse of the C++ contract test. That one asks "is every name the Java
     * layer can emit still accepted by llama.cpp"; this one asks "is every name the Java layer
     * declares actually emittable at all". A constant no builder writes is invisible to the C++
     * check too — it would be fed to the parser forever without any caller being able to reach it.
     */
    @Test
    public void everyModelOptionAndFlagIsReachableFromAPublicSetter() {
        Set<String> emitted = drivePublicSetters();
        assertThat("no setter produced any key — the driver itself is broken", emitted.size(), greaterThan(50));

        Set<String> unreachable = new TreeSet<>();
        for (ModelOption option : ModelOption.values()) {
            if (!emitted.contains(option.getCliOption())) {
                unreachable.add("ModelOption." + option.name() + " (" + option.getCliOption() + ")");
            }
        }
        for (ModelFlag flag : ModelFlag.values()) {
            if (!emitted.contains(flag.getCliFlag())) {
                unreachable.add("ModelFlag." + flag.name() + " (" + flag.getCliFlag() + ")");
            }
        }
        assertThat(unreachable, is(empty()));
    }

    /**
     * Same reachability question for the request side, driven through the immutable withers.
     *
     * <p>A String argument is tried against several candidate shapes because the raw-JSON entry
     * points reject anything that is not exactly one well-formed JSON value — a single sample would
     * make those fields look unreachable when they are merely picky.
     */
    @Test
    public void everyRequestFieldIsReachableFromAPublicWither() {
        Set<String> emitted =
                new HashSet<>(InferenceParameters.empty().parameters.keySet());
        for (Method method : InferenceParameters.class.getMethods()) {
            if (!InferenceParameters.class.equals(method.getDeclaringClass())
                    || !InferenceParameters.class.equals(method.getReturnType())
                    || method.getParameterCount() == 0) {
                continue;
            }
            for (int shape = 0; shape < SAMPLE_SHAPES; shape++) {
                Object[] args = sampleArguments(method, shape);
                if (args == null) {
                    break;
                }
                try {
                    Object result = method.invoke(InferenceParameters.empty(), args);
                    emitted.addAll(((InferenceParameters) result).parameters.keySet());
                } catch (ReflectiveOperationException | RuntimeException e) {
                    // This shape was rejected or ignored; the next one may be accepted.
                }
            }
        }
        assertThat("no wither produced any key — the driver itself is broken", emitted.size(), greaterThan(20));

        Set<String> unreachable = new TreeSet<>();
        for (RequestField field : RequestField.values()) {
            if (!emitted.contains(field.getKey())) {
                unreachable.add("RequestField." + field.name() + " (" + field.getKey() + ")");
            }
        }
        assertThat(unreachable, is(empty()));
    }

    // -------------------------------------------------------------------------
    // The SpotBugs suppression list is derived, not hand-maintained
    // -------------------------------------------------------------------------

    /**
     * Every method named in the {@code OCP_OVERLY_CONCRETE_PARAMETER} suppressions must still exist
     * as an enum-valued setter.
     *
     * <p>Only this direction is checkable, and it is the one nothing else covers. The opposite
     * direction — a flagged setter missing from the list — already reds {@code spotbugs:check}, and
     * is not derivable here anyway: SpotBugs raises OCP only when a method uses nothing beyond the
     * interface, so {@code setPoolingType} (which compares against a concrete constant) and
     * {@code withMiroStat} (which calls {@code ordinal()}) are legitimately absent. A <em>stale</em>
     * entry is the silent half: a suppression naming a method that no longer exists simply does
     * nothing, and the next real finding on the renamed method is a surprise. That is exactly how
     * the {@code setTensorReadLazy} to {@code setLazyMode} rename reddened {@code main}.
     */
    @Test
    public void everyOcpSuppressionStillNamesAnEnumValuedSetter() {
        Set<String> declared = ocpSuppressedMethodNames();
        assertThat("the OCP suppression block was not found", declared, is(not(empty())));

        Set<String> enumValuedSetters = new TreeSet<>();
        for (Class<?> builder : new Class<?>[] {ModelParameters.class, InferenceParameters.class}) {
            for (Method method : builder.getMethods()) {
                if (!builder.equals(method.getDeclaringClass()) || method.getParameterCount() != 1) {
                    continue;
                }
                Class<?> type = method.getParameterTypes()[0];
                if (type.isEnum() && CliArg.class.isAssignableFrom(type)) {
                    enumValuedSetters.add(method.getName());
                }
            }
        }

        Set<String> stale = new TreeSet<>(declared);
        stale.removeAll(enumValuedSetters);
        assertThat("suppression entries with no matching enum-valued setter", stale, is(empty()));
    }

    /** Read the method names out of the ModelParameters OCP_OVERLY_CONCRETE_PARAMETER Match block. */
    private static Set<String> ocpSuppressedMethodNames() {
        String xml = readSpotbugsExclude();
        int bug = xml.indexOf("OCP_OVERLY_CONCRETE_PARAMETER");
        Set<String> names = new TreeSet<>();
        while (bug >= 0) {
            int end = xml.indexOf("</Match>", bug);
            if (end < 0) {
                break;
            }
            java.util.regex.Matcher m = java.util.regex.Pattern.compile("<Method name=\"([A-Za-z0-9_]+)\"")
                    .matcher(xml.substring(bug, end));
            while (m.find()) {
                names.add(m.group(1));
            }
            bug = xml.indexOf("OCP_OVERLY_CONCRETE_PARAMETER", end);
        }
        return names;
    }

    /**
     * Surefire's working directory is the module basedir, but a developer may run from the reactor
     * root; accept either layout rather than depending on which one the runner chose.
     */
    private static String readSpotbugsExclude() {
        for (String candidate : new String[] {"spotbugs-exclude.xml", "llama/spotbugs-exclude.xml"}) {
            java.io.File file = new java.io.File(candidate);
            if (file.isFile()) {
                try {
                    byte[] bytes = java.nio.file.Files.readAllBytes(file.toPath());
                    return new String(bytes, java.nio.charset.StandardCharsets.UTF_8);
                } catch (java.io.IOException e) {
                    throw new IllegalStateException("cannot read " + candidate, e);
                }
            }
        }
        throw new IllegalStateException(
                "spotbugs-exclude.xml not found from " + new java.io.File(".").getAbsolutePath());
    }

    /**
     * Invoke every public instance setter of {@code ModelParameters} that returns the builder, on a
     * fresh instance each time, and union the keys each one wrote. A setter that rejects the sample
     * argument is skipped rather than failing the run — it is the union that matters, and every
     * option is written by at least one setter that accepts a plain value.
     */
    private static Set<String> drivePublicSetters() {
        // The constructor seeds defaults of its own (--fit today). Those are emitted by
        // construction, so they count as reachable rather than as noise to subtract.
        Set<String> emitted = new HashSet<>(new ModelParameters().parameters.keySet());
        for (Method method : ModelParameters.class.getMethods()) {
            if (!ModelParameters.class.equals(method.getDeclaringClass())
                    || !ModelParameters.class.equals(method.getReturnType())) {
                continue;
            }
            for (int shape = 0; shape < SAMPLE_SHAPES; shape++) {
                Object[] args = sampleArguments(method, shape);
                if (args == null) {
                    break;
                }
                try {
                    ModelParameters target = new ModelParameters();
                    method.invoke(target, args);
                    emitted.addAll(target.parameters.keySet());
                } catch (ReflectiveOperationException | RuntimeException e) {
                    // This shape was rejected; the next one may be accepted.
                }
            }
        }
        // setFlag/clearFlag take the registry itself, so drive them explicitly.
        for (ModelFlag flag : ModelFlag.values()) {
            emitted.addAll(new ModelParameters().setFlag(flag).parameters.keySet());
        }
        return emitted;
    }

    /**
     * Sample argument shapes, tried in order. A setter may reject a value (a range guard) or ignore
     * it (an empty container is documented as a no-op on several withers), so one shape is not
     * enough to prove reachability — the driver retries until a call writes something.
     */
    private static final int SAMPLE_SHAPES = 4;

    /** Build a plausible argument list for one shape, or {@code null} when a type is not handled. */
    private static Object @org.jspecify.annotations.Nullable [] sampleArguments(Method method, int shape) {
        Class<?>[] types = method.getParameterTypes();
        Object[] args = new Object[types.length];
        for (int i = 0; i < types.length; i++) {
            Class<?> type = types[i];
            if (type == int.class) {
                args[i] = 1;
            } else if (type == long.class) {
                args[i] = 1L;
            } else if (type == float.class) {
                args[i] = 0.5f;
            } else if (type == double.class) {
                args[i] = 0.5d;
            } else if (type == boolean.class) {
                args[i] = true;
            } else if (type == String.class) {
                args[i] = sampleString(shape);
            } else if (type.isEnum() && CliArg.class.isAssignableFrom(type)) {
                Object[] values = type.getEnumConstants();
                args[i] = values[values.length - 1];
            } else if (java.util.Map.class.isAssignableFrom(type)) {
                args[i] = sampleMap(shape);
            } else if (java.util.Collection.class.isAssignableFrom(type)) {
                args[i] = sampleCollection(shape);
            } else if (type.isArray() && type.getComponentType().isEnum()) {
                // Varargs setters such as setSamplers(Sampler...) arrive as an array type.
                Object[] values = type.getComponentType().getEnumConstants();
                Object array = java.lang.reflect.Array.newInstance(type.getComponentType(), 1);
                java.lang.reflect.Array.set(array, 0, values[values.length - 1]);
                args[i] = array;
            } else if (type.isArray() && type.getComponentType() == String.class) {
                args[i] = new String[] {"x"};
            } else if (type.isArray() && type.getComponentType() == int.class) {
                args[i] = new int[] {1};
            } else {
                return null;
            }
        }
        return args;
    }

    private static String sampleString(int shape) {
        switch (shape) {
            case 1:
                return "{}";
            case 2:
                return "[]";
            case 3:
                return "{\"type\":\"object\"}";
            default:
                return "x";
        }
    }

    private static Map<?, ?> sampleMap(int shape) {
        Map<Object, Object> map = new HashMap<>();
        switch (shape) {
            case 1:
                map.put(1, 1.0f);
                break;
            case 2:
                map.put("a", 1.0f);
                break;
            case 3:
                map.put("a", "b");
                break;
            default:
                break;
        }
        return map;
    }

    private static java.util.Collection<?> sampleCollection(int shape) {
        switch (shape) {
            case 1:
                return java.util.Collections.singletonList(1);
            case 2:
                return java.util.Collections.singletonList("a");
            case 3:
                return java.util.Collections.singletonList(1);
            default:
                return java.util.Collections.emptyList();
        }
    }
}
