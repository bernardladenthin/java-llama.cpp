// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.parameters;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import net.ladenthin.llama.ClaudeGenerated;
import net.ladenthin.llama.args.CacheType;
import net.ladenthin.llama.args.CliArg;
import net.ladenthin.llama.args.ModelOption;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose = "Verify the withScalar / withOptionalJson / withRaw helpers on the "
                + "immutable JsonParameters base: that they store the expected string form for every "
                + "primitive type used by InferenceParameters (int, long, float, double, boolean), "
                + "that every stored value is exactly one well-formed JSON value, that every helper "
                + "returns a NEW instance whose parameter map carries the entry inserted or replaced "
                + "without touching the original, and that the inherited parameters map is an "
                + "unmodifiable view. The CliParameters subclass tests cover the legacy put-style "
                + "helpers used by ModelParameters (which still extends CliParameters and remains "
                + "mutable).")
public class JsonParametersTest {

    private static final class TestBuilder extends JsonParameters {
        TestBuilder() {
            super();
        }

        TestBuilder(Map<String, String> parameters) {
            super(parameters);
        }

        @Override
        @SuppressWarnings("unchecked")
        protected <T extends JsonParameters> T withParameters(Map<String, String> newParameters) {
            return (T) new TestBuilder(newParameters);
        }

        TestBuilder withScalarPublic(RequestField field, Object value) {
            return withScalar(field, value);
        }

        TestBuilder withRawPublic(RequestField field, String value) {
            return withRaw(field, value);
        }

        TestBuilder withOptionalJsonPublic(RequestField field, String text) {
            return withOptionalJson(field, text);
        }
    }

    @Test
    public void withScalar_int_storesDecimalString() {
        TestBuilder b = new TestBuilder().withScalarPublic(RequestField.N_KEEP, 8);
        assertEquals("8", b.parameters.get("n_keep"));
    }

    @Test
    public void withScalar_negativeInt_storesSignedDecimal() {
        TestBuilder b = new TestBuilder().withScalarPublic(RequestField.N_PREDICT, -1);
        assertEquals("-1", b.parameters.get("n_predict"));
    }

    @Test
    public void withScalar_zero_storesZero() {
        TestBuilder b = new TestBuilder().withScalarPublic(RequestField.N_KEEP, 0);
        assertEquals("0", b.parameters.get("n_keep"));
    }

    @Test
    public void withScalar_long_storesDecimalString() {
        TestBuilder b = new TestBuilder().withScalarPublic(RequestField.SEED, 4242424242L);
        assertEquals("4242424242", b.parameters.get("seed"));
    }

    @Test
    public void withScalar_float_storesDotSeparatedDecimal() {
        TestBuilder b = new TestBuilder().withScalarPublic(RequestField.TEMPERATURE, 0.7f);
        // String.valueOf(float) is locale-independent and uses '.' as the decimal separator.
        assertEquals("0.7", b.parameters.get("temperature"));
    }

    @Test
    public void withScalar_double_storesDotSeparatedDecimal() {
        TestBuilder b = new TestBuilder().withScalarPublic(RequestField.TOP_P, 0.95d);
        assertEquals("0.95", b.parameters.get("top_p"));
    }

    @Test
    public void withScalar_booleanTrue_storesLowercaseTrue() {
        TestBuilder b = new TestBuilder().withScalarPublic(RequestField.CACHE_PROMPT, true);
        assertEquals("true", b.parameters.get("cache_prompt"));
    }

    @Test
    public void withScalar_booleanFalse_storesLowercaseFalse() {
        TestBuilder b = new TestBuilder().withScalarPublic(RequestField.CACHE_PROMPT, false);
        assertEquals("false", b.parameters.get("cache_prompt"));
    }

    @Test
    public void withScalar_overwritesPreviousValue() {
        TestBuilder b =
                new TestBuilder().withScalarPublic(RequestField.N_KEEP, 4).withScalarPublic(RequestField.N_KEEP, 16);
        assertEquals("16", b.parameters.get("n_keep"));
        assertEquals(1, b.parameters.size());
    }

    @Test
    public void withScalar_returnsFreshInstance() {
        TestBuilder original = new TestBuilder();
        TestBuilder derived = original.withScalarPublic(RequestField.N_KEEP, 1);
        assertNotSame(original, derived, "wither must allocate a new instance");
        assertTrue(original.parameters.isEmpty(), "original must remain empty");
        assertEquals("1", derived.parameters.get("n_keep"));
    }

    @Test
    public void withRaw_storesValueVerbatim() {
        TestBuilder b = new TestBuilder().withRawPublic(RequestField.JSON_SCHEMA, "{\"type\":\"object\"}");
        assertEquals("{\"type\":\"object\"}", b.parameters.get("json_schema"));
    }

    @Test
    public void withOptionalJson_nullIsNoOpReturnsSameInstance() {
        TestBuilder original = new TestBuilder();
        TestBuilder derived = original.withOptionalJsonPublic(RequestField.GRAMMAR, null);
        assertSame(original, derived, "null input must short-circuit to this");
    }

    @Test
    public void withOptionalJson_nonNullEncodesAndAllocates() {
        TestBuilder original = new TestBuilder();
        TestBuilder derived = original.withOptionalJsonPublic(RequestField.GRAMMAR, "abc");
        assertNotSame(original, derived);
        assertEquals("\"abc\"", derived.parameters.get("grammar"), "value must be JSON-encoded");
    }

    @Test
    public void parametersAccessorIsUnmodifiable() {
        TestBuilder b = new TestBuilder().withScalarPublic(RequestField.N_KEEP, 1);
        assertThrows(UnsupportedOperationException.class, () -> b.parameters.put("evil", "x"));
    }

    // The CliParameters base class still carries the legacy putScalar / putEnum helpers
    // because ModelParameters does not extend JsonParameters. The CliParameters subclass
    // remains mutable by design.

    private static final class CliTestBuilder extends CliParameters {
        CliTestBuilder putScalarPublic(ModelOption option, Object value) {
            return putScalar(option, value);
        }

        CliTestBuilder putEnumPublic(ModelOption option, CliArg value) {
            return putEnum(option, value);
        }
    }

    @Test
    public void cliPutScalar_int_storesDecimalString() {
        CliTestBuilder b = new CliTestBuilder();
        b.putScalarPublic(ModelOption.THREADS, 8);
        assertEquals("8", b.parameters.get("--threads"));
    }

    @Test
    public void cliPutScalar_returnsSameBuilderInstance() {
        CliTestBuilder b = new CliTestBuilder();
        CliTestBuilder returned = b.putScalarPublic(ModelOption.THREADS, 1);
        assertSame(returned, b);
    }

    @Test
    public void cliPutEnum_usesGetArgValueNotEnumName() {
        CliTestBuilder b = new CliTestBuilder();
        b.putEnumPublic(ModelOption.CACHE_TYPE_K, CacheType.Q8_0);
        assertEquals("q8_0", b.parameters.get("--cache-type-k"));
    }

    @Test
    public void cliPutEnum_returnsSameBuilderInstance() {
        CliTestBuilder b = new CliTestBuilder();
        CliTestBuilder returned = b.putEnumPublic(ModelOption.CACHE_TYPE_K, CacheType.F16);
        assertSame(returned, b);
    }
    // -------------------------------------------------------------------------
    // The one-JSON-value invariant on the base class
    // -------------------------------------------------------------------------

    /**
     * withPut is the single choke point: every stored value must be exactly one well-formed JSON
     * value, whichever helper wrote it. That is the invariant the wire renderer relies on, and the
     * reason a raw fragment cannot inject sibling fields.
     */
    @Test
    public void withRaw_rejectsAValueFollowedByMoreText() {
        assertThrows(
                IllegalArgumentException.class,
                () -> new TestBuilder().withRawPublic(RequestField.JSON_SCHEMA, "1, \"x\": 2"));
    }

    @Test
    public void withRaw_rejectsMalformedJson() {
        assertThrows(
                IllegalArgumentException.class, () -> new TestBuilder().withRawPublic(RequestField.JSON_SCHEMA, "{"));
    }

    /**
     * The removed withEnum helper stored getArgValue() unquoted (e.g. {@code q8_0}), which is not a
     * JSON value at all -- it had no production caller on the JSON side, and the invariant is what
     * makes that unrepresentable rather than merely unused. The CLI side keeps its own putEnum,
     * where a bare string is exactly right because argv values are not JSON.
     */
    @Test
    public void withRaw_rejectsABareEnumArgValue() {
        assertThrows(
                IllegalArgumentException.class,
                () -> new TestBuilder().withRawPublic(RequestField.JSON_SCHEMA, "q8_0"));
    }

    /**
     * The rejection message carries a bounded excerpt so a large or hostile fragment cannot flood a
     * log through it. These two pin the boundary itself: exactly at the limit the value is shown
     * whole, one character past it the excerpt is elided.
     */
    @Test
    public void rejectionMessageShowsAValueAtTheExcerptLimitInFull() {
        StringBuilder atLimit = new StringBuilder("{");
        while (atLimit.length() < 80) {
            atLimit.append('a');
        }
        IllegalArgumentException e = assertThrows(
                IllegalArgumentException.class,
                () -> new TestBuilder().withRawPublic(RequestField.JSON_SCHEMA, atLimit.toString()));
        assertTrue(e.getMessage().endsWith(atLimit.toString()), e.getMessage());
    }

    @Test
    public void rejectionMessageElidesAValuePastTheExcerptLimit() {
        StringBuilder pastLimit = new StringBuilder("{");
        while (pastLimit.length() < 81) {
            pastLimit.append('a');
        }
        IllegalArgumentException e = assertThrows(
                IllegalArgumentException.class,
                () -> new TestBuilder().withRawPublic(RequestField.JSON_SCHEMA, pastLimit.toString()));
        assertTrue(e.getMessage().endsWith("..."), e.getMessage());
    }

    @Test
    public void toJson_rendersACompactObject() {
        TestBuilder b = new TestBuilder()
                .withScalarPublic(RequestField.TOP_K, 1)
                .withOptionalJsonPublic(RequestField.PROMPT, "x");
        assertEquals("{\"prompt\":\"x\",\"top_k\":1}", b.toJson());
    }
}
