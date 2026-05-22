// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import net.ladenthin.llama.args.CacheType;
import net.ladenthin.llama.args.CliArg;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;

@ClaudeGenerated(
        purpose = "Verify the putScalar and putEnum helpers on JsonParameters: that they store the " +
                  "expected string form for every primitive type used by the ModelParameters / " +
                  "InferenceParameters setters (int, long, float, double, boolean), that they " +
                  "overwrite a previously-set key, that putEnum uses getArgValue() rather than the " +
                  "enum name, and that both helpers return the concrete builder subtype so callers " +
                  "can chain in a single statement."
)
public class JsonParametersTest {

    private static final class TestBuilder extends JsonParameters {
        TestBuilder putScalarPublic(String key, Object value) {
            return putScalar(key, value);
        }

        TestBuilder putEnumPublic(String key, CliArg value) {
            return putEnum(key, value);
        }
    }

    @Test
    public void putScalar_int_storesDecimalString() {
        TestBuilder b = new TestBuilder();
        b.putScalarPublic("--threads", 8);
        assertEquals("8", b.parameters.get("--threads"));
    }

    @Test
    public void putScalar_negativeInt_storesSignedDecimal() {
        TestBuilder b = new TestBuilder();
        b.putScalarPublic("--predict", -1);
        assertEquals("-1", b.parameters.get("--predict"));
    }

    @Test
    public void putScalar_zero_storesZero() {
        TestBuilder b = new TestBuilder();
        b.putScalarPublic("--keep", 0);
        assertEquals("0", b.parameters.get("--keep"));
    }

    @Test
    public void putScalar_long_storesDecimalString() {
        TestBuilder b = new TestBuilder();
        b.putScalarPublic("--seed", 4242424242L);
        assertEquals("4242424242", b.parameters.get("--seed"));
    }

    @Test
    public void putScalar_float_storesDotSeparatedDecimal() {
        TestBuilder b = new TestBuilder();
        b.putScalarPublic("--temp", 0.7f);
        // String.valueOf(float) is locale-independent and uses '.' as the decimal separator.
        assertEquals("0.7", b.parameters.get("--temp"));
    }

    @Test
    public void putScalar_double_storesDotSeparatedDecimal() {
        TestBuilder b = new TestBuilder();
        b.putScalarPublic("--top-p", 0.95d);
        assertEquals("0.95", b.parameters.get("--top-p"));
    }

    @Test
    public void putScalar_booleanTrue_storesLowercaseTrue() {
        TestBuilder b = new TestBuilder();
        b.putScalarPublic("--cache", true);
        assertEquals("true", b.parameters.get("--cache"));
    }

    @Test
    public void putScalar_booleanFalse_storesLowercaseFalse() {
        TestBuilder b = new TestBuilder();
        b.putScalarPublic("--cache", false);
        assertEquals("false", b.parameters.get("--cache"));
    }

    @Test
    public void putScalar_overwritesPreviousValue() {
        TestBuilder b = new TestBuilder();
        b.putScalarPublic("--threads", 4);
        b.putScalarPublic("--threads", 16);
        assertEquals("16", b.parameters.get("--threads"));
        assertEquals(1, b.parameters.size());
    }

    @Test
    public void putScalar_returnsSameBuilderInstance() {
        TestBuilder b = new TestBuilder();
        TestBuilder returned = b.putScalarPublic("--threads", 1);
        assertSame(b, returned);
    }

    @Test
    public void putEnum_usesGetArgValueNotEnumName() {
        TestBuilder b = new TestBuilder();
        b.putEnumPublic("--cache-type-k", CacheType.Q8_0);
        assertEquals(CacheType.Q8_0.getArgValue(), b.parameters.get("--cache-type-k"));
        // Sanity check: the stored string is not the Java enum constant name.
        assertEquals("q8_0", b.parameters.get("--cache-type-k"));
    }

    @Test
    public void putEnum_returnsSameBuilderInstance() {
        TestBuilder b = new TestBuilder();
        TestBuilder returned = b.putEnumPublic("--cache-type-k", CacheType.F16);
        assertSame(b, returned);
    }

    @Test
    public void putEnum_overwritesPreviousValue() {
        TestBuilder b = new TestBuilder();
        b.putEnumPublic("--cache-type-k", CacheType.F16);
        b.putEnumPublic("--cache-type-k", CacheType.Q8_0);
        assertEquals("q8_0", b.parameters.get("--cache-type-k"));
        assertEquals(1, b.parameters.size());
    }

    // The CliParameters base class carries the same putScalar / putEnum helpers
    // because ModelParameters does not extend JsonParameters. Verify both
    // helpers work on a CliParameters subclass as well.

    private static final class CliTestBuilder extends CliParameters {
        CliTestBuilder putScalarPublic(String key, Object value) {
            return putScalar(key, value);
        }

        CliTestBuilder putEnumPublic(String key, CliArg value) {
            return putEnum(key, value);
        }
    }

    @Test
    public void cliPutScalar_int_storesDecimalString() {
        CliTestBuilder b = new CliTestBuilder();
        b.putScalarPublic("--threads", 8);
        assertEquals("8", b.parameters.get("--threads"));
    }

    @Test
    public void cliPutScalar_returnsSameBuilderInstance() {
        CliTestBuilder b = new CliTestBuilder();
        CliTestBuilder returned = b.putScalarPublic("--threads", 1);
        assertSame(b, returned);
    }

    @Test
    public void cliPutEnum_usesGetArgValueNotEnumName() {
        CliTestBuilder b = new CliTestBuilder();
        b.putEnumPublic("--cache-type-k", CacheType.Q8_0);
        assertEquals("q8_0", b.parameters.get("--cache-type-k"));
    }

    @Test
    public void cliPutEnum_returnsSameBuilderInstance() {
        CliTestBuilder b = new CliTestBuilder();
        CliTestBuilder returned = b.putEnumPublic("--cache-type-k", CacheType.F16);
        assertSame(b, returned);
    }
}
