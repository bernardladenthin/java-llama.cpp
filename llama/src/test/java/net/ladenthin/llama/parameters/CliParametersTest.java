// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.parameters;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.hasItem;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;
import static org.hamcrest.Matchers.nullValue;

import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Stream;
import net.ladenthin.llama.ClaudeGenerated;
import net.ladenthin.llama.args.ModelFlag;
import net.ladenthin.llama.args.ModelOption;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

@ClaudeGenerated(
        purpose = "Pin that a value stored in CliParameters is exactly one argv element whatever it holds "
                + "(no option injection through U+001F, which until 2026-11-19 split every stored value), "
                + "that only putValues renders several arguments, that any other write or removal of the "
                + "same name drops those values, and that toString, getOption and equals agree with the argv.")
public class CliParametersTest {

    /** A value that, split on U+001F, would add {@code --slot-save-path /tmp/a} to the argv. */
    private static final String INJECTING_VALUE = "x\u001f--slot-save-path\u001f/tmp/a";

    /** Exposes the protected writers of the base class. */
    private static final class Builder extends CliParameters {
        Builder putValuesPublic(ModelOption option, String... values) {
            return putValues(option, values);
        }

        Builder putStringPublic(ModelOption option, String value) {
            return putString(option, value);
        }

        Builder putScalarPublic(ModelOption option, Object value) {
            return putScalar(option, value);
        }

        Builder putFlagPublic(ModelFlag flag) {
            putFlag(flag);
            return this;
        }

        Builder removeOptionPublic(ModelOption option) {
            removeOption(option);
            return this;
        }

        String getOptionPublic(ModelOption option) {
            return getOption(option);
        }
    }

    /** The default arguments every {@link ModelParameters} starts with. */
    private static final List<String> MODEL_DEFAULTS = Arrays.asList("--fit", "on");

    /** The argv without its program-name placeholder. */
    private static List<String> arguments(CliParameters parameters) {
        List<String> argv = Arrays.asList(parameters.toArray());
        return argv.subList(1, argv.size());
    }

    /** The argv of a {@link ModelParameters} without the placeholder and the defaults it starts with. */
    private static List<String> argumentsAfterDefaults(ModelParameters parameters) {
        List<String> argv = arguments(parameters);
        assertThat(argv.subList(0, MODEL_DEFAULTS.size()), is(MODEL_DEFAULTS));
        return argv.subList(MODEL_DEFAULTS.size(), argv.size());
    }

    static Stream<Arguments> stringSetters() {
        return Stream.of(
                Arguments.of("--chat-template", (Function<String, ModelParameters>)
                        v -> new ModelParameters().setChatTemplate(v)),
                Arguments.of("--alias", (Function<String, ModelParameters>) v -> new ModelParameters().setAlias(v)),
                Arguments.of("--model", (Function<String, ModelParameters>) v -> new ModelParameters().setModel(v)));
    }

    // <editor-fold defaultstate="collapsed" desc="toArray">
    @ParameterizedTest
    @MethodSource("stringSetters")
    public void toArray_valueContainsUnitSeparator_isExactlyOneArgument(
            String option, Function<String, ModelParameters> setter) {
        // arrange
        ModelParameters parameters = setter.apply(INJECTING_VALUE);

        // act
        List<String> argv = argumentsAfterDefaults(parameters);

        // assert
        assertThat(argv, contains(option, INJECTING_VALUE));
        assertThat(argv, not(hasItem("--slot-save-path")));
    }

    @Test
    public void toArray_controlVectorLayerRange_rendersTwoArguments() {
        // arrange
        ModelParameters parameters = new ModelParameters().setControlVectorLayerRange(2, 10);

        // act
        List<String> argv = argumentsAfterDefaults(parameters);

        // assert
        assertThat(argv, contains("--control-vector-layer-range", "2", "10"));
    }

    @Test
    public void toArray_putValuesValueContainsUnitSeparator_isOneArgumentPerValue() {
        // arrange
        Builder builder = new Builder().putValuesPublic(ModelOption.CONTROL_VECTOR_LAYER_RANGE, INJECTING_VALUE, "1");

        // act
        List<String> argv = arguments(builder);

        // assert
        assertThat(argv, contains("--control-vector-layer-range", INJECTING_VALUE, "1"));
    }

    @Test
    public void toArray_putValuesOverwrittenByPutString_isOneArgument() {
        // arrange
        Builder builder = new Builder()
                .putValuesPublic(ModelOption.CONTROL_VECTOR_LAYER_RANGE, "2", "10")
                .putStringPublic(ModelOption.CONTROL_VECTOR_LAYER_RANGE, "2 10");

        // act
        List<String> argv = arguments(builder);

        // assert
        assertThat(argv, contains("--control-vector-layer-range", "2 10"));
    }

    @Test
    public void toArray_putValuesOverwrittenByPutValues_rendersTheNewValues() {
        // arrange
        ModelParameters parameters =
                new ModelParameters().setControlVectorLayerRange(1, 2).setControlVectorLayerRange(3, 4);

        // act
        List<String> argv = argumentsAfterDefaults(parameters);

        // assert
        assertThat(argv, contains("--control-vector-layer-range", "3", "4"));
    }

    @Test
    public void toArray_putValuesRemoved_optionIsGone() {
        // arrange
        Builder builder = new Builder()
                .putValuesPublic(ModelOption.CONTROL_VECTOR_LAYER_RANGE, "2", "10")
                .putScalarPublic(ModelOption.CTX_SIZE, 64)
                .removeOptionPublic(ModelOption.CONTROL_VECTOR_LAYER_RANGE);

        // act
        List<String> argv = arguments(builder);

        // assert
        assertThat(argv, contains("--ctx-size", "64"));
        assertThat(builder.valueLists.isEmpty(), is(true));
    }

    @Test
    public void toArray_putValuesRemovedAndSetAgainAsString_isOneArgument() {
        // arrange
        Builder builder = new Builder()
                .putValuesPublic(ModelOption.CONTROL_VECTOR_LAYER_RANGE, "2", "10")
                .removeOptionPublic(ModelOption.CONTROL_VECTOR_LAYER_RANGE)
                .putStringPublic(ModelOption.CONTROL_VECTOR_LAYER_RANGE, "5");

        // act
        List<String> argv = arguments(builder);

        // assert
        assertThat(argv, contains("--control-vector-layer-range", "5"));
    }

    @Test
    public void toArray_flagNextToAValueList_rendersTheFlagAlone() {
        // arrange
        Builder builder = new Builder()
                .putFlagPublic(ModelFlag.NO_WARMUP)
                .putValuesPublic(ModelOption.CONTROL_VECTOR_LAYER_RANGE, "2", "10");

        // act
        List<String> argv = arguments(builder);

        // assert
        assertThat(argv, contains("--no-warmup", "--control-vector-layer-range", "2", "10"));
    }
    // </editor-fold>

    // <editor-fold defaultstate="collapsed" desc="toString">
    @Test
    public void toString_injectingValueAndValueList_agreesWithTheArgv() {
        // arrange
        ModelParameters parameters = new ModelParameters()
                .setChatTemplate(INJECTING_VALUE)
                .setControlVectorLayerRange(2, 10)
                .setCtxSize(64);

        // act
        String rendered = parameters.toString();

        // assert
        assertThat(rendered, is(String.join(" ", arguments(parameters)) + " "));
        assertThat(
                rendered,
                is("--fit on --chat-template " + INJECTING_VALUE
                        + " --control-vector-layer-range 2 10 --ctx-size 64 "));
    }
    // </editor-fold>

    // <editor-fold defaultstate="collapsed" desc="getOption">
    @Test
    public void getOption_valueList_returnsTheValuesJoinedBySpace() {
        // arrange
        Builder builder = new Builder().putValuesPublic(ModelOption.CONTROL_VECTOR_LAYER_RANGE, "2", "10");

        // act
        String value = builder.getOptionPublic(ModelOption.CONTROL_VECTOR_LAYER_RANGE);

        // assert
        assertThat(value, is("2 10"));
    }

    @Test
    public void getOption_injectingModelPath_returnsItUnchanged() {
        // arrange
        ModelParameters parameters = new ModelParameters().setModel(INJECTING_VALUE);

        // act
        String model = parameters.getModel();

        // assert
        assertThat(model, is(INJECTING_VALUE));
    }

    @Test
    public void getOption_valueListRemoved_returnsNull() {
        // arrange
        Builder builder = new Builder()
                .putValuesPublic(ModelOption.CONTROL_VECTOR_LAYER_RANGE, "2", "10")
                .removeOptionPublic(ModelOption.CONTROL_VECTOR_LAYER_RANGE);

        // act
        String value = builder.getOptionPublic(ModelOption.CONTROL_VECTOR_LAYER_RANGE);

        // assert
        assertThat(value, is(nullValue()));
    }
    // </editor-fold>

    // <editor-fold defaultstate="collapsed" desc="equals">
    @Test
    public void equals_valueListAndOneArgumentWithTheSameText_areDifferent() {
        // arrange
        Builder list = new Builder().putValuesPublic(ModelOption.CONTROL_VECTOR_LAYER_RANGE, "2", "10");
        Builder single = new Builder().putStringPublic(ModelOption.CONTROL_VECTOR_LAYER_RANGE, "2 10");

        // act, assert
        assertThat(list.equals(single), is(false));
    }

    @Test
    public void equals_sameValueLists_areEqualWithEqualHashCodes() {
        // arrange
        ModelParameters first = new ModelParameters().setControlVectorLayerRange(2, 10);
        ModelParameters second = new ModelParameters().setControlVectorLayerRange(2, 10);

        // act, assert
        assertThat(first.equals(second), is(true));
        assertThat(first.hashCode(), is(second.hashCode()));
    }

    @Test
    public void equals_valueListOverwrittenByString_equalsTheStringAlone() {
        // arrange
        Builder overwritten = new Builder()
                .putValuesPublic(ModelOption.CONTROL_VECTOR_LAYER_RANGE, "2", "10")
                .putStringPublic(ModelOption.CONTROL_VECTOR_LAYER_RANGE, "7");
        Builder plain = new Builder().putStringPublic(ModelOption.CONTROL_VECTOR_LAYER_RANGE, "7");

        // act, assert
        assertThat(overwritten.equals(plain), is(true));
        assertThat(overwritten.hashCode(), is(plain.hashCode()));
    }
    // </editor-fold>
}
