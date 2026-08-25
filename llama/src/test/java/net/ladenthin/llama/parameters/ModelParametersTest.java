// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.parameters;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.arrayWithSize;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.hasItem;
import static org.hamcrest.Matchers.hasKey;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;
import static org.hamcrest.Matchers.sameInstance;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Arrays;
import java.util.List;
import net.ladenthin.llama.ClaudeGenerated;
import net.ladenthin.llama.args.CacheType;
import net.ladenthin.llama.args.GpuSplitMode;
import net.ladenthin.llama.args.MiroStat;
import net.ladenthin.llama.args.NumaStrategy;
import net.ladenthin.llama.args.PoolingType;
import net.ladenthin.llama.args.RopeScalingType;
import net.ladenthin.llama.args.Sampler;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose = "Verify ModelParameters input validation (priority 0-3, repeatLastN/dryPenaltyLastN >= -1), "
                + "correct CLI argument formatting for enum-based setters (PoolingType, RopeScalingType, "
                + "CacheType, GpuSplitMode, NumaStrategy, MiroStat) and composite-value setters "
                + "(loraScaled, controlVectorScaled, controlVectorLayerRange), semicolon-separated "
                + "lowercase sampler list, isUnset key-presence check, and the CliParameters base "
                + "behaviour: toString omits 'null' for flag-only entries, toArray always prepends an "
                + "empty argv[0] string and omits values for null-valued flags.")
public class ModelParametersTest {

    // -------------------------------------------------------------------------
    // setPriority — validation (0-3 only)
    // -------------------------------------------------------------------------

    @Test
    public void testSetPriorityValid0() {
        ModelParameters p = new ModelParameters().setPriority(0);
        assertThat(p.parameters.get("--prio"), is("0"));
    }

    @Test
    public void testSetPriorityValid3() {
        ModelParameters p = new ModelParameters().setPriority(3);
        assertThat(p.parameters.get("--prio"), is("3"));
    }

    @Test
    public void testSetPriorityNegative() {
        assertThrows(IllegalArgumentException.class, () -> new ModelParameters().setPriority(-1));
    }

    @Test
    public void testSetPriorityTooHigh() {
        assertThrows(IllegalArgumentException.class, () -> new ModelParameters().setPriority(4));
    }

    // -------------------------------------------------------------------------
    // setPriorityBatch — validation (0-3 only)
    // -------------------------------------------------------------------------

    @Test
    public void testSetPriorityBatchValid1() {
        ModelParameters p = new ModelParameters().setPriorityBatch(1);
        assertThat(p.parameters.get("--prio-batch"), is("1"));
    }

    @Test
    public void testSetPriorityBatchNegative() {
        assertThrows(IllegalArgumentException.class, () -> new ModelParameters().setPriorityBatch(-1));
    }

    @Test
    public void testSetPriorityBatchTooHigh() {
        assertThrows(IllegalArgumentException.class, () -> new ModelParameters().setPriorityBatch(4));
    }

    // -------------------------------------------------------------------------
    // setRepeatLastN — validation (>= -1)
    // -------------------------------------------------------------------------

    @Test
    public void testSetRepeatLastNValidZero() {
        ModelParameters p = new ModelParameters().setRepeatLastN(0);
        assertThat(p.parameters.get("--repeat-last-n"), is("0"));
    }

    @Test
    public void testSetRepeatLastNRejectsMinusOne() {
        // llama.cpp b10275 dropped the -1 = ctx_size sentinel; common_params_parse now throws on a
        // negative value, so the model would fail to load. Reject it here where the message can say why.
        assertThrows(IllegalArgumentException.class, () -> new ModelParameters().setRepeatLastN(-1));
    }

    @Test
    public void testSetRepeatLastNValid64() {
        ModelParameters p = new ModelParameters().setRepeatLastN(64);
        assertThat(p.parameters.get("--repeat-last-n"), is("64"));
    }

    @Test
    public void testSetRepeatLastNTooLow() {
        assertThrows(IllegalArgumentException.class, () -> new ModelParameters().setRepeatLastN(-2));
    }

    // -------------------------------------------------------------------------
    // setDryPenaltyLastN — validation (>= 0 since llama.cpp b10275)
    // -------------------------------------------------------------------------

    @Test
    public void testSetDryPenaltyLastNRejectsMinusOne() {
        // Same b10275 change as setRepeatLastN: --dry-penalty-last-n -1 makes common_params_parse throw.
        assertThrows(IllegalArgumentException.class, () -> new ModelParameters().setDryPenaltyLastN(-1));
    }

    @Test
    public void testSetDryPenaltyLastNValidPositive() {
        ModelParameters p = new ModelParameters().setDryPenaltyLastN(256);
        assertThat(p.parameters.get("--dry-penalty-last-n"), is("256"));
    }

    @Test
    public void testSetDryPenaltyLastNValidZero() {
        ModelParameters p = new ModelParameters().setDryPenaltyLastN(0);
        assertThat(p.parameters.get("--dry-penalty-last-n"), is("0"));
    }

    @Test
    public void testSetDryPenaltyLastNTooLow() {
        assertThrows(IllegalArgumentException.class, () -> new ModelParameters().setDryPenaltyLastN(-2));
    }

    // -------------------------------------------------------------------------
    // setSamplers — semicolon-separated lowercase names
    // -------------------------------------------------------------------------

    @Test
    public void testSetSamplersSingle() {
        ModelParameters p = new ModelParameters().setSamplers(Sampler.TOP_K);
        assertThat(p.parameters.get("--samplers"), is("top_k"));
    }

    @Test
    public void testSetSamplersMultiple() {
        ModelParameters p = new ModelParameters().setSamplers(Sampler.TOP_K, Sampler.TOP_P, Sampler.TEMPERATURE);
        assertThat(p.parameters.get("--samplers"), is("top_k;top_p;temperature"));
    }

    @Test
    public void testSetSamplersEmpty() {
        ModelParameters p = new ModelParameters().setSamplers();
        assertThat(p.parameters, not(hasKey("--samplers")));
    }

    @Test
    public void testSetSamplersAllLowercase() {
        for (Sampler s : Sampler.values()) {
            ModelParameters p = new ModelParameters().setSamplers(s);
            assertThat(p.parameters.get("--samplers"), is(s.name().toLowerCase()));
        }
    }

    // -------------------------------------------------------------------------
    // addLoraScaledAdapter / addControlVectorScaled — "fname,scale" format
    // -------------------------------------------------------------------------

    @Test
    public void testAddLoraScaledAdapter() {
        ModelParameters p = new ModelParameters().addLoraScaledAdapter("adapter.bin", 0.5f);
        assertThat(p.parameters.get("--lora-scaled"), is("adapter.bin,0.5"));
    }

    @Test
    public void testAddControlVectorScaled() {
        ModelParameters p = new ModelParameters().addControlVectorScaled("vec.bin", 1.5f);
        assertThat(p.parameters.get("--control-vector-scaled"), is("vec.bin,1.5"));
    }

    // -------------------------------------------------------------------------
    // setControlVectorLayerRange — "start,end" format
    // -------------------------------------------------------------------------

    @Test
    public void testSetControlVectorLayerRange() {
        ModelParameters p = new ModelParameters().setControlVectorLayerRange(2, 10);
        assertThat(p.parameters.get("--control-vector-layer-range"), is("2,10"));
    }

    @Test
    public void testSetControlVectorLayerRangeSameStartEnd() {
        ModelParameters p = new ModelParameters().setControlVectorLayerRange(5, 5);
        assertThat(p.parameters.get("--control-vector-layer-range"), is("5,5"));
    }

    // -------------------------------------------------------------------------
    // isUnset
    // -------------------------------------------------------------------------

    @Test
    public void testIsDefaultTrueWhenNotSet() {
        ModelParameters p = new ModelParameters();
        assertThat(p.isUnset("threads"), is(true));
    }

    @Test
    public void testIsDefaultFalseWhenSet() {
        ModelParameters p = new ModelParameters().setThreads(4);
        assertThat(p.isUnset("threads"), is(false));
    }

    @Test
    public void testIsDefaultFalseAfterFlagOnly() {
        ModelParameters p = new ModelParameters().enableEmbedding();
        assertThat(p.isUnset("embedding"), is(false));
    }

    // -------------------------------------------------------------------------
    // Enum-based setters (PoolingType, RopeScalingType, CacheType, etc.)
    // -------------------------------------------------------------------------

    @Test
    public void testSetPoolingTypeMean() {
        ModelParameters p = new ModelParameters().setPoolingType(PoolingType.MEAN);
        assertThat(p.parameters.get(ModelParameters.ARG_POOLING), is(PoolingType.MEAN.getArgValue()));
    }

    @Test
    public void testSetPoolingTypeNone() {
        ModelParameters p = new ModelParameters().setPoolingType(PoolingType.NONE);
        assertThat(p.parameters.get(ModelParameters.ARG_POOLING), is(PoolingType.NONE.getArgValue()));
    }

    @Test
    public void testSetPoolingTypeCls() {
        ModelParameters p = new ModelParameters().setPoolingType(PoolingType.CLS);
        assertThat(p.parameters.get(ModelParameters.ARG_POOLING), is(PoolingType.CLS.getArgValue()));
    }

    @Test
    public void testSetPoolingTypeLast() {
        ModelParameters p = new ModelParameters().setPoolingType(PoolingType.LAST);
        assertThat(p.parameters.get(ModelParameters.ARG_POOLING), is(PoolingType.LAST.getArgValue()));
    }

    @Test
    public void testSetPoolingTypeRank() {
        ModelParameters p = new ModelParameters().setPoolingType(PoolingType.RANK);
        assertThat(p.parameters.get(ModelParameters.ARG_POOLING), is(PoolingType.RANK.getArgValue()));
    }

    @Test
    public void testSetPoolingTypeUnspecifiedDoesNotSetParam() {
        ModelParameters p = new ModelParameters().setPoolingType(PoolingType.UNSPECIFIED);
        assertThat(
                "UNSPECIFIED pooling type must not add " + ModelParameters.ARG_POOLING + " to parameters",
                p.parameters,
                not(hasKey(ModelParameters.ARG_POOLING)));
    }

    @Test
    public void testSetPoolingTypeUnspecifiedLeavesDefaultUntouched() {
        // A fresh ModelParameters must not have ARG_POOLING set by default either
        ModelParameters fresh = new ModelParameters();
        assertThat(fresh.parameters, not(hasKey(ModelParameters.ARG_POOLING)));
        // Calling setPoolingType(UNSPECIFIED) must leave that invariant intact
        fresh.setPoolingType(PoolingType.UNSPECIFIED);
        assertThat(fresh.parameters, not(hasKey(ModelParameters.ARG_POOLING)));
    }

    @Test
    public void testSetRopeScaling() {
        ModelParameters p = new ModelParameters().setRopeScaling(RopeScalingType.YARN2);
        assertThat(p.parameters.get("--rope-scaling"), is("yarn"));
    }

    @Test
    public void testSetCacheTypeKLowercase() {
        ModelParameters p = new ModelParameters().setCacheTypeK(CacheType.F16);
        assertThat(p.parameters.get("--cache-type-k"), is("f16"));
    }

    @Test
    public void testSetCacheTypeVLowercase() {
        ModelParameters p = new ModelParameters().setCacheTypeV(CacheType.Q8_0);
        assertThat(p.parameters.get("--cache-type-v"), is("q8_0"));
    }

    @Test
    public void testSetSplitModeLowercase() {
        ModelParameters p = new ModelParameters().setSplitMode(GpuSplitMode.LAYER);
        assertThat(p.parameters.get("--split-mode"), is("layer"));
    }

    @Test
    public void testSetNumaLowercase() {
        ModelParameters p = new ModelParameters().setNuma(NumaStrategy.DISTRIBUTE);
        assertThat(p.parameters.get("--numa"), is("distribute"));
    }

    @Test
    public void testSetMirostatOrdinal() {
        ModelParameters p = new ModelParameters().setMirostat(MiroStat.V2);
        assertThat(p.parameters.get("--mirostat"), is("2"));
    }

    // -------------------------------------------------------------------------
    // CliParameters.toString() — space-separated key[space value] pairs
    // -------------------------------------------------------------------------

    @Test
    public void testToStringContainsKey() {
        ModelParameters p = new ModelParameters().setThreads(4);
        assertThat(p.toString(), containsString("--threads"));
        assertThat(p.toString(), containsString("4"));
    }

    @Test
    public void testToStringFlagOnlyNoValue() {
        ModelParameters p = new ModelParameters().enableEmbedding();
        String s = p.toString();
        assertThat(s, containsString("--embedding"));
        // Flag-only: value is null, so no "null" text should appear
        assertThat(s, not(containsString("null")));
    }

    @Test
    public void testFitValueTrueReturnsFitOn() {
        assertThat(ModelParameters.fitValue(true), is(ModelParameters.FIT_ON));
    }

    @Test
    public void testFitValueFalseReturnsFitOff() {
        assertThat(ModelParameters.fitValue(false), is(ModelParameters.FIT_OFF));
    }

    @Test
    public void testToStringDefaultContainsFit() {
        ModelParameters p = new ModelParameters();
        String s = p.toString();
        assertThat(s, containsString("--fit"));
        assertThat(s, containsString(ModelParameters.DEFAULT_FIT_VALUE));
    }

    // -------------------------------------------------------------------------
    // CliParameters.toArray() — leading empty string + key/value pairs
    // -------------------------------------------------------------------------

    @Test
    public void testToArrayDefaultParametersHasFit() {
        // toArray() = ["", "--fit", DEFAULT_FIT_VALUE]
        ModelParameters p = new ModelParameters();
        String[] arr = p.toArray();
        assertThat(arr, arrayWithSize(3));
        assertThat(arr[0], is(""));
        List<String> list = Arrays.asList(arr);
        assertThat(list, hasItem("--fit"));
        assertThat(list, hasItem(ModelParameters.DEFAULT_FIT_VALUE));
    }

    @Test
    public void testToArrayScalarParameterHasFiveElements() {
        // argv[0]="" + "--fit" + DEFAULT_FIT_VALUE + "--threads" + "4" = 5
        ModelParameters p = new ModelParameters().setThreads(4);
        String[] arr = p.toArray();
        assertThat(arr, arrayWithSize(5));
        assertThat(arr[0], is(""));
        List<String> list = Arrays.asList(arr);
        assertThat(list, hasItem("--threads"));
        assertThat(list, hasItem("4"));
        assertThat(list, hasItem("--fit"));
        assertThat(list, hasItem(ModelParameters.DEFAULT_FIT_VALUE));
    }

    @Test
    public void testToArrayFlagOnlyHasFourElements() {
        // argv[0]="" + "--fit" + DEFAULT_FIT_VALUE + "--embedding" (no value) = 4
        ModelParameters p = new ModelParameters().enableEmbedding();
        String[] arr = p.toArray();
        assertThat(arr, arrayWithSize(4));
        assertThat(arr[0], is(""));
        List<String> list = Arrays.asList(arr);
        assertThat(list, hasItem("--embedding"));
        assertThat(list, hasItem("--fit"));
        assertThat(list, hasItem(ModelParameters.DEFAULT_FIT_VALUE));
    }

    @Test
    public void testToArrayMultipleParameters() {
        ModelParameters p = new ModelParameters().setThreads(4).enableEmbedding();
        String[] arr = p.toArray();
        // 1 (argv[0]) + 2 (--fit DEFAULT_FIT_VALUE) + 2 (--threads 4) + 1 (--embedding) = 6
        assertThat(arr, arrayWithSize(6));
        assertThat(arr[0], is(""));
        List<String> list = Arrays.asList(arr);
        assertThat(list, hasItem("--threads"));
        assertThat(list, hasItem("4"));
        assertThat(list, hasItem("--embedding"));
        assertThat(list, hasItem("--fit"));
        assertThat(list, hasItem(ModelParameters.DEFAULT_FIT_VALUE));
    }

    // -------------------------------------------------------------------------
    // Builder chaining returns same instance
    // -------------------------------------------------------------------------

    @Test
    public void testBuilderChainingReturnsSameInstance() {
        ModelParameters p = new ModelParameters();
        assertThat(p.setThreads(4), is(sameInstance(p)));
        assertThat(p.setGpuLayers(10), is(sameInstance(p)));
        assertThat(p.enableEmbedding(), is(sameInstance(p)));
    }

    // -------------------------------------------------------------------------
    // mmproj — vision model projection file/url
    // -------------------------------------------------------------------------

    @Test
    public void testSetMmproj() {
        ModelParameters p = new ModelParameters().setMmproj("/models/mmproj.gguf");
        assertThat(p.parameters.get("--mmproj"), is("/models/mmproj.gguf"));
    }

    @Test
    public void testSetMmprojUrl() {
        ModelParameters p = new ModelParameters().setMmprojUrl("https://example.com/mmproj.gguf");
        assertThat(p.parameters.get("--mmproj-url"), is("https://example.com/mmproj.gguf"));
    }

    @Test
    public void testEnableMmprojAuto() {
        ModelParameters p = new ModelParameters().enableMmprojAuto();
        assertThat(p.parameters, hasKey("--mmproj-auto"));
    }

    @Test
    public void testDisableMmprojAuto() {
        ModelParameters p = new ModelParameters().enableMmprojAuto().setMmprojAuto(false);
        assertThat(p.parameters, hasKey("--no-mmproj-auto"));
        assertThat(p.parameters, not(hasKey("--mmproj-auto")));
    }

    @Test
    public void testSetMmprojDevice() {
        ModelParameters p = new ModelParameters().setMmprojDevice("CUDA1");
        assertThat(p.parameters.get("--mmproj-device"), is("CUDA1"));
    }

    @Test
    public void testSetMmprojDeviceNoneIsPassedThroughVerbatim() {
        // "none" is upstream's sentinel for "do not offload the projector"; it must reach the
        // native parser as-is rather than being translated into --no-mmproj-offload here.
        ModelParameters p = new ModelParameters().setMmprojDevice("none");
        assertThat(p.parameters.get("--mmproj-device"), is("none"));
        assertThat(p.parameters, not(hasKey("--no-mmproj-offload")));
    }

    @Test
    public void testSetMmprojDeviceIsIndependentOfTheMainModelDevices() {
        ModelParameters p = new ModelParameters().setDevices("CUDA0").setMmprojDevice("CUDA1");
        assertThat(p.parameters.get("--device"), is("CUDA0"));
        assertThat(p.parameters.get("--mmproj-device"), is("CUDA1"));
    }

    @Test
    public void testEnableMmprojOffload() {
        ModelParameters p = new ModelParameters().enableMmprojOffload();
        assertThat(p.parameters, hasKey("--mmproj-offload"));
    }

    @Test
    public void testDisableMmprojOffload() {
        ModelParameters p = new ModelParameters().enableMmprojOffload().setMmprojOffload(false);
        assertThat(p.parameters, hasKey("--no-mmproj-offload"));
        assertThat(p.parameters, not(hasKey("--mmproj-offload")));
    }

    // -------------------------------------------------------------------------
    // Reasoning format / budget — model-level defaults for thinking models
    // -------------------------------------------------------------------------

    @Test
    public void testSetReasoningFormatNone() {
        ModelParameters p = new ModelParameters().setReasoningFormat(net.ladenthin.llama.args.ReasoningFormat.NONE);
        assertThat(p.parameters.get("--reasoning-format"), is("none"));
    }

    @Test
    public void testSetReasoningFormatAuto() {
        ModelParameters p = new ModelParameters().setReasoningFormat(net.ladenthin.llama.args.ReasoningFormat.AUTO);
        assertThat(p.parameters.get("--reasoning-format"), is("auto"));
    }

    @Test
    public void testSetReasoningFormatDeepseek() {
        ModelParameters p = new ModelParameters().setReasoningFormat(net.ladenthin.llama.args.ReasoningFormat.DEEPSEEK);
        assertThat(p.parameters.get("--reasoning-format"), is("deepseek"));
    }

    @Test
    public void testSetReasoningFormatDeepseekLegacy() {
        ModelParameters p =
                new ModelParameters().setReasoningFormat(net.ladenthin.llama.args.ReasoningFormat.DEEPSEEK_LEGACY);
        assertThat(p.parameters.get("--reasoning-format"), is("deepseek-legacy"));
    }

    @Test
    public void testSetReasoningBudgetPositive() {
        ModelParameters p = new ModelParameters().setReasoningBudget(1024);
        assertThat(p.parameters.get("--reasoning-budget"), is("1024"));
    }

    @Test
    public void testSetReasoningBudgetDisabled() {
        ModelParameters p = new ModelParameters().setReasoningBudget(-1);
        assertThat(p.parameters.get("--reasoning-budget"), is("-1"));
    }

    // -------------------------------------------------------------------------
    // setSleepIdleSeconds
    // -------------------------------------------------------------------------

    @Test
    public void testSetSleepIdleSeconds() {
        ModelParameters p = new ModelParameters().setSleepIdleSeconds(60);
        assertThat(p.parameters.get("--sleep-idle-seconds"), is("60"));
    }

    @Test
    public void testSetSleepIdleSecondsZero() {
        ModelParameters p = new ModelParameters().setSleepIdleSeconds(0);
        assertThat(p.parameters.get("--sleep-idle-seconds"), is("0"));
    }

    // -------------------------------------------------------------------------
    // setClearIdle / setKvUnified — correct flag names (regression)
    // -------------------------------------------------------------------------

    @Test
    public void testSetClearIdleTrue_usesCacheIdleSlotsFlag() {
        ModelParameters p = new ModelParameters().setClearIdle(true);
        assertThat(p.parameters, hasKey("--cache-idle-slots"));
        assertThat(p.parameters, not(hasKey("--no-cache-idle-slots")));
    }

    @Test
    public void testSetClearIdleFalse_usesNoCacheIdleSlotsFlag() {
        ModelParameters p = new ModelParameters().setClearIdle(false);
        assertThat(p.parameters, hasKey("--no-cache-idle-slots"));
        assertThat(p.parameters, not(hasKey("--cache-idle-slots")));
    }
}
