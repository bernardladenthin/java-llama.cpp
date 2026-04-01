package de.kherud.llama;

import de.kherud.llama.args.CacheType;
import de.kherud.llama.args.GpuSplitMode;
import de.kherud.llama.args.MiroStat;
import de.kherud.llama.args.NumaStrategy;
import de.kherud.llama.args.PoolingType;
import de.kherud.llama.args.RopeScalingType;
import de.kherud.llama.args.Sampler;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;

@ClaudeGenerated(
        purpose = "Verify ModelParameters input validation (priority 0-3, repeatLastN/dryPenaltyLastN >= -1), " +
                  "correct CLI argument formatting for enum-based setters (PoolingType, RopeScalingType, " +
                  "CacheType, GpuSplitMode, NumaStrategy, MiroStat) and composite-value setters " +
                  "(loraScaled, controlVectorScaled, controlVectorLayerRange), semicolon-separated " +
                  "lowercase sampler list, isDefault key-presence check, and the CliParameters base " +
                  "behaviour: toString omits 'null' for flag-only entries, toArray always prepends an " +
                  "empty argv[0] string and omits values for null-valued flags."
)
public class ModelParametersTest {

	// -------------------------------------------------------------------------
	// setPriority — validation (0-3 only)
	// -------------------------------------------------------------------------

	@Test
	public void testSetPriorityValid0() {
		ModelParameters p = new ModelParameters().setPriority(0);
		assertEquals("0", p.parameters.get("--prio"));
	}

	@Test
	public void testSetPriorityValid3() {
		ModelParameters p = new ModelParameters().setPriority(3);
		assertEquals("3", p.parameters.get("--prio"));
	}

	@Test(expected = IllegalArgumentException.class)
	public void testSetPriorityNegative() {
		new ModelParameters().setPriority(-1);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testSetPriorityTooHigh() {
		new ModelParameters().setPriority(4);
	}

	// -------------------------------------------------------------------------
	// setPriorityBatch — validation (0-3 only)
	// -------------------------------------------------------------------------

	@Test
	public void testSetPriorityBatchValid1() {
		ModelParameters p = new ModelParameters().setPriorityBatch(1);
		assertEquals("1", p.parameters.get("--prio-batch"));
	}

	@Test(expected = IllegalArgumentException.class)
	public void testSetPriorityBatchNegative() {
		new ModelParameters().setPriorityBatch(-1);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testSetPriorityBatchTooHigh() {
		new ModelParameters().setPriorityBatch(4);
	}

	// -------------------------------------------------------------------------
	// setRepeatLastN — validation (>= -1)
	// -------------------------------------------------------------------------

	@Test
	public void testSetRepeatLastNValidZero() {
		ModelParameters p = new ModelParameters().setRepeatLastN(0);
		assertEquals("0", p.parameters.get("--repeat-last-n"));
	}

	@Test
	public void testSetRepeatLastNValidMinusOne() {
		ModelParameters p = new ModelParameters().setRepeatLastN(-1);
		assertEquals("-1", p.parameters.get("--repeat-last-n"));
	}

	@Test
	public void testSetRepeatLastNValid64() {
		ModelParameters p = new ModelParameters().setRepeatLastN(64);
		assertEquals("64", p.parameters.get("--repeat-last-n"));
	}

	@Test(expected = RuntimeException.class)
	public void testSetRepeatLastNTooLow() {
		new ModelParameters().setRepeatLastN(-2);
	}

	// -------------------------------------------------------------------------
	// setDryPenaltyLastN — validation (>= -1)
	// -------------------------------------------------------------------------

	@Test
	public void testSetDryPenaltyLastNValidMinusOne() {
		ModelParameters p = new ModelParameters().setDryPenaltyLastN(-1);
		assertEquals("-1", p.parameters.get("--dry-penalty-last-n"));
	}

	@Test
	public void testSetDryPenaltyLastNValidZero() {
		ModelParameters p = new ModelParameters().setDryPenaltyLastN(0);
		assertEquals("0", p.parameters.get("--dry-penalty-last-n"));
	}

	@Test(expected = RuntimeException.class)
	public void testSetDryPenaltyLastNTooLow() {
		new ModelParameters().setDryPenaltyLastN(-2);
	}

	// -------------------------------------------------------------------------
	// setSamplers — semicolon-separated lowercase names
	// -------------------------------------------------------------------------

	@Test
	public void testSetSamplersSingle() {
		ModelParameters p = new ModelParameters().setSamplers(Sampler.TOP_K);
		assertEquals("top_k", p.parameters.get("--samplers"));
	}

	@Test
	public void testSetSamplersMultiple() {
		ModelParameters p = new ModelParameters().setSamplers(Sampler.TOP_K, Sampler.TOP_P, Sampler.TEMPERATURE);
		assertEquals("top_k;top_p;temperature", p.parameters.get("--samplers"));
	}

	@Test
	public void testSetSamplersEmpty() {
		ModelParameters p = new ModelParameters().setSamplers();
		assertFalse(p.parameters.containsKey("--samplers"));
	}

	@Test
	public void testSetSamplersAllLowercase() {
		for (Sampler s : Sampler.values()) {
			ModelParameters p = new ModelParameters().setSamplers(s);
			assertEquals(s.name().toLowerCase(), p.parameters.get("--samplers"));
		}
	}

	// -------------------------------------------------------------------------
	// addLoraScaledAdapter / addControlVectorScaled — "fname,scale" format
	// -------------------------------------------------------------------------

	@Test
	public void testAddLoraScaledAdapter() {
		ModelParameters p = new ModelParameters().addLoraScaledAdapter("adapter.bin", 0.5f);
		assertEquals("adapter.bin,0.5", p.parameters.get("--lora-scaled"));
	}

	@Test
	public void testAddControlVectorScaled() {
		ModelParameters p = new ModelParameters().addControlVectorScaled("vec.bin", 1.5f);
		assertEquals("vec.bin,1.5", p.parameters.get("--control-vector-scaled"));
	}

	// -------------------------------------------------------------------------
	// setControlVectorLayerRange — "start,end" format
	// -------------------------------------------------------------------------

	@Test
	public void testSetControlVectorLayerRange() {
		ModelParameters p = new ModelParameters().setControlVectorLayerRange(2, 10);
		assertEquals("2,10", p.parameters.get("--control-vector-layer-range"));
	}

	@Test
	public void testSetControlVectorLayerRangeSameStartEnd() {
		ModelParameters p = new ModelParameters().setControlVectorLayerRange(5, 5);
		assertEquals("5,5", p.parameters.get("--control-vector-layer-range"));
	}

	// -------------------------------------------------------------------------
	// isDefault
	// -------------------------------------------------------------------------

	@Test
	public void testIsDefaultTrueWhenNotSet() {
		ModelParameters p = new ModelParameters();
		assertTrue(p.isDefault("threads"));
	}

	@Test
	public void testIsDefaultFalseWhenSet() {
		ModelParameters p = new ModelParameters().setThreads(4);
		assertFalse(p.isDefault("threads"));
	}

	@Test
	public void testIsDefaultFalseAfterFlagOnly() {
		ModelParameters p = new ModelParameters().enableEmbedding();
		assertFalse(p.isDefault("embedding"));
	}

	// -------------------------------------------------------------------------
	// Enum-based setters (PoolingType, RopeScalingType, CacheType, etc.)
	// -------------------------------------------------------------------------

	@Test
	public void testSetPoolingTypeMean() {
		ModelParameters p = new ModelParameters().setPoolingType(PoolingType.MEAN);
		assertEquals(PoolingType.MEAN.getArgValue(), p.parameters.get(ModelParameters.ARG_POOLING));
	}

	@Test
	public void testSetPoolingTypeNone() {
		ModelParameters p = new ModelParameters().setPoolingType(PoolingType.NONE);
		assertEquals(PoolingType.NONE.getArgValue(), p.parameters.get(ModelParameters.ARG_POOLING));
	}

	@Test
	public void testSetPoolingTypeCls() {
		ModelParameters p = new ModelParameters().setPoolingType(PoolingType.CLS);
		assertEquals(PoolingType.CLS.getArgValue(), p.parameters.get(ModelParameters.ARG_POOLING));
	}

	@Test
	public void testSetPoolingTypeLast() {
		ModelParameters p = new ModelParameters().setPoolingType(PoolingType.LAST);
		assertEquals(PoolingType.LAST.getArgValue(), p.parameters.get(ModelParameters.ARG_POOLING));
	}

	@Test
	public void testSetPoolingTypeRank() {
		ModelParameters p = new ModelParameters().setPoolingType(PoolingType.RANK);
		assertEquals(PoolingType.RANK.getArgValue(), p.parameters.get(ModelParameters.ARG_POOLING));
	}

	@Test
	public void testSetPoolingTypeUnspecifiedDoesNotSetParam() {
		ModelParameters p = new ModelParameters().setPoolingType(PoolingType.UNSPECIFIED);
		assertFalse("UNSPECIFIED pooling type must not add " + ModelParameters.ARG_POOLING + " to parameters",
				p.parameters.containsKey(ModelParameters.ARG_POOLING));
	}

	@Test
	public void testSetPoolingTypeUnspecifiedLeavesDefaultUntouched() {
		// A fresh ModelParameters must not have ARG_POOLING set by default either
		ModelParameters fresh = new ModelParameters();
		assertFalse(fresh.parameters.containsKey(ModelParameters.ARG_POOLING));
		// Calling setPoolingType(UNSPECIFIED) must leave that invariant intact
		fresh.setPoolingType(PoolingType.UNSPECIFIED);
		assertFalse(fresh.parameters.containsKey(ModelParameters.ARG_POOLING));
	}

	@Test
	public void testSetRopeScaling() {
		ModelParameters p = new ModelParameters().setRopeScaling(RopeScalingType.YARN2);
		assertEquals("yarn", p.parameters.get("--rope-scaling"));
	}

	@Test
	public void testSetCacheTypeKLowercase() {
		ModelParameters p = new ModelParameters().setCacheTypeK(CacheType.F16);
		assertEquals("f16", p.parameters.get("--cache-type-k"));
	}

	@Test
	public void testSetCacheTypeVLowercase() {
		ModelParameters p = new ModelParameters().setCacheTypeV(CacheType.Q8_0);
		assertEquals("q8_0", p.parameters.get("--cache-type-v"));
	}

	@Test
	public void testSetSplitModeLowercase() {
		ModelParameters p = new ModelParameters().setSplitMode(GpuSplitMode.LAYER);
		assertEquals("layer", p.parameters.get("--split-mode"));
	}

	@Test
	public void testSetNumaLowercase() {
		ModelParameters p = new ModelParameters().setNuma(NumaStrategy.DISTRIBUTE);
		assertEquals("distribute", p.parameters.get("--numa"));
	}

	@Test
	public void testSetMirostatOrdinal() {
		ModelParameters p = new ModelParameters().setMirostat(MiroStat.V2);
		assertEquals("2", p.parameters.get("--mirostat"));
	}

	// -------------------------------------------------------------------------
	// CliParameters.toString() — space-separated key[space value] pairs
	// -------------------------------------------------------------------------

	@Test
	public void testToStringContainsKey() {
		ModelParameters p = new ModelParameters().setThreads(4);
		assertTrue(p.toString().contains("--threads"));
		assertTrue(p.toString().contains("4"));
	}

	@Test
	public void testToStringFlagOnlyNoValue() {
		ModelParameters p = new ModelParameters().enableEmbedding();
		String s = p.toString();
		assertTrue(s.contains("--embedding"));
		// Flag-only: value is null, so no "null" text should appear
		assertFalse(s.contains("null"));
	}

	@Test
	public void testFitValueTrueReturnsFitOn() {
		assertEquals(ModelParameters.FIT_ON, ModelParameters.fitValue(true));
	}

	@Test
	public void testFitValueFalseReturnsFitOff() {
		assertEquals(ModelParameters.FIT_OFF, ModelParameters.fitValue(false));
	}

	@Test
	public void testToStringDefaultContainsFit() {
		ModelParameters p = new ModelParameters();
		String s = p.toString();
		assertTrue(s.contains("--fit"));
		assertTrue(s.contains(ModelParameters.DEFAULT_FIT_VALUE));
	}

	// -------------------------------------------------------------------------
	// CliParameters.toArray() — leading empty string + key/value pairs
	// -------------------------------------------------------------------------

	@Test
	public void testToArrayDefaultParametersHasFit() {
		// toArray() = ["", "--fit", DEFAULT_FIT_VALUE]
		ModelParameters p = new ModelParameters();
		String[] arr = p.toArray();
		assertEquals(3, arr.length);
		assertEquals("", arr[0]);
		List<String> list = Arrays.asList(arr);
		assertTrue(list.contains("--fit"));
		assertTrue(list.contains(ModelParameters.DEFAULT_FIT_VALUE));
	}

	@Test
	public void testToArrayScalarParameterHasFiveElements() {
		// argv[0]="" + "--fit" + DEFAULT_FIT_VALUE + "--threads" + "4" = 5
		ModelParameters p = new ModelParameters().setThreads(4);
		String[] arr = p.toArray();
		assertEquals(5, arr.length);
		assertEquals("", arr[0]);
		List<String> list = Arrays.asList(arr);
		assertTrue(list.contains("--threads"));
		assertTrue(list.contains("4"));
		assertTrue(list.contains("--fit"));
		assertTrue(list.contains(ModelParameters.DEFAULT_FIT_VALUE));
	}

	@Test
	public void testToArrayFlagOnlyHasFourElements() {
		// argv[0]="" + "--fit" + DEFAULT_FIT_VALUE + "--embedding" (no value) = 4
		ModelParameters p = new ModelParameters().enableEmbedding();
		String[] arr = p.toArray();
		assertEquals(4, arr.length);
		assertEquals("", arr[0]);
		List<String> list = Arrays.asList(arr);
		assertTrue(list.contains("--embedding"));
		assertTrue(list.contains("--fit"));
		assertTrue(list.contains(ModelParameters.DEFAULT_FIT_VALUE));
	}

	@Test
	public void testToArrayMultipleParameters() {
		ModelParameters p = new ModelParameters()
				.setThreads(4)
				.enableEmbedding();
		String[] arr = p.toArray();
		// 1 (argv[0]) + 2 (--fit DEFAULT_FIT_VALUE) + 2 (--threads 4) + 1 (--embedding) = 6
		assertEquals(6, arr.length);
		assertEquals("", arr[0]);
		List<String> list = Arrays.asList(arr);
		assertTrue(list.contains("--threads"));
		assertTrue(list.contains("4"));
		assertTrue(list.contains("--embedding"));
		assertTrue(list.contains("--fit"));
		assertTrue(list.contains(ModelParameters.DEFAULT_FIT_VALUE));
	}

	// -------------------------------------------------------------------------
	// Builder chaining returns same instance
	// -------------------------------------------------------------------------

	@Test
	public void testBuilderChainingReturnsSameInstance() {
		ModelParameters p = new ModelParameters();
		assertSame(p, p.setThreads(4));
		assertSame(p, p.setGpuLayers(10));
		assertSame(p, p.enableEmbedding());
	}
}
