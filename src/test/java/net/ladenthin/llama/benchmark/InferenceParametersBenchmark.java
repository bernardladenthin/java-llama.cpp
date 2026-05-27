// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
package net.ladenthin.llama.benchmark;

import java.util.concurrent.TimeUnit;
import net.ladenthin.llama.InferenceParameters;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

/**
 * Throughput benchmark for {@link InferenceParameters} JSON serialization.
 *
 * <p>{@link InferenceParameters#toString()} serializes the parameter map to a JSON string
 * that is passed across the JNI boundary on every inference call. This benchmark
 * measures the allocation and serialization cost of building a typical parameter set.</p>
 *
 * <p>Run locally:</p>
 * <pre>
 * mvn test-compile exec:java -Dexec.args="InferenceParametersBenchmark -prof gc"
 * </pre>
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@State(Scope.Thread)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 2)
@Fork(1)
public class InferenceParametersBenchmark {

    /**
     * Serializes a minimal {@link InferenceParameters} instance to JSON.
     *
     * <p>Baseline: measures the cost of constructing a parameter object with
     * no custom settings — the default path every inference call takes.</p>
     *
     * @param bh JMH blackhole to prevent dead-code elimination
     */
    @Benchmark
    public void serializeDefault(Blackhole bh) {
        bh.consume(new InferenceParameters("").toString());
    }

    /**
     * Serializes a fully-populated {@link InferenceParameters} instance to JSON.
     *
     * <p>Measures the serialization overhead when callers set several sampling
     * parameters — representative of a typical chat inference request.</p>
     *
     * @param bh JMH blackhole to prevent dead-code elimination
     */
    @Benchmark
    public void serializeWithSamplingParams(Blackhole bh) {
        bh.consume(new InferenceParameters("")
                .setTemperature(0.7f)
                .setTopP(0.9f)
                .setNPredict(512)
                .setStopStrings("</s>", "<|im_end|>")
                .toString());
    }
}
