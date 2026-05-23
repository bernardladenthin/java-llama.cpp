// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import com.fasterxml.jackson.databind.JsonNode;

/**
 * Typed view over the JSON returned by {@link LlamaModel#getMetrics()}.
 * <p>
 * Wraps the underlying {@link JsonNode} so future fields added on the C++ side remain
 * accessible via {@link #asJson()} without code changes here. Mirrors the
 * {@link ModelMeta} pattern.
 * </p>
 * <p>
 * The native {@code server_task_result_metrics::to_json()} emits server-wide counters:
 * {@code idle}, {@code processing}, {@code deferred}, {@code t_start},
 * {@code n_prompt_tokens_processed_total}, {@code t_prompt_processing_total},
 * {@code n_tokens_predicted_total}, {@code t_tokens_generation_total},
 * {@code n_prompt_tokens_processed}, {@code t_prompt_processing},
 * {@code n_tokens_predicted}, {@code t_tokens_generation},
 * {@code n_decode_total}, {@code n_busy_slots_total}, optionally {@code n_tokens_max},
 * and a {@code slots} array.
 * </p>
 */
public final class ServerMetrics {

    private final JsonNode node;

    ServerMetrics(JsonNode node) {
        this.node = node;
    }

    /**
     * Idle slot count.
     * @return number of slots currently idle
     */
    public int getIdleSlots() {
        return node.path("idle").asInt(0);
    }

    /**
     * Busy slot count.
     * @return number of slots currently processing a task
     */
    public int getProcessingSlots() {
        return node.path("processing").asInt(0);
    }

    /**
     * Backlog size.
     * @return number of tasks waiting in the deferred queue
     */
    public int getDeferredTasks() {
        return node.path("deferred").asInt(0);
    }

    /**
     * Server start timestamp.
     * @return millis since epoch as reported by llama.cpp
     */
    public long getStartTimestamp() {
        return node.path("t_start").asLong(0L);
    }

    /**
     * Lifetime decode counter.
     * @return total decode invocations since server start
     */
    public long getDecodeTotal() {
        return node.path("n_decode_total").asLong(0L);
    }

    /**
     * Lifetime busy-slot counter.
     * @return cumulative number of busy-slot ticks (1 per decode per busy slot)
     */
    public long getBusySlotsTotal() {
        return node.path("n_busy_slots_total").asLong(0L);
    }

    /**
     * Maximum number of tokens any active slot is configured to hold. Absent in the
     * upstream JSON when no slot has been used yet; this getter returns {@code 0} then.
     *
     * @return the slot token-budget ceiling, or {@code 0} when no slot has been active
     */
    public int getTokensMax() {
        return node.path("n_tokens_max").asInt(0);
    }

    /**
     * Cumulative server-wide token usage since startup. Prompt tokens come from
     * {@code n_prompt_tokens_processed_total} and completion tokens from
     * {@code n_tokens_predicted_total}.
     *
     * @return cumulative {@link Usage} across all completions since server start
     */
    public Usage getCumulativeUsage() {
        return new Usage(
                node.path("n_prompt_tokens_processed_total").asLong(0L),
                node.path("n_tokens_predicted_total").asLong(0L));
    }

    /**
     * Usage counters from the most recent measurement window (current bucket) &mdash;
     * {@code n_prompt_tokens_processed} and {@code n_tokens_predicted}.
     *
     * @return per-window {@link Usage} since the previous metrics emission
     */
    public Usage getWindowUsage() {
        return new Usage(
                node.path("n_prompt_tokens_processed").asLong(0L),
                node.path("n_tokens_predicted").asLong(0L));
    }

    /**
     * Cumulative throughput derived from the totals fields. Returns {@code 0.0} for
     * any rate where the corresponding ms total is zero.
     *
     * @return cumulative {@link Timings} aggregated from server-wide totals
     */
    public Timings getCumulativeTimings() {
        long promptN = node.path("n_prompt_tokens_processed_total").asLong(0L);
        long predictedN = node.path("n_tokens_predicted_total").asLong(0L);
        double promptMs = node.path("t_prompt_processing_total").asDouble(0.0);
        double predictedMs = node.path("t_tokens_generation_total").asDouble(0.0);
        double promptPerSec = promptMs > 0.0 ? promptN * 1000.0 / promptMs : 0.0;
        double predictedPerSec = predictedMs > 0.0 ? predictedN * 1000.0 / predictedMs : 0.0;
        return new Timings(0, (int) promptN, promptMs, promptPerSec,
                (int) predictedN, predictedMs, predictedPerSec, 0, 0);
    }

    /**
     * Slots array passthrough.
     * @return the {@code slots} array node, or a missing-node when no slots are reported
     */
    public JsonNode getSlots() {
        return node.path("slots");
    }

    /**
     * Raw passthrough escape hatch.
     * @return underlying JSON for direct access to fields not yet exposed by typed getters
     */
    public JsonNode asJson() {
        return node;
    }

    @Override
    public String toString() {
        return node.toString();
    }
}
