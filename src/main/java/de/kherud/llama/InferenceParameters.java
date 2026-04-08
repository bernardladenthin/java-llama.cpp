package de.kherud.llama;

import java.util.Collection;
import java.util.List;
import java.util.Map;

import de.kherud.llama.args.MiroStat;
import de.kherud.llama.args.Sampler;

/**
 * Parameters used throughout inference of a {@link LlamaModel}, e.g., {@link LlamaModel#generate(InferenceParameters)}
 * and
 * {@link LlamaModel#complete(InferenceParameters)}.
 */
@SuppressWarnings("unused")
public final class InferenceParameters extends JsonParameters {

	private static final String PARAM_PROMPT = "prompt";
	private static final String PARAM_INPUT_PREFIX = "input_prefix";
	private static final String PARAM_INPUT_SUFFIX = "input_suffix";
	private static final String PARAM_CACHE_PROMPT = "cache_prompt";
	private static final String PARAM_N_PREDICT = "n_predict";
	private static final String PARAM_TOP_K = "top_k";
	private static final String PARAM_TOP_P = "top_p";
	private static final String PARAM_MIN_P = "min_p";
	private static final String PARAM_TFS_Z = "tfs_z";
	private static final String PARAM_TYPICAL_P = "typical_p";
	private static final String PARAM_TEMPERATURE = "temperature";
	private static final String PARAM_DYNATEMP_RANGE = "dynatemp_range";
	private static final String PARAM_DYNATEMP_EXPONENT = "dynatemp_exponent";
	private static final String PARAM_REPEAT_LAST_N = "repeat_last_n";
	private static final String PARAM_REPEAT_PENALTY = "repeat_penalty";
	private static final String PARAM_FREQUENCY_PENALTY = "frequency_penalty";
	private static final String PARAM_PRESENCE_PENALTY = "presence_penalty";
	private static final String PARAM_MIROSTAT = "mirostat";
	private static final String PARAM_MIROSTAT_TAU = "mirostat_tau";
	private static final String PARAM_MIROSTAT_ETA = "mirostat_eta";
	private static final String PARAM_PENALIZE_NL = "penalize_nl";
	private static final String PARAM_N_KEEP = "n_keep";
	private static final String PARAM_SEED = "seed";
	private static final String PARAM_N_PROBS = "n_probs";
	private static final String PARAM_MIN_KEEP = "min_keep";
	private static final String PARAM_GRAMMAR = "grammar";
	private static final String PARAM_PENALTY_PROMPT = "penalty_prompt";
	private static final String PARAM_IGNORE_EOS = "ignore_eos";
	private static final String PARAM_LOGIT_BIAS = "logit_bias";
	private static final String PARAM_STOP = "stop";
	private static final String PARAM_SAMPLERS = "samplers";
	private static final String PARAM_STREAM = "stream";
	private static final String PARAM_USE_CHAT_TEMPLATE = "use_chat_template";
	private static final String PARAM_CHAT_TEMPLATE = "chat_template";
	private static final String PARAM_USE_JINJA = "use_jinja";
	private static final String PARAM_CHAT_TEMPLATE_KWARGS = "chat_template_kwargs";
	private static final String PARAM_MESSAGES = "messages";

	public InferenceParameters(String prompt) {
		// we always need a prompt
		setPrompt(prompt);
	}

	/**
	 * Set the prompt to start generation with (default: empty)
	 *
	 * @param prompt the prompt to start generation with
	 * @return this builder
	 */
	public InferenceParameters setPrompt(String prompt) {
		parameters.put(PARAM_PROMPT, toJsonString(prompt));
		return this;
	}

	/**
	 * Set a prefix for infilling (default: empty)
	 *
	 * @param inputPrefix the prefix for infilling
	 * @return this builder
	 */
	public InferenceParameters setInputPrefix(String inputPrefix) {
		parameters.put(PARAM_INPUT_PREFIX, toJsonString(inputPrefix));
		return this;
	}

	/**
	 * Set a suffix for infilling (default: empty)
	 *
	 * @param inputSuffix the suffix for infilling
	 * @return this builder
	 */
	public InferenceParameters setInputSuffix(String inputSuffix) {
		parameters.put(PARAM_INPUT_SUFFIX, toJsonString(inputSuffix));
		return this;
	}

	/**
	 * Whether to remember the prompt to avoid reprocessing it
	 *
	 * @param cachePrompt whether to cache the prompt
	 * @return this builder
	 */
	public InferenceParameters setCachePrompt(boolean cachePrompt) {
		parameters.put(PARAM_CACHE_PROMPT, String.valueOf(cachePrompt));
		return this;
	}

	/**
	 * Set the number of tokens to predict (default: -1, -1 = infinity, -2 = until context filled)
	 *
	 * @param nPredict number of tokens to predict (-1 = infinity, -2 = until context filled)
	 * @return this builder
	 */
	public InferenceParameters setNPredict(int nPredict) {
		parameters.put(PARAM_N_PREDICT, String.valueOf(nPredict));
		return this;
	}

	/**
	 * Set top-k sampling (default: 40, 0 = disabled)
	 *
	 * @param topK the top-k value (0 = disabled)
	 * @return this builder
	 */
	public InferenceParameters setTopK(int topK) {
		parameters.put(PARAM_TOP_K, String.valueOf(topK));
		return this;
	}

	/**
	 * Set top-p sampling (default: 0.9, 1.0 = disabled)
	 *
	 * @param topP the top-p value (1.0 = disabled)
	 * @return this builder
	 */
	public InferenceParameters setTopP(float topP) {
		parameters.put(PARAM_TOP_P, String.valueOf(topP));
		return this;
	}

	/**
	 * Set min-p sampling (default: 0.1, 0.0 = disabled)
	 *
	 * @param minP the min-p value (0.0 = disabled)
	 * @return this builder
	 */
	public InferenceParameters setMinP(float minP) {
		parameters.put(PARAM_MIN_P, String.valueOf(minP));
		return this;
	}

	/**
	 * Set tail free sampling, parameter z (default: 1.0, 1.0 = disabled)
	 *
	 * @param tfsZ tail free sampling parameter z (1.0 = disabled)
	 * @return this builder
	 */
	public InferenceParameters setTfsZ(float tfsZ) {
		parameters.put(PARAM_TFS_Z, String.valueOf(tfsZ));
		return this;
	}

	/**
	 * Set locally typical sampling, parameter p (default: 1.0, 1.0 = disabled)
	 *
	 * @param typicalP the locally typical sampling parameter p (1.0 = disabled)
	 * @return this builder
	 */
	public InferenceParameters setTypicalP(float typicalP) {
		parameters.put(PARAM_TYPICAL_P, String.valueOf(typicalP));
		return this;
	}

	/**
	 * Set the temperature (default: 0.8)
	 *
	 * @param temperature the sampling temperature
	 * @return this builder
	 */
	public InferenceParameters setTemperature(float temperature) {
		parameters.put(PARAM_TEMPERATURE, String.valueOf(temperature));
		return this;
	}

	/**
	 * Set the dynamic temperature range (default: 0.0, 0.0 = disabled)
	 *
	 * @param dynatempRange the dynamic temperature range (0.0 = disabled)
	 * @return this builder
	 */
	public InferenceParameters setDynamicTemperatureRange(float dynatempRange) {
		parameters.put(PARAM_DYNATEMP_RANGE, String.valueOf(dynatempRange));
		return this;
	}

	/**
	 * Set the dynamic temperature exponent (default: 1.0)
	 *
	 * @param dynatempExponent the dynamic temperature exponent
	 * @return this builder
	 */
	public InferenceParameters setDynamicTemperatureExponent(float dynatempExponent) {
		parameters.put(PARAM_DYNATEMP_EXPONENT, String.valueOf(dynatempExponent));
		return this;
	}

	/**
	 * Set the last n tokens to consider for penalties (default: 64, 0 = disabled, -1 = ctx_size)
	 *
	 * @param repeatLastN the number of last tokens to consider for penalties (0 = disabled, -1 = ctx_size)
	 * @return this builder
	 */
	public InferenceParameters setRepeatLastN(int repeatLastN) {
		parameters.put(PARAM_REPEAT_LAST_N, String.valueOf(repeatLastN));
		return this;
	}

	/**
	 * Set the penalty of repeated sequences of tokens (default: 1.0, 1.0 = disabled)
	 *
	 * @param repeatPenalty the repeat penalty (1.0 = disabled)
	 * @return this builder
	 */
	public InferenceParameters setRepeatPenalty(float repeatPenalty) {
		parameters.put(PARAM_REPEAT_PENALTY, String.valueOf(repeatPenalty));
		return this;
	}

	/**
	 * Set the repetition alpha frequency penalty (default: 0.0, 0.0 = disabled)
	 *
	 * @param frequencyPenalty the repetition alpha frequency penalty (0.0 = disabled)
	 * @return this builder
	 */
	public InferenceParameters setFrequencyPenalty(float frequencyPenalty) {
		parameters.put(PARAM_FREQUENCY_PENALTY, String.valueOf(frequencyPenalty));
		return this;
	}

	/**
	 * Set the repetition alpha presence penalty (default: 0.0, 0.0 = disabled)
	 *
	 * @param presencePenalty the repetition alpha presence penalty (0.0 = disabled)
	 * @return this builder
	 */
	public InferenceParameters setPresencePenalty(float presencePenalty) {
		parameters.put(PARAM_PRESENCE_PENALTY, String.valueOf(presencePenalty));
		return this;
	}

	/**
	 * Set MiroStat sampling strategies.
	 *
	 * @param mirostat the MiroStat sampling strategy
	 * @return this builder
	 */
	public InferenceParameters setMiroStat(MiroStat mirostat) {
		parameters.put(PARAM_MIROSTAT, String.valueOf(mirostat.ordinal()));
		return this;
	}

	/**
	 * Set the MiroStat target entropy, parameter tau (default: 5.0)
	 *
	 * @param mirostatTau the MiroStat target entropy parameter tau
	 * @return this builder
	 */
	public InferenceParameters setMiroStatTau(float mirostatTau) {
		parameters.put(PARAM_MIROSTAT_TAU, String.valueOf(mirostatTau));
		return this;
	}

	/**
	 * Set the MiroStat learning rate, parameter eta (default: 0.1)
	 *
	 * @param mirostatEta the MiroStat learning rate parameter eta
	 * @return this builder
	 */
	public InferenceParameters setMiroStatEta(float mirostatEta) {
		parameters.put(PARAM_MIROSTAT_ETA, String.valueOf(mirostatEta));
		return this;
	}

	/**
	 * Whether to penalize newline tokens
	 *
	 * @param penalizeNl whether to penalize newline tokens
	 * @return this builder
	 */
	public InferenceParameters setPenalizeNl(boolean penalizeNl) {
		parameters.put(PARAM_PENALIZE_NL, String.valueOf(penalizeNl));
		return this;
	}

	/**
	 * Set the number of tokens to keep from the initial prompt (default: 0, -1 = all)
	 *
	 * @param nKeep the number of tokens to keep from the initial prompt (-1 = all)
	 * @return this builder
	 */
	public InferenceParameters setNKeep(int nKeep) {
		parameters.put(PARAM_N_KEEP, String.valueOf(nKeep));
		return this;
	}

	/**
	 * Set the RNG seed (default: -1, use random seed for &lt; 0)
	 *
	 * @param seed the RNG seed (use a negative value for a random seed)
	 * @return this builder
	 */
	public InferenceParameters setSeed(int seed) {
		parameters.put(PARAM_SEED, String.valueOf(seed));
		return this;
	}

	/**
	 * Set the amount top tokens probabilities to output if greater than 0.
	 *
	 * @param nProbs the number of top token probabilities to output
	 * @return this builder
	 */
	public InferenceParameters setNProbs(int nProbs) {
		parameters.put(PARAM_N_PROBS, String.valueOf(nProbs));
		return this;
	}

	/**
	 * Set the amount of tokens the samplers should return at least (0 = disabled)
	 *
	 * @param minKeep the minimum number of tokens samplers should return (0 = disabled)
	 * @return this builder
	 */
	public InferenceParameters setMinKeep(int minKeep) {
		parameters.put(PARAM_MIN_KEEP, String.valueOf(minKeep));
		return this;
	}

	/**
	 * Set BNF-like grammar to constrain generations (see samples in grammars/ dir)
	 *
	 * @param grammar the BNF-like grammar string
	 * @return this builder
	 */
	public InferenceParameters setGrammar(String grammar) {
		parameters.put(PARAM_GRAMMAR, toJsonString(grammar));
		return this;
	}

	/**
	 * Override which part of the prompt is penalized for repetition.
	 * E.g. if original prompt is "Alice: Hello!" and penaltyPrompt is "Hello!", only the latter will be penalized if
	 * repeated. See <a href="https://github.com/ggerganov/llama.cpp/pull/3727">pull request 3727</a> for more details.
	 *
	 * @param penaltyPrompt the string portion of the prompt to penalize for repetition
	 * @return this builder
	 */
	public InferenceParameters setPenaltyPrompt(String penaltyPrompt) {
		parameters.put(PARAM_PENALTY_PROMPT, toJsonString(penaltyPrompt));
		return this;
	}

	/**
	 * Override which tokens to penalize for repetition.
	 * E.g. if original prompt is "Alice: Hello!" and penaltyPrompt corresponds to the token ids of "Hello!", only the
	 * latter will be penalized if repeated.
	 * See <a href="https://github.com/ggerganov/llama.cpp/pull/3727">pull request 3727</a> for more details.
	 *
	 * @param tokens the token ids of the prompt portion to penalize for repetition
	 * @return this builder
	 */
	public InferenceParameters setPenaltyPrompt(int[] tokens) {
		if (tokens.length > 0) {
			StringBuilder builder = new StringBuilder();
			builder.append("[");
			for (int i = 0; i < tokens.length; i++) {
				builder.append(tokens[i]);
				if (i < tokens.length - 1) {
					builder.append(", ");
				}
			}
			builder.append("]");
			parameters.put(PARAM_PENALTY_PROMPT, builder.toString());
		}
		return this;
	}

	/**
	 * Set whether to ignore end of stream token and continue generating (implies --logit-bias 2-inf)
	 *
	 * @param ignoreEos whether to ignore the end-of-stream token
	 * @return this builder
	 */
	public InferenceParameters setIgnoreEos(boolean ignoreEos) {
		parameters.put(PARAM_IGNORE_EOS, String.valueOf(ignoreEos));
		return this;
	}

	/**
	 * Modify the likelihood of tokens appearing in the completion by their id. E.g., <code>Map.of(15043, 1f)</code>
	 * to increase the  likelihood of token ' Hello', or a negative value to decrease it.
	 * Note, this method overrides any previous calls to
	 * <ul>
	 *     <li>{@link #setTokenBias(Map)}</li>
	 *     <li>{@link #disableTokens(Collection)}</li>
	 *     <li>{@link #disableTokenIds(Collection)}}</li>
	 * </ul>
	 *
	 * @param logitBias a map from token id to bias value
	 * @return this builder
	 */
	public InferenceParameters setTokenIdBias(Map<Integer, Float> logitBias) {
		if (!logitBias.isEmpty()) {
			parameters.put(PARAM_LOGIT_BIAS, buildBiasPairArray(logitBias, String::valueOf));
		}
		return this;
	}

	/**
	 * Set tokens to disable, this corresponds to {@link #setTokenIdBias(Map)} with a value of
	 * {@link Float#NEGATIVE_INFINITY}.
	 * Note, this method overrides any previous calls to
	 * <ul>
	 *     <li>{@link #setTokenIdBias(Map)}</li>
	 *     <li>{@link #setTokenBias(Map)}</li>
	 *     <li>{@link #disableTokens(Collection)}</li>
	 * </ul>
	 *
	 * @param tokenIds the collection of token ids to disable
	 * @return this builder
	 */
	public InferenceParameters disableTokenIds(Collection<Integer> tokenIds) {
		if (!tokenIds.isEmpty()) {
			parameters.put(PARAM_LOGIT_BIAS, buildDisablePairArray(tokenIds, String::valueOf));
		}
		return this;
	}

	/**
	 * Modify the likelihood of tokens appearing in the completion by their id. E.g., <code>Map.of(" Hello", 1f)</code>
	 * to increase the  likelihood of token id 15043, or a negative value to decrease it.
	 * Note, this method overrides any previous calls to
	 * <ul>
	 *     <li>{@link #setTokenIdBias(Map)}</li>
	 *     <li>{@link #disableTokens(Collection)}</li>
	 *     <li>{@link #disableTokenIds(Collection)}}</li>
	 * </ul>
	 *
	 * @param logitBias a map from token string to bias value
	 * @return this builder
	 */
	public InferenceParameters setTokenBias(Map<String, Float> logitBias) {
		if (!logitBias.isEmpty()) {
			parameters.put(PARAM_LOGIT_BIAS, buildBiasPairArray(logitBias, this::toJsonString));
		}
		return this;
	}

	/**
	 * Set tokens to disable, this corresponds to {@link #setTokenBias(Map)} with a value of
	 * {@link Float#NEGATIVE_INFINITY}.
	 * Note, this method overrides any previous calls to
	 * <ul>
	 *     <li>{@link #setTokenBias(Map)}</li>
	 *     <li>{@link #setTokenIdBias(Map)}</li>
	 *     <li>{@link #disableTokenIds(Collection)}</li>
	 * </ul>
	 *
	 * @param tokens the collection of token strings to disable
	 * @return this builder
	 */
	public InferenceParameters disableTokens(Collection<String> tokens) {
		if (!tokens.isEmpty()) {
			parameters.put(PARAM_LOGIT_BIAS, buildDisablePairArray(tokens, this::toJsonString));
		}
		return this;
	}

	/**
	 * Set strings upon seeing which token generation is stopped
	 *
	 * @param stopStrings one or more strings that stop generation when encountered
	 * @return this builder
	 */
	public InferenceParameters setStopStrings(String... stopStrings) {
		if (stopStrings.length > 0) {
			StringBuilder builder = new StringBuilder();
			builder.append("[");
			for (int i = 0; i < stopStrings.length; i++) {
				builder.append(toJsonString(stopStrings[i]));
				if (i < stopStrings.length - 1) {
					builder.append(", ");
				}
			}
			builder.append("]");
			parameters.put(PARAM_STOP, builder.toString());
		}
		return this;
	}

	/**
	 * Set which samplers to use for token generation in the given order
	 *
	 * @param samplers the samplers to use for token generation, in order
	 * @return this builder
	 */
	public InferenceParameters setSamplers(Sampler... samplers) {
		if (samplers.length > 0) {
			StringBuilder builder = new StringBuilder();
			builder.append("[");
			for (int i = 0; i < samplers.length; i++) {
				switch (samplers[i]) {
					case TOP_K:
						builder.append("\"top_k\"");
						break;
					case TOP_P:
						builder.append("\"top_p\"");
						break;
					case MIN_P:
						builder.append("\"min_p\"");
						break;
					case TEMPERATURE:
						builder.append("\"temperature\"");
						break;
				}
				if (i < samplers.length - 1) {
					builder.append(", ");
				}
			}
			builder.append("]");
			parameters.put(PARAM_SAMPLERS, builder.toString());
		}
		return this;
	}

	/**
	 * Set whether generate should apply a chat template (default: false)
	 *
	 * @param useChatTemplate whether to apply a chat template
	 * @return this builder
	 */
	public InferenceParameters setUseChatTemplate(boolean useChatTemplate) {
		parameters.put(PARAM_USE_JINJA, String.valueOf(useChatTemplate));
		return this;
	}

	/**
	 * Set the chat template string.
	 *
	 * @param chatTemplate the Jinja-style chat template to use
	 * @return this builder
	 */
	public InferenceParameters setChatTemplate(String chatTemplate) {
		parameters.put(PARAM_CHAT_TEMPLATE, toJsonString(chatTemplate));
		return this;
	}

	/**
	 * Set custom Jinja template variables for this request. These are injected into
	 * the chat template context during rendering. Values must be valid JSON.
	 * <p>
	 * Example:
	 * <pre>{@code
	 * Map<String, String> kwargs = new HashMap<>();
	 * kwargs.put("enable_thinking", "true");
	 * params.setChatTemplateKwargs(kwargs);
	 * }</pre>
	 *
	 * @param kwargs map of variable names to JSON-serialized values
	 * @return this builder
	 */
	public InferenceParameters setChatTemplateKwargs(java.util.Map<String, String> kwargs) {
		parameters.put(PARAM_CHAT_TEMPLATE_KWARGS, mapToJsonObject(kwargs));
		return this;
	}

	/**
     * Set the messages for chat-based inference.
     * - Allows <b>only one</b> system message.
     * - Allows <b>one or more</b> user/assistant messages.
     *
     * @param systemMessage an optional system message (may be null or empty)
     * @param messages a list of user/assistant message pairs (role as key, content as value)
     * @return this builder
     */
    public InferenceParameters setMessages(String systemMessage, List<Pair<String, String>> messages) {
		StringBuilder messagesBuilder = new StringBuilder();
		messagesBuilder.append("[");

        // Add system message (if provided)
        if (systemMessage != null && !systemMessage.isEmpty()) {
			messagesBuilder.append("{\"role\": \"system\", \"content\": ")
					.append(toJsonString(systemMessage))
					.append("}");
			if (!messages.isEmpty()) {
				messagesBuilder.append(", ");
			}
        }

        // Add user/assistant messages
        for (int i = 0; i < messages.size(); i++) {
            Pair<String, String> message = messages.get(i);
            String role = message.getKey();
            String content = message.getValue();

            if (!role.equals("user") && !role.equals("assistant")) {
                throw new IllegalArgumentException("Invalid role: " + role + ". Role must be 'user' or 'assistant'.");
            }

			messagesBuilder.append("{\"role\":")
					.append(toJsonString(role))
					.append(", \"content\": ")
					.append(toJsonString(content))
					.append("}");

			if (i < messages.size() - 1) {
				messagesBuilder.append(", ");
			}
        }

		messagesBuilder.append("]");

        // Convert ArrayNode to a JSON string and store it in parameters
        parameters.put(PARAM_MESSAGES, messagesBuilder.toString());
        return this;
    }

	InferenceParameters setStream(boolean stream) {
		parameters.put(PARAM_STREAM, String.valueOf(stream));
		return this;
	}

	private static <K, V> String buildBiasPairArray(Map<K, V> map,
			java.util.function.Function<K, String> keySerializer) {
		StringBuilder builder = new StringBuilder("[");
		int i = 0;
		for (Map.Entry<K, V> entry : map.entrySet()) {
			builder.append("[")
					.append(keySerializer.apply(entry.getKey()))
					.append(", ")
					.append(entry.getValue())
					.append("]");
			if (i++ < map.size() - 1) {
				builder.append(", ");
			}
		}
		builder.append("]");
		return builder.toString();
	}

	private static <T> String buildDisablePairArray(Collection<T> items,
			java.util.function.Function<T, String> serializer) {
		StringBuilder builder = new StringBuilder("[");
		int i = 0;
		for (T item : items) {
			builder.append("[")
					.append(serializer.apply(item))
					.append(", false]");
			if (i++ < items.size() - 1) {
				builder.append(", ");
			}
		}
		builder.append("]");
		return builder.toString();
	}

}
