package net.ladenthin.llama.args;

/**
 * Implemented by every enum in this package that maps to a CLI argument value.
 *
 * <p>The contract: {@link #getArgValue()} returns the exact string accepted by the
 * corresponding llama.cpp CLI argument (e.g. {@code "q8_0"} for {@code --cache-type-k q8_0}).
 * Callers pass this string directly to {@code parameters.put("--flag", arg.getArgValue())}
 * without any post-processing.
 */
public interface CliArg {

    /**
     * Returns the CLI argument value string for this constant.
     *
     * @return the value string accepted by the corresponding llama.cpp CLI argument
     */
    String getArgValue();
}
