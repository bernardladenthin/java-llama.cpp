// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

/**
 * What a CLI name in {@link ModelFlag} / {@link ModelOption} is contractually required to satisfy.
 *
 * <p>The kind lives on the constant rather than in a list inside the checking test, so it cannot
 * rot: a name can only exist by declaring which contract it belongs to, and the generated header
 * that {@code src/test/cpp/test_model_flags.cpp} reads carries the declaration along with the name.
 */
public enum CliContract {

    /**
     * The name must be registered in llama.cpp's
     * {@code common_params_parser_init(params, LLAMA_EXAMPLE_SERVER).options} table.
     *
     * <p>That parser treats an unregistered option as a hard error rather than a warning, so a name
     * that falls out of the table does not merely stop working — every caller of the matching
     * builder method gets {@code "Failed to parse model parameters"} instead of a loaded model.
     * Note the table is what has to be consulted, not the source: an option can be present in
     * {@code common/arg.cpp} yet {@code set_examples()}-scoped away from the server example, in
     * which case the server parser rejects it exactly like a deleted one.
     */
    SERVER_PARSER,

    /**
     * The name is this project's own and never reaches llama.cpp's parser: {@code jllama.cpp}
     * removes it from argv (see {@code strip_flag_from_argv}) before {@code common_params_parse}
     * runs. Today that is {@code --vocab-only} alone.
     */
    PROJECT_PSEUDO
}
