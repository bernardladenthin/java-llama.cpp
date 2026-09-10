// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.parameters;

/**
 * What a request key in {@link RequestField} is contractually required to satisfy.
 *
 * <p>A request body is the quieter of the two wire surfaces: llama.cpp's schema silently discards a
 * key it does not know, so a field that stops being read produces no error anywhere — it simply
 * stops having an effect. Declaring the contract on the constant is what makes that checkable.
 */
enum RequestContract {

    /**
     * The key must appear in llama.cpp's {@code server_schema::make_llama_cmpl_schema(...)} field
     * table (primary name or alias).
     */
    SCHEMA,

    /**
     * The key is consumed before the schema runs — by {@code oaicompat_*_params_parse} or by the
     * task layer in {@code server-context.cpp} — so it is legitimately absent from the schema table
     * and must not be checked against it.
     */
    OAI_LAYER
}
