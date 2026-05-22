// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import net.ladenthin.llama.args.CliArg;
import org.jetbrains.annotations.Nullable;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

abstract class CliParameters {

    final Map<String, @Nullable String> parameters = new HashMap<>();

    /**
     * Store a scalar value (typically a primitive: int, long, float, double, boolean)
     * for the given key using {@link String#valueOf(Object)} and return this builder
     * typed as the concrete subtype so callers can collapse the
     * {@code parameters.put(...); return this;} pair into a single
     * {@code return putScalar(...);}.
     *
     * @param key   the parameter key
     * @param value the scalar value; autoboxed at the call site
     * @param <T>   the concrete subtype of this builder
     * @return this builder
     */
    @SuppressWarnings("unchecked")
    protected final <T extends CliParameters> T putScalar(String key, Object value) {
        parameters.put(key, String.valueOf(value));
        return (T) this;
    }

    /**
     * Store the CLI-argument string of the given enum constant for the given key and
     * return this builder typed as the concrete subtype.
     *
     * @param key   the parameter key
     * @param value the enum constant; must implement {@link CliArg}
     * @param <T>   the concrete subtype of this builder
     * @return this builder
     */
    @SuppressWarnings("unchecked")
    protected final <T extends CliParameters> T putEnum(String key, CliArg value) {
        parameters.put(key, value.getArgValue());
        return (T) this;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        for (String key : parameters.keySet()) {
            String value = parameters.get(key);
            builder.append(key).append(" ");
            if (value != null) {
                builder.append(value).append(" ");
            }
        }
        return builder.toString();
    }

    public String[] toArray() {
        List<String> result = new ArrayList<>();
        result.add(""); // c args contain the program name as the first argument, so we add an empty entry
        for (String key : parameters.keySet()) {
            result.add(key);
            String value = parameters.get(key);
            if (value != null) {
                result.add(value);
            }
        }
        return result.toArray(new String[0]);
    }

}
