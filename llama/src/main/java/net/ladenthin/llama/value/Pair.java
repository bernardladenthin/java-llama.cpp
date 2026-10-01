// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.value;

import lombok.EqualsAndHashCode;
import lombok.ToString;

/**
 * A generic immutable key-value pair.
 *
 * @param <K> the key type
 * @param <V> the value type
 */
@ToString
@EqualsAndHashCode
public class Pair<K, V> {

    private final K key;
    private final V value;

    /**
     * Creates a new immutable pair of the given key and value.
     *
     * @param key   the key
     * @param value the value
     */
    public Pair(K key, V value) {
        this.key = key;
        this.value = value;
    }

    /**
     * Returns the key of this pair.
     *
     * @return the key
     */
    public K getKey() {
        return key;
    }

    /**
     * Returns the value of this pair.
     *
     * @return the value
     */
    public V getValue() {
        return value;
    }
}
