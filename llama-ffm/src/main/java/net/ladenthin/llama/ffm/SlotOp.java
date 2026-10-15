// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
package net.ladenthin.llama.ffm;

/** A slot operation of {@link FfmLlamaModel#slotAction(SlotOp, int, String)}: {@code jllama_slot_op}. */
public enum SlotOp {
    /** Saves a slot's KV cache to a file under {@code --slot-save-path}. */
    SAVE(1),
    /** Restores a slot's KV cache from such a file. */
    RESTORE(2),
    /** Erases a slot's KV cache. */
    ERASE(3);

    private final int code;

    SlotOp(int code) {
        this.code = code;
    }

    /**
     * The {@code jllama_slot_op} value.
     *
     * @return the C enum value
     */
    public int code() {
        return code;
    }
}
