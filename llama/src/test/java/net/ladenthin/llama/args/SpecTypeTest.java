// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

import java.util.Arrays;
import java.util.Collection;

public class SpecTypeTest extends AbstractCliArgEnumTest<SpecType> {

    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][] {
            {SpecType.NONE, "none", 13},
            {SpecType.DRAFT_SIMPLE, "draft-simple", 13},
            {SpecType.DRAFT_EAGLE3, "draft-eagle3", 13},
            {SpecType.DRAFT_MTP, "draft-mtp", 13},
            {SpecType.DRAFT_DFLASH, "draft-dflash", 13},
            {SpecType.DRAFT_DSPARK, "draft-dspark", 13},
            {SpecType.DRAFT_DFLASH2, "draft-dflash2", 13},
            {SpecType.DRAFT_LILICORR, "draft-lilicorr", 13},
            {SpecType.NGRAM_SIMPLE, "ngram-simple", 13},
            {SpecType.NGRAM_MAP_K, "ngram-map-k", 13},
            {SpecType.NGRAM_MAP_K4V, "ngram-map-k4v", 13},
            {SpecType.NGRAM_MOD, "ngram-mod", 13},
            {SpecType.NGRAM_CACHE, "ngram-cache", 13},
        });
    }
}
