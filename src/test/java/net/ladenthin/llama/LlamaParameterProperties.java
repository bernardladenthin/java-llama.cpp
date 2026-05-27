// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
package net.ladenthin.llama;

import net.jqwik.api.ForAll;
import net.jqwik.api.Property;
import net.jqwik.api.constraints.FloatRange;

public class LlamaParameterProperties {

    @Property
    boolean setTemperatureNeverThrows(@ForAll @FloatRange(min = 0.0f, max = 2.0f) float temperature) {
        String json = new InferenceParameters("").setTemperature(temperature).toString();
        return json.contains("temperature");
    }

    @Property
    boolean setTopPNeverThrows(@ForAll @FloatRange(min = 0.0f, max = 1.0f) float topP) {
        String json = new InferenceParameters("").setTopP(topP).toString();
        return json.contains("top_p");
    }
}
