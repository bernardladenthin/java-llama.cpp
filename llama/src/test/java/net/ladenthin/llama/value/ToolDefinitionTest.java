// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.value;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;

import net.ladenthin.llama.ClaudeGenerated;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose = "Pin ToolDefinition's name/description/parametersSchemaJson accessors to distinct non-empty "
                + "values plus its Lombok toString/equals so every getter mutation is covered.")
public class ToolDefinitionTest {

    private static final String SCHEMA = "{\"type\":\"object\",\"properties\":{\"s\":{\"type\":\"string\"}}}";

    @Test
    public void accessorsReturnConstructorValues() {
        ToolDefinition d = new ToolDefinition("echo", "Echo a string", SCHEMA);
        assertThat(d.getName(), is("echo"));
        // A distinct non-empty value kills the empty-string return mutant on getDescription.
        assertThat(d.getDescription(), is("Echo a string"));
        assertThat(d.getParametersSchemaJson(), is(SCHEMA));
    }

    @Test
    public void aToolIsNotDeferredUnlessAsked() {
        assertThat(new ToolDefinition("echo", "d", SCHEMA).isDeferLoading(), is(false));
        assertThat(new ToolDefinition("echo", "d", SCHEMA, false).isDeferLoading(), is(false));
        assertThat(new ToolDefinition("echo", "d", SCHEMA, true).isDeferLoading(), is(true));
    }

    @Test
    public void deferLoadingTakesPartInEquality() {
        assertThat(new ToolDefinition("echo", "d", "{}", true), is(new ToolDefinition("echo", "d", "{}", true)));
        assertThat(
                new ToolDefinition("echo", "d", "{}", true).equals(new ToolDefinition("echo", "d", "{}")), is(false));
    }

    @Test
    public void toStringRendersAllFields() {
        ToolDefinition d = new ToolDefinition("echo", "Echo a string", SCHEMA);
        String s = d.toString();
        assertThat(s, containsString("echo"));
        assertThat(s, containsString("Echo a string"));
    }

    @Test
    public void equalsAndHashCodeAreValueBased() {
        ToolDefinition a = new ToolDefinition("echo", "d", "{}");
        ToolDefinition b = new ToolDefinition("echo", "d", "{}");
        assertThat(a, is(b));
        assertThat(a.hashCode(), is(b.hashCode()));
    }

    @Test
    public void differingDescriptionBreaksEquality() {
        assertThat(new ToolDefinition("echo", "d1", "{}"), is(not(new ToolDefinition("echo", "d2", "{}"))));
    }
}
