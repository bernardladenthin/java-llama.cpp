// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.json;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.is;

import java.util.List;
import net.ladenthin.llama.ClaudeGenerated;
import net.ladenthin.llama.value.RouterModel;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose = "Verify the router GET /models wire-format parser against the upstream "
                + "get_router_models JSON shape: data/models array fallback, id/name fallback, "
                + "status.value mapping, the failed/exit_code marker, and tolerance for "
                + "missing-status and unparseable input.")
public class RouterModelsResponseParserTest {

    private final RouterModelsResponseParser parser = new RouterModelsResponseParser();

    @Test
    public void parsesUpstreamShape() {
        String json = "{\"object\":\"list\",\"data\":["
                + "{\"id\":\"qwen\",\"status\":{\"value\":\"loaded\",\"args\":[\"-c\",\"512\"]},\"source\":\"models_dir\"},"
                + "{\"id\":\"llama\",\"status\":{\"value\":\"unloaded\"}}"
                + "]}";

        List<RouterModel> models = parser.parse(json);

        assertThat(models.size(), is(2));
        assertThat(models.get(0).getId(), is("qwen"));
        assertThat(models.get(0).getStatus(), is(RouterModel.Status.LOADED));
        assertThat(models.get(0).getStatusValue(), is("loaded"));
        assertThat(models.get(0).isFailed(), is(false));
        assertThat(models.get(1).getId(), is("llama"));
        assertThat(models.get(1).getStatus(), is(RouterModel.Status.UNLOADED));
    }

    @Test
    public void parsesFailedWorkerMarker() {
        String json = "{\"data\":[{\"id\":\"broken\","
                + "\"status\":{\"value\":\"unloaded\",\"failed\":true,\"exit_code\":1}}]}";

        RouterModel model = parser.parse(json).get(0);

        assertThat(model.isFailed(), is(true));
        assertThat(model.getExitCode(), is(1));
        assertThat(model.getStatus(), is(RouterModel.Status.UNLOADED));
    }

    @Test
    public void fallsBackToModelsArrayAndNameField() {
        // Alternate shape tolerance: "models" array with "name" identifiers.
        String json = "{\"models\":[{\"name\":\"by-name\",\"status\":{\"value\":\"loading\"}}]}";

        List<RouterModel> models = parser.parse(json);

        assertThat(models.size(), is(1));
        assertThat(models.get(0).getId(), is("by-name"));
        assertThat(models.get(0).getStatus(), is(RouterModel.Status.LOADING));
    }

    @Test
    public void missingStatusMapsToUnknownWithEmptyRawValue() {
        String json = "{\"data\":[{\"id\":\"bare\"}]}";

        RouterModel model = parser.parse(json).get(0);

        assertThat(model.getStatus(), is(RouterModel.Status.UNKNOWN));
        assertThat(model.getStatusValue(), is(""));
        assertThat(model.isFailed(), is(false));
        assertThat(model.getExitCode(), is(0));
    }

    @Test
    public void unrecognizedStatusKeepsRawValue() {
        String json = "{\"data\":[{\"id\":\"m\",\"status\":{\"value\":\"hibernating\"}}]}";

        RouterModel model = parser.parse(json).get(0);

        assertThat(model.getStatus(), is(RouterModel.Status.UNKNOWN));
        assertThat(model.getStatusValue(), is("hibernating"));
    }

    @Test
    public void emptyDataYieldsEmptyList() {
        assertThat(parser.parse("{\"data\":[],\"object\":\"list\"}").isEmpty(), is(true));
    }

    @Test
    public void missingArraysYieldEmptyList() {
        assertThat(parser.parse("{\"object\":\"list\"}").isEmpty(), is(true));
    }

    @Test
    public void unparseableInputYieldsEmptyList() {
        assertThat(parser.parse("not json").isEmpty(), is(true));
    }

    @Test
    public void parsesTheArchitectureModalities() {
        // GET /models since llama.cpp b11429 (#29987): computed offline, present before the first load.
        String json = "{\"data\":[{\"id\":\"decider\",\"status\":{\"value\":\"unloaded\"},"
                + "\"architecture\":{\"input_modalities\":[\"text\",\"image\"],"
                + "\"output_modalities\":[\"decisions\"]}}]}";

        RouterModel model = parser.parse(json).get(0);
        assertThat(model.getInputModalities(), contains("text", "image"));
        assertThat(model.getOutputModalities(), contains("decisions"));
        assertThat(model.isDecisionModel(), is(true));
    }

    @Test
    public void missingArchitectureYieldsNoModalities() {
        // A server before b11429 omits the object; upstream asks clients to fall back, not to guess.
        RouterModel model = parser.parse("{\"data\":[{\"id\":\"old\"}]}").get(0);
        assertThat(model.getInputModalities(), is(empty()));
        assertThat(model.getOutputModalities(), is(empty()));
        assertThat(model.isDecisionModel(), is(false));
    }
}
