// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.nullValue;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import org.junit.jupiter.api.Test;

class AgentOptionsTest {

    @Test
    void baseUrlModeWithDefaults() {
        AgentOptions options = AgentOptions.parse(new String[] {"--base-url", "http://127.0.0.1:8080/v1/"});

        assertThat(options.getBaseUrl(), is("http://127.0.0.1:8080/v1"));
        assertThat(options.getModelPath(), is(nullValue()));
        assertThat(options.getApiKey(), is(AgentOptions.DEFAULT_API_KEY));
        assertThat(options.getModelId(), is(AgentOptions.DEFAULT_MODEL_ID));
        assertThat(options.getWorkspace(), is(Paths.get("").toAbsolutePath().normalize()));
        assertThat(options.isAllowShell(), is(false));
        assertThat(options.getTemperature(), is(AgentOptions.DEFAULT_TEMPERATURE));
        assertThat(options.getMaxTokens(), is(AgentOptions.DEFAULT_MAX_TOKENS));
        assertThat(options.getMaxToolRounds(), is(AgentOptions.DEFAULT_MAX_TOOL_ROUNDS));
        assertThat(options.getLogVerbosity(), is(AgentOptions.DEFAULT_LOG_VERBOSITY));
        assertThat(options.isVerbose(), is(false));
        assertThat(options.getPrompt(), is(nullValue()));
        assertThat(options.isHelp(), is(false));
    }

    @Test
    void logVerbosityIsAnIntegerThreshold() {
        AgentOptions options = AgentOptions.parse(new String[] {"--model", "m.gguf", "--log-verbosity", "4"});

        assertThat(options.getLogVerbosity(), is(4));
        assertThat(options.isVerbose(), is(false));
        assertThat(
                assertThrows(
                                IllegalArgumentException.class,
                                () -> AgentOptions.parse(new String[] {"--model", "m.gguf", "--log-verbosity", "loud"}))
                        .getMessage(),
                containsString("--log-verbosity"));
    }

    @Test
    void verboseIsAFlagWithAShortForm() {
        assertThat(
                AgentOptions.parse(new String[] {"--model", "m.gguf", "--verbose"})
                        .isVerbose(),
                is(true));
        assertThat(AgentOptions.parse(new String[] {"--model", "m.gguf", "-v"}).isVerbose(), is(true));
        assertThat(AgentOptions.usage(), containsString("--log-verbosity"));
        assertThat(AgentOptions.usage(), containsString("--verbose"));
    }

    @Test
    void inProcessModeParsesEveryOption() {
        AgentOptions options = AgentOptions.parse(new String[] {
            "--model",
            "m.gguf",
            "--ngl",
            "99",
            "--ctx-size",
            "4096",
            "--api-key",
            "k",
            "--model-id",
            "id",
            "--workspace",
            "/tmp/ws",
            "--allow-shell",
            "--temperature",
            "0.5",
            "--max-tokens",
            "10",
            "--max-tool-rounds",
            "3",
            "--system",
            "sys",
            "-p",
            "do it"
        });

        assertThat(options.getBaseUrl(), is(nullValue()));
        assertThat(options.getModelPath(), is("m.gguf"));
        assertThat(options.getGpuLayers(), is(99));
        assertThat(options.getCtxSize(), is(4096));
        assertThat(options.getApiKey(), is("k"));
        assertThat(options.getModelId(), is("id"));
        assertThat(
                options.getWorkspace(), is(Paths.get("/tmp/ws").toAbsolutePath().normalize()));
        assertThat(options.isAllowShell(), is(true));
        assertThat(options.getTemperature(), is(0.5));
        assertThat(options.getMaxTokens(), is(10));
        assertThat(options.getMaxToolRounds(), is(3));
        assertThat(options.getSystemPrompt(), is("sys"));
        assertThat(options.getPrompt(), is("do it"));
    }

    @Test
    void exactlyOneEndpointIsRequired() {
        IllegalArgumentException none =
                assertThrows(IllegalArgumentException.class, () -> AgentOptions.parse(new String[0]));
        assertThat(none.getMessage(), containsString("Exactly one of --base-url"));
        assertThrows(
                IllegalArgumentException.class,
                () -> AgentOptions.parse(new String[] {"--base-url", "http://x/v1", "--model", "m.gguf"}));
    }

    @Test
    void helpNeedsNoEndpoint() {
        assertThat(AgentOptions.parse(new String[] {"--help"}).isHelp(), is(true));
        assertThat(AgentOptions.usage(), containsString("--base-url"));
        assertThat(AgentOptions.usage(), containsString("--allow-shell"));
    }

    @Test
    void unknownFlagAndMissingValueAreRejected() {
        assertThat(
                assertThrows(IllegalArgumentException.class, () -> AgentOptions.parse(new String[] {"--bogus"}))
                        .getMessage(),
                containsString("Unknown argument: --bogus"));
        assertThat(
                assertThrows(IllegalArgumentException.class, () -> AgentOptions.parse(new String[] {"--base-url"}))
                        .getMessage(),
                containsString("Missing value for --base-url"));
        assertThat(
                assertThrows(
                                IllegalArgumentException.class,
                                () -> AgentOptions.parse(new String[] {"--base-url", "u", "--ngl", "many"}))
                        .getMessage(),
                containsString("Expected an integer for --ngl"));
    }

    @Test
    void systemPromptMentionsTheShellToolOnlyWhenEnabled() {
        AgentOptions plain = AgentOptions.parse(new String[] {"--base-url", "http://x/v1"});
        AgentOptions shell = AgentOptions.parse(new String[] {"--base-url", "http://x/v1", "--allow-shell"});

        assertThat(LocalAgent.systemPrompt(plain).contains("run_command"), is(false));
        assertThat(LocalAgent.systemPrompt(shell), containsString("run_command"));
    }

    @Test
    void defaultSystemPromptIsGeneralPurposeAndAllowsAnyCommandWithTheShell() {
        // A narrow "coding agent ... build, test or inspect the project" framing made a 4B model refuse
        // "list the docker images" although run_command could run it; the prompt must grant it outright.
        String shell = LocalAgent.systemPrompt(
                AgentOptions.parse(new String[] {"--base-url", "http://x/v1", "--allow-shell"}));
        assertThat(shell, containsString("general-purpose"));
        assertThat(shell, containsString("any command line through " + ShellTool.shellName()));
        assertThat(shell, containsString("run the command instead of explaining"));
        assertThat(shell.contains("coding agent"), is(false));

        // Without the shell the model must not invent a limitation: it is told why and how to lift it.
        String plain = LocalAgent.systemPrompt(AgentOptions.parse(new String[] {"--base-url", "http://x/v1"}));
        assertThat(plain, containsString("--allow-shell"));
    }

    @Test
    void shellToolDescriptionDoesNotNarrowItToTheProject() {
        String description =
                ShellTool.definition(Path.of("."), Duration.ofSeconds(1), 100).description();
        assertThat(description, containsString("any command line"));
        assertThat(description, containsString(ShellTool.shellName()));
    }

    @Test
    void systemPromptOverrideReplacesTheDefault() {
        assertThat(
                LocalAgent.systemPrompt(AgentOptions.parse(new String[] {"--base-url", "u", "--system", "custom"})),
                is("custom"));
    }
}
