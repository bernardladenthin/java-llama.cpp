// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.io.PrintStream;

/**
 * {@code --web}: the agent in a browser, for as long as the process runs.
 *
 * <p>One conversation for the whole process, whoever connects: two tabs, or a laptop and a phone through
 * the same tunnel, drive the same session. A message sent while another is running replaces it, as typing
 * does on the console.
 */
final class WebAgent {

    private WebAgent() {}

    /**
     * Serve the agent until the process is stopped.
     *
     * @param options the parsed command line
     * @param err where the address and diagnostics go
     * @return the exit code
     * @throws Exception when the model cannot be reached or the port cannot be bound
     */
    static int run(AgentOptions options, PrintStream err) throws Exception {
        try (ModelEndpoint endpoint = ModelEndpoint.open(options, err)) {
            AgentSession session = AgentSession.open(options, endpoint.baseUrl(), endpoint.contextSize());
            err.println("Endpoint " + endpoint.baseUrl() + " models=" + session.models() + " workspace="
                    + options.getWorkspace() + " tools=" + session.toolNames());
            if (!WebConsole.available()) {
                err.println("The console pages are missing from the classpath (atmosphere-spring-boot-starter);"
                        + " the endpoint runs, but there is nothing to open in a browser.");
            }
            try (WebServer server = WebServer.start(
                    session,
                    options.getWebHost(),
                    options.getWebPort(),
                    options.getWebToken(),
                    options.getModelId() + " · " + options.getWorkspace())) {
                for (String line : banner(server, options)) {
                    err.println(line);
                }
                server.join();
            }
        }
        return 0;
    }

    /**
     * What is printed once the server runs: where to go, and how to get there from elsewhere.
     *
     * @param server the running server
     * @param options the parsed command line
     * @return the lines
     */
    static java.util.List<String> banner(WebServer server, AgentOptions options) {
        java.util.List<String> lines = new java.util.ArrayList<>();
        lines.add("Open in a browser: " + server.url());
        if (WebAccessGuard.isLoopback(options.getWebHost())) {
            lines.add("From another machine, tunnel first: ssh -L " + server.port() + ":127.0.0.1:" + server.port()
                    + " <user>@<this-host> (PuTTY: Connection > SSH > Tunnels), then open the same address there.");
        } else {
            lines.add("WARNING: listening on " + options.getWebHost() + ", not loopback. The token is all that stands"
                    + " between the network and a shell on this machine; there is no TLS.");
        }
        lines.add("Approval mode: " + session(options)
                + " — /mode auto in the browser to stop asking. Ctrl-C here stops" + " the agent.");
        return lines;
    }

    private static String session(AgentOptions options) {
        return (options.isAuto() ? ApprovalMode.AUTO : ApprovalMode.MANUAL).badge();
    }
}
