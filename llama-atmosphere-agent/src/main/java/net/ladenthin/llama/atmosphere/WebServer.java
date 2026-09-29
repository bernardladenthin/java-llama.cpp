// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import jakarta.servlet.DispatcherType;
import jakarta.servlet.ServletConfig;
import jakarta.servlet.ServletException;
import java.lang.annotation.Annotation;
import java.security.SecureRandom;
import java.util.Base64;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import org.atmosphere.ai.annotation.AiEndpoint;
import org.atmosphere.ai.processor.AiEndpointProcessor;
import org.atmosphere.config.AtmosphereAnnotation;
import org.atmosphere.cpr.ApplicationConfig;
import org.atmosphere.cpr.AtmosphereServlet;
import org.atmosphere.cpr.DefaultAnnotationProcessor;
import org.eclipse.jetty.ee10.servlet.FilterHolder;
import org.eclipse.jetty.ee10.servlet.ServletContextHandler;
import org.eclipse.jetty.ee10.servlet.ServletHolder;
import org.eclipse.jetty.ee10.websocket.jakarta.server.config.JakartaWebSocketServletContainerInitializer;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.server.ServerConnector;

/**
 * The browser front end: an embedded Jetty 12 serving Atmosphere's AI Console and the agent's endpoint.
 *
 * <p>No Spring and no classpath scanning. Atmosphere is given its annotation map explicitly — the one
 * processor and the one endpoint class — which is also what keeps this working inside a fat jar, where a
 * scan finds nothing reliable. The WebSocket container is configured before the Atmosphere servlet starts;
 * in the other order Atmosphere finds no {@code ServerContainer} and upgrades fail (Atmosphere issue #2510).
 *
 * <p>The server binds loopback by default. Reaching it from another machine is meant to go through an SSH
 * tunnel ({@code ssh -L 8787:127.0.0.1:8787 host}, or PuTTY's Connection → SSH → Tunnels), which keeps the
 * port closed to the network. Everything behind it is guarded by {@link WebAccessGuard}.
 */
public final class WebServer implements AutoCloseable {

    /** The path of the agent's endpoint. */
    static final String AGENT_PATH = "/atmosphere/agent";

    private final Server server;
    private final String host;
    private final int port;
    private final String token;

    private WebServer(Server server, String host, int port, String token) {
        this.server = server;
        this.host = host;
        this.port = port;
        this.token = token;
    }

    /**
     * Serve {@code session} in a browser.
     *
     * @param session the conversation
     * @param host the address to bind, normally loopback
     * @param port the port, {@code 0} for any free one
     * @param token the access token, or {@code null} for a new random one
     * @param subtitle the line under the console's title
     * @return the running server
     * @throws Exception when the port cannot be bound or Jetty does not start
     * @throws IllegalStateException when a browser front end is already running in this JVM
     */
    public static WebServer start(
            AgentSession session,
            String host,
            int port,
            @org.jspecify.annotations.Nullable String token,
            String subtitle)
            throws Exception {
        if (!WebAgentEndpoint.SESSION.compareAndSet(null, session)) {
            throw new IllegalStateException("a browser front end is already running in this JVM");
        }
        String access = token != null ? token : newToken();
        Server server = new Server();
        try {
            ServerConnector connector = new ServerConnector(server);
            connector.setHost(host);
            connector.setPort(port);
            server.addConnector(connector);

            ServletContextHandler context = new ServletContextHandler(ServletContextHandler.NO_SESSIONS);
            context.setContextPath("/");
            JakartaWebSocketServletContainerInitializer.configure(context, null);

            ServletHolder atmosphere = new ServletHolder("atmosphere", new ExplicitAtmosphereServlet());
            atmosphere.setAsyncSupported(true);
            atmosphere.setInitOrder(0);
            atmosphere.setInitParameter(ApplicationConfig.DISABLE_ATMOSPHERE_INITIALIZER, "true");
            atmosphere.setInitParameter(ApplicationConfig.PROPERTY_SESSION_SUPPORT, "false");
            atmosphere.setInitParameter(ApplicationConfig.WEBSOCKET_SUPPORT, "true");
            // The explicit annotation map is only read when Atmosphere scans a package, and it scans the
            // whole classpath only when JUnit is NOT on it (ClasspathScanner.preventOOM) -- so without these
            // two lines the endpoint registers in production and silently not in a test. "all" is the value
            // that makes the map processor skip its resource lookup (which a fat jar may not satisfy), and
            // switching the classpath scan off makes production take the very same path as the tests.
            atmosphere.setInitParameter(ApplicationConfig.ANNOTATION_PACKAGE, "all");
            atmosphere.setInitParameter(ApplicationConfig.SCAN_CLASSPATH, "false");
            context.addServlet(atmosphere, "/atmosphere/*");
            context.addFilter(
                    new FilterHolder(new WebConsole.Files()),
                    WebConsole.PATH + "/*",
                    EnumSet.of(DispatcherType.REQUEST));
            context.addServlet(new ServletHolder(new WebConsole.Info(subtitle)), WebConsole.Info.PATH);

            WebAccessGuard guard = new WebAccessGuard(access, host, WebConsole.PATH + "/");
            guard.setHandler(context);
            server.setHandler(guard);
            server.start();
            return new WebServer(server, host, connector.getLocalPort(), access);
        } catch (Exception | Error e) {
            WebAgentEndpoint.SESSION.set(null);
            server.stop();
            throw e;
        }
    }

    /**
     * A fresh random token: 24 bytes, URL-safe.
     *
     * @return the token
     */
    static String newToken() {
        byte[] random = new byte[24];
        new SecureRandom().nextBytes(random);
        return Base64.getUrlEncoder().withoutPadding().encodeToString(random);
    }

    /**
     * The port the server listens on.
     *
     * @return the port, resolved when {@code 0} was asked for
     */
    public int port() {
        return port;
    }

    /**
     * The access token.
     *
     * @return the token
     */
    public String token() {
        return token;
    }

    /**
     * The address to open in a browser, token included.
     *
     * @return e.g. {@code http://127.0.0.1:8787/?token=…}
     */
    public String url() {
        String shown = host.contains(":") && !host.startsWith("[") ? "[" + host + "]" : host;
        return "http://" + shown + ":" + port + "/?token=" + token;
    }

    /** Block until the server stops. */
    public void join() throws InterruptedException {
        server.join();
    }

    @Override
    public void close() throws Exception {
        try {
            server.stop();
        } finally {
            WebAgentEndpoint.SESSION.set(null);
        }
    }

    /** The Atmosphere servlet with its annotation map given explicitly instead of scanned. */
    static final class ExplicitAtmosphereServlet extends AtmosphereServlet {

        private static final long serialVersionUID = 1L;

        @Override
        public void init(ServletConfig config) throws ServletException {
            Map<Class<? extends Annotation>, Set<Class<?>>> annotations = new HashMap<>();
            annotations.put(AtmosphereAnnotation.class, Set.of(AiEndpointProcessor.class));
            annotations.put(AiEndpoint.class, Set.of(WebAgentEndpoint.class));
            config.getServletContext().setAttribute(DefaultAnnotationProcessor.ANNOTATION_ATTRIBUTE, annotations);
            super.init(config);
        }
    }
}
