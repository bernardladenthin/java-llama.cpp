///usr/bin/env jbang "$0" "$@" ; exit $?
// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
//JAVA 8+
//DEPS net.ladenthin:llama:5.2.0
//DEPS net.ladenthin:llama:5.2.0:cpu-linux-x86-64
//DEPS net.ladenthin:llama:5.2.0:cpu-linux-aarch64
//DEPS net.ladenthin:llama:5.2.0:cpu-linux-s390x
//DEPS net.ladenthin:llama:5.2.0:cpu-windows-x86-64
//DEPS net.ladenthin:llama:5.2.0:cpu-windows-x86
//DEPS net.ladenthin:llama:5.2.0:cpu-windows-aarch64
//DEPS net.ladenthin:llama:5.2.0:metal-macos-aarch64

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.Session;
import net.ladenthin.llama.parameters.ModelParameters;
import net.ladenthin.llama.value.LlamaOutput;

/**
 * A multi-turn chat on the console that JBang runs with no project and no build:
 *
 * <pre>
 *   jbang Chat.java path/to/an-instruct-model.gguf
 * </pre>
 *
 * The {@code //DEPS} lines above are the classes jar and the CPU natives jars of every desktop platform
 * -- the jars {@code net.ladenthin:llama-platform} names, listed here one by one because JBang treats a
 * {@code pom} dependency as a BOM and puts nothing of it on the classpath. The loader picks the jar of
 * this machine; the others are downloaded once and ignored. A GPU backend (for example
 * {@code net.ladenthin:llama:5.2.0:cuda13-linux-x86-64}) is one more DEPS line and is tried before the CPU.
 *
 * <p>A {@link Session} keeps the conversation, so every turn passes only the new message, and the model's
 * chat template is applied for it; each reply is streamed as it is generated. An empty line or the end
 * of the input (Ctrl+D, on Windows Ctrl+Z) ends the chat.
 */
public class Chat {

    public static void main(String... args) throws IOException {
        if (args.length != 1) {
            System.err.println("usage: jbang Chat.java <model.gguf>");
            System.exit(2);
        }
        BufferedReader console = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        try (LlamaModel model = new LlamaModel(new ModelParameters().setModel(args[0]).setCtxSize(4096));
                // Slot 0 holds this conversation; the customizer applies to every turn.
                Session session = new Session(model, 0, "You are a concise, helpful assistant.",
                        p -> p.withNPredict(512).withTemperature(0.7f))) {
            while (true) {
                System.out.print("\nYou: ");
                String message = console.readLine();
                if (message == null || message.trim().isEmpty()) {
                    return;
                }
                System.out.print("Assistant: ");
                StringBuilder reply = new StringBuilder();
                try {
                    for (LlamaOutput output : session.stream(message)) {
                        System.out.print(output.text);
                        reply.append(output.text);
                    }
                } catch (RuntimeException e) {
                    // Without a committed reply the session would refuse every further turn.
                    session.cancelStream();
                    throw e;
                }
                session.commitStreamedReply(reply.toString());
                System.out.println();
            }
        }
    }
}
