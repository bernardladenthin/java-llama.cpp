// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package examples;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.parameters.ModelParameters;
import net.ladenthin.llama.value.LlamaOutput;
import net.ladenthin.llama.value.Pair;
import org.junit.jupiter.api.Disabled;

// Model file (models/codellama-7b.Q2_K.gguf) is not available in the models directory
@Disabled
public class ChatExample {

    public static void main(String... args) throws Exception {
        ModelParameters modelParams =
                new ModelParameters().setModel("models/codellama-7b.Q2_K.gguf").setGpuLayers(43);
        try (LlamaModel model = new LlamaModel(modelParams)) {
            BufferedReader reader = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
            List<Pair<String, String>> messages = new ArrayList<>();
            String system = "You are a helpful assistant.";
            while (true) {
                System.out.print("User: ");
                String input = reader.readLine();
                messages.add(new Pair<>("user", input));
                StringBuilder response = new StringBuilder();
                InferenceParameters inferParams = new InferenceParameters("").withMessages(system, messages);
                System.out.print("Assistant: ");
                for (LlamaOutput output : model.generate(inferParams)) {
                    System.out.print(output);
                    response.append(output);
                }
                System.out.println();
                messages.add(new Pair<>("assistant", response.toString()));
            }
        }
    }
}
