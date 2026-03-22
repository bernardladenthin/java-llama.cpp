package examples;

import de.kherud.llama.InferenceParameters;
import de.kherud.llama.LlamaModel;
import de.kherud.llama.LlamaOutput;
import de.kherud.llama.ModelParameters;
import de.kherud.llama.Pair;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import org.junit.Ignore;

// Model file (models/codellama-7b.Q2_K.gguf) is not available in the models directory
@Ignore
public class ChatExample {

    public static void main(String... args) throws Exception {
        ModelParameters modelParams = new ModelParameters()
                .setModel("models/codellama-7b.Q2_K.gguf")
                .setGpuLayers(43);
        try (LlamaModel model = new LlamaModel(modelParams)) {
            BufferedReader reader = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
            List<Pair<String, String>> messages = new ArrayList<>();
            String system = "You are a helpful assistant.";
            while (true) {
                System.out.print("User: ");
                String input = reader.readLine();
                messages.add(new Pair<>("user", input));
                StringBuilder response = new StringBuilder();
                InferenceParameters inferParams = new InferenceParameters("")
                        .setMessages(system, messages)
                        .setUseChatTemplate(true);
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
