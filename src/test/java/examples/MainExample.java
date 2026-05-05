package examples;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

import net.ladenthin.llama.InferenceParameters;
import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.LlamaOutput;
import net.ladenthin.llama.ModelParameters;
import net.ladenthin.llama.args.MiroStat;

@SuppressWarnings("InfiniteLoopStatement")
public class MainExample {

    public static void main(String... args) throws IOException {
        ModelParameters modelParams = new ModelParameters()
				.setModel("models/mistral-7b-instruct-v0.2.Q2_K.gguf")
                .setGpuLayers(43);
        String system = "This is a conversation between User and Llama, a friendly chatbot.\n" +
                "Llama is helpful, kind, honest, good at writing, and never fails to answer any " +
                "requests immediately and with precision.\n\n" +
                "User: Hello Llama\n" +
                "Llama: Hello.  How may I help you today?";
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        try (LlamaModel model = new LlamaModel(modelParams)) {
            System.out.print(system);
            String prompt = system;
            while (true) {
                prompt += "\nUser: ";
                System.out.print("\nUser: ");
                String input = reader.readLine();
                prompt += input;
                System.out.print("Llama: ");
                prompt += "\nLlama: ";
				InferenceParameters inferParams = new InferenceParameters(prompt)
						.setTemperature(0.7f)
						.setPenalizeNl(true)
						.setMiroStat(MiroStat.V2)
						.setStopStrings("User:");
                for (LlamaOutput output : model.generate(inferParams)) {
                    System.out.print(output);
                    prompt += output;
                }
            }
        }
    }
}
