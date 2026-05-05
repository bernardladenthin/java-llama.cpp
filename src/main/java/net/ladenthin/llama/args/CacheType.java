package net.ladenthin.llama.args;

/**
 * KV cache quantization type for {@code --cache-type-k} and {@code --cache-type-v}.
 */
public enum CacheType implements CliArg {

    F32("f32"),
    F16("f16"),
    BF16("bf16"),
    Q8_0("q8_0"),
    Q4_0("q4_0"),
    Q4_1("q4_1"),
    IQ4_NL("iq4_nl"),
    Q5_0("q5_0"),
    Q5_1("q5_1");

    private final String argValue;

    CacheType(String argValue) {
        this.argValue = argValue;
    }

    @Override
    public String getArgValue() {
        return argValue;
    }
}
