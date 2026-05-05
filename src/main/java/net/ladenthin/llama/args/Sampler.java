package net.ladenthin.llama.args;

/**
 * Sampling algorithm for {@code --samplers} (CLI) and the {@code "samplers"} JSON field.
 */
public enum Sampler implements CliArg {

    DRY("dry"),
    TOP_K("top_k"),
    TOP_P("top_p"),
    TYP_P("typ_p"),
    MIN_P("min_p"),
    TEMPERATURE("temperature"),
    XTC("xtc"),
    INFILL("infill"),
    PENALTIES("penalties");

    private final String argValue;

    Sampler(String argValue) {
        this.argValue = argValue;
    }

    @Override
    public String getArgValue() {
        return argValue;
    }
}
