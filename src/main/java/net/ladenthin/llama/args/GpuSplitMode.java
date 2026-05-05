package net.ladenthin.llama.args;

/**
 * GPU tensor split mode for {@code --split-mode}.
 */
public enum GpuSplitMode implements CliArg {

    NONE("none"),
    LAYER("layer"),
    ROW("row");

    private final String argValue;

    GpuSplitMode(String argValue) {
        this.argValue = argValue;
    }

    @Override
    public String getArgValue() {
        return argValue;
    }
}
