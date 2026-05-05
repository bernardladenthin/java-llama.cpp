package net.ladenthin.llama.args;

/**
 * Mirostat sampling mode for {@code --mirostat}.
 *
 * <p>The arg values ({@code "0"}, {@code "1"}, {@code "2"}) are the integer strings
 * accepted by the CLI flag, matching llama.cpp's {@code MIROSTAT_*} constants.
 */
public enum MiroStat implements CliArg {

    DISABLED("0"),
    V1("1"),
    V2("2");

    private final String argValue;

    MiroStat(String argValue) {
        this.argValue = argValue;
    }

    @Override
    public String getArgValue() {
        return argValue;
    }
}
