package net.ladenthin.llama.args;

public enum RopeScalingType implements CliArg {

    UNSPECIFIED("unspecified"),
    NONE("none"),
    LINEAR("linear"),
    YARN2("yarn"),
    LONGROPE("longrope"),
    MAX_VALUE("maxvalue");

    private final String argValue;

    RopeScalingType(String value) {
        this.argValue = value;
    }

    public String getArgValue() {
        return argValue;
    }
}