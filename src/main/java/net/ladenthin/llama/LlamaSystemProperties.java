package net.ladenthin.llama;

/**
 * Resolves library-specific system properties under the {@link #PREFIX} domain prefix.
 */
public class LlamaSystemProperties {

    public static final String PREFIX = "net.ladenthin.llama";

    private String getProperty(String suffix) {
        return System.getProperty(PREFIX + suffix);
    }

    /** Custom directory containing the native jllama shared library. */
    public String getLibPath() {
        return getProperty(".lib.path");
    }

    /** Override for the native library file name. */
    public String getLibName() {
        return getProperty(".lib.name");
    }

    /**
     * Custom temporary directory used when extracting the native library from
     * the JAR. Falls back to {@code java.io.tmpdir} if absent.
     */
    public String getTmpDir() {
        return getProperty(".tmpdir");
    }

    /** Architecture override for OS/arch detection in {@link OSInfo}. */
    public String getOsinfoArchitecture() {
        return getProperty(".osinfo.architecture");
    }

    /** Number of GPU layers used in tests; parsed by the test suite. */
    public String getTestNgl() {
        return getProperty(".test.ngl");
    }
}
