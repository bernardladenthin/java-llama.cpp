// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.loader;

import static org.junit.jupiter.api.Assertions.*;

import net.ladenthin.llama.ClaudeGenerated;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose = "Verify that OSInfo correctly maps OS names to folder names used for native "
                + "library resolution (Windows, Mac/Darwin, AIX, Linux, unknown with special-char "
                + "stripping), that architecture names are normalised, that the system property "
                + "'net.ladenthin.llama.osinfo.architecture' overrides arch detection, and that "
                + "getNativeLibFolderPathForCurrentOS returns a two-part os/arch path.")
public class OSInfoTest {

    private static final String ARCH_OVERRIDE_PROP = LlamaSystemProperties.PREFIX + ".osinfo.architecture";
    private static final String OS_ARCH_PROP = "os.arch";
    private String previousArchOverride;
    private String previousOsArch;

    @BeforeEach
    public void saveProperties() {
        previousArchOverride = System.getProperty(ARCH_OVERRIDE_PROP);
        previousOsArch = System.getProperty(OS_ARCH_PROP);
    }

    @AfterEach
    public void restoreProperties() {
        if (previousArchOverride == null) {
            System.clearProperty(ARCH_OVERRIDE_PROP);
        } else {
            System.setProperty(ARCH_OVERRIDE_PROP, previousArchOverride);
        }
        if (previousOsArch == null) {
            System.clearProperty(OS_ARCH_PROP);
        } else {
            System.setProperty(OS_ARCH_PROP, previousOsArch);
        }
    }

    // -------------------------------------------------------------------------
    // translateOSNameToFolderName
    // -------------------------------------------------------------------------

    @Test
    public void testTranslateWindowsXP() {
        assertEquals("Windows", OSInfo.translateOSNameToFolderName("Windows XP"));
    }

    @Test
    public void testTranslateWindows10() {
        assertEquals("Windows", OSInfo.translateOSNameToFolderName("Windows 10"));
    }

    @Test
    public void testTranslateMacOSX() {
        assertEquals("Mac", OSInfo.translateOSNameToFolderName("Mac OS X"));
    }

    @Test
    public void testTranslateDarwin() {
        assertEquals("Mac", OSInfo.translateOSNameToFolderName("Darwin"));
    }

    @Test
    public void testTranslateAIX() {
        assertEquals("AIX", OSInfo.translateOSNameToFolderName("AIX"));
    }

    @Test
    public void testTranslateLinuxOnNonMuslNonAndroid() {
        // On a standard Linux test environment (non-musl, non-Android) this should return "Linux"
        String result = OSInfo.translateOSNameToFolderName("Linux");
        assertTrue(
                result.equals("Linux") || result.equals("Linux-Musl") || result.equals("Linux-Android"),
                "Expected Linux or Linux-Musl or Linux-Android, got: " + result);
    }

    @Test
    public void testTranslateUnknownOsStripsNonWordChars() {
        // Unknown OS names have non-word characters stripped
        assertEquals("SomeUnknownOS", OSInfo.translateOSNameToFolderName("Some Unknown OS!"));
    }

    // -------------------------------------------------------------------------
    // translateArchNameToFolderName
    // -------------------------------------------------------------------------

    @Test
    public void testTranslateArchStripsDots() {
        assertEquals("sparc64", OSInfo.translateArchNameToFolderName("sparc.64"));
    }

    @Test
    public void testTranslateArchStripsHyphens() {
        assertEquals("aarch64", OSInfo.translateArchNameToFolderName("aarch-64"));
    }

    @Test
    public void testTranslateArchNoSpecialChars() {
        assertEquals("x86", OSInfo.translateArchNameToFolderName("x86"));
    }

    @Test
    public void testTranslateArchEmptyString() {
        assertEquals("", OSInfo.translateArchNameToFolderName(""));
    }

    // -------------------------------------------------------------------------
    // getArchName with system property override
    // -------------------------------------------------------------------------

    @Test
    public void testGetArchNameWithOverride() {
        System.setProperty(ARCH_OVERRIDE_PROP, "custom_arch");
        assertEquals("custom_arch", OSInfo.getArchName());
    }

    @Test
    public void testGetArchNameWithoutOverrideReturnsNonEmpty() {
        System.clearProperty(ARCH_OVERRIDE_PROP);
        String arch = OSInfo.getArchName();
        assertNotNull(arch);
        assertFalse(arch.isEmpty());
    }

    // -------------------------------------------------------------------------
    // getNativeLibFolderPathForCurrentOS
    // -------------------------------------------------------------------------

    @Test
    public void testGetNativeLibFolderPathContainsSlash() {
        String path = OSInfo.getNativeLibFolderPathForCurrentOS();
        assertTrue(path.contains("/"), "Expected os/arch format, got: " + path);
    }

    @Test
    public void testGetNativeLibFolderPathHasTwoParts() {
        String path = OSInfo.getNativeLibFolderPathForCurrentOS();
        String[] parts = path.split("/");
        assertEquals(2, parts.length, "Expected exactly 2 parts in path: " + path);
        assertFalse(parts[0].isEmpty());
        assertFalse(parts[1].isEmpty());
    }

    // -------------------------------------------------------------------------
    // isAndroidRuntime (observable via system property)
    // -------------------------------------------------------------------------

    @Test
    public void testIsAndroidRuntimeFalseOnNonAndroid() {
        // On a JVM (not Android runtime), this should be false
        String runtimeName = System.getProperty("java.runtime.name", "");
        boolean expected = runtimeName.toLowerCase().contains("android");
        assertEquals(expected, OSInfo.isAndroidRuntime());
    }

    // -------------------------------------------------------------------------
    // getArchName: the archMapping alias table
    //
    // Only the NON-IDENTITY aliases are asserted. An identity entry such as
    // s390x -> s390x is behaviourally redundant with the \W-stripping fallback,
    // so pinning it could not detect that entry being lost. Every pair below can:
    // the expected value differs from what translateArchNameToFolderName alone
    // would produce, so a dropped map entry sends LlamaLoader to a resource
    // directory that was never shipped (amd64 -> "amd64", power_pc -> "powerpc").
    // -------------------------------------------------------------------------

    @Test
    public void testArchMappingAliasesResolveToTheCanonicalFolderName() {
        final String[][] aliases = {
            {"i386", OSInfo.X86},
            {"i486", OSInfo.X86},
            {"i586", OSInfo.X86},
            {"i686", OSInfo.X86},
            {"pentium", OSInfo.X86},
            {"amd64", OSInfo.X86_64},
            {"em64t", OSInfo.X86_64},
            {"universal", OSInfo.X86_64},
            {"ia64w", OSInfo.IA64},
            {"ia64n", OSInfo.IA64_32},
            {"power", OSInfo.PPC},
            {"powerpc", OSInfo.PPC},
            {"power_pc", OSInfo.PPC},
            {"power_rs", OSInfo.PPC},
            {"power64", OSInfo.PPC64},
            {"powerpc64", OSInfo.PPC64},
            {"power_pc64", OSInfo.PPC64},
            {"power_rs64", OSInfo.PPC64},
        };
        System.clearProperty(ARCH_OVERRIDE_PROP);
        for (String[] alias : aliases) {
            System.setProperty(OS_ARCH_PROP, alias[0]);
            assertEquals(alias[1], OSInfo.getArchName(), "os.arch=" + alias[0] + " must resolve to " + alias[1]);
        }
    }

    @Test
    public void testArchMappingLookupIsCaseInsensitive() {
        // getArchName lowercases before the map lookup; a JVM reporting "AMD64" must still
        // land on x86_64 rather than falling through to the literal folder name "AMD64".
        System.clearProperty(ARCH_OVERRIDE_PROP);
        System.setProperty(OS_ARCH_PROP, "AMD64");
        assertEquals(OSInfo.X86_64, OSInfo.getArchName());
    }

    @Test
    public void testUnmappedArchFallsThroughToTheStrippedName() {
        // The fallback that carries aarch64 / s390x: absent from archMapping, so it is the
        // \W-stripping translate step that produces the folder name.
        System.clearProperty(ARCH_OVERRIDE_PROP);
        System.setProperty(OS_ARCH_PROP, "s390x");
        assertEquals("s390x", OSInfo.getArchName());
    }
}
