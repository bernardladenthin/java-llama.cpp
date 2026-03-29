package de.kherud.llama;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

@ClaudeGenerated(
        purpose = "Verify that OSInfo correctly maps OS names to folder names used for native " +
                  "library resolution (Windows, Mac/Darwin, AIX, Linux, unknown with special-char " +
                  "stripping), that architecture names are normalised, that the system property " +
                  "'de.kherud.llama.osinfo.architecture' overrides arch detection, and that " +
                  "getNativeLibFolderPathForCurrentOS returns a two-part os/arch path."
)
public class OSInfoTest {

	private static final String ARCH_OVERRIDE_PROP = "de.kherud.llama.osinfo.architecture";
	private String previousArchOverride;

	@Before
	public void saveProperties() {
		previousArchOverride = System.getProperty(ARCH_OVERRIDE_PROP);
	}

	@After
	public void restoreProperties() {
		if (previousArchOverride == null) {
			System.clearProperty(ARCH_OVERRIDE_PROP);
		} else {
			System.setProperty(ARCH_OVERRIDE_PROP, previousArchOverride);
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
		assertTrue("Expected Linux or Linux-Musl or Linux-Android, got: " + result,
				result.equals("Linux") || result.equals("Linux-Musl") || result.equals("Linux-Android"));
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
		assertTrue("Expected os/arch format, got: " + path, path.contains("/"));
	}

	@Test
	public void testGetNativeLibFolderPathHasTwoParts() {
		String path = OSInfo.getNativeLibFolderPathForCurrentOS();
		String[] parts = path.split("/");
		assertEquals("Expected exactly 2 parts in path: " + path, 2, parts.length);
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
}
