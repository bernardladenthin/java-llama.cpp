// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.loader;

/**
 * Whether a native library for this platform is on the test classpath, i.e. whether the
 * native-backed tests can run. The single copy of this check: it spells out the resource layout
 * ({@code net/ladenthin/llama/<os>/<arch>/<backend>/}) instead of asking the loader, so a loader
 * regression cannot make these tests skip silently, and a layout change needs one edit here
 * instead of one per test class (three had drifted to the old flat layout once).
 */
public final class NativeLibraryPresence {

    private NativeLibraryPresence() {}

    /**
     * Checks every backend directory the loader would try.
     *
     * @return whether any of them holds the {@code jllama} library for this platform
     */
    public static boolean onClasspath() {
        ClassLoader loader = NativeLibraryPresence.class.getClassLoader();
        for (String backend : LlamaLoader.BACKEND_PRIORITY) {
            String resource = "net/ladenthin/llama/" + OSInfo.getNativeLibFolderPathForCurrentOS() + "/" + backend + "/"
                    + System.mapLibraryName("jllama");
            if (loader.getResource(resource) != null) {
                return true;
            }
        }
        return false;
    }
}
