// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.arrayContaining;
import static org.hamcrest.Matchers.emptyArray;
import static org.hamcrest.Matchers.is;

import org.junit.jupiter.api.Test;

/**
 * Pure-Java unit tests for {@link ServerLauncher}'s single dispatch primitive,
 * {@link ServerLauncher#withoutFlag(String[], String)}. Selection is derived from the length change
 * (result shorter iff the flag was present), so these tests cover both the stripping behaviour and
 * that selection signal. No server is started and no native library is required.
 */
public class ServerLauncherTest {

    private static final String FLAG = ServerLauncher.OPENAI_COMPAT_FLAG;

    // --- selection signal: shorter iff the flag was present ---

    @Test
    public void resultIsShorterWhenFlagPresent() {
        String[] in = {FLAG, "-m", "m.gguf", "--port", "9931"};
        assertThat(ServerLauncher.withoutFlag(in, FLAG).length < in.length, is(true));
    }

    @Test
    public void resultKeepsLengthWhenFlagAbsent() {
        String[] in = {"-m", "m.gguf", "--port", "9931"};
        assertThat(ServerLauncher.withoutFlag(in, FLAG).length == in.length, is(true));
    }

    @Test
    public void flagPositionDoesNotMatter() {
        String[] in = {"-m", "m.gguf", FLAG};
        assertThat(ServerLauncher.withoutFlag(in, FLAG).length < in.length, is(true));
    }

    // --- stripping behaviour ---

    @Test
    public void stripsTheSelectorAndPreservesTheRest() {
        String[] out = ServerLauncher.withoutFlag(new String[] {FLAG, "-m", "m.gguf", "--port", "9931"}, FLAG);
        assertThat(out, arrayContaining("-m", "m.gguf", "--port", "9931"));
    }

    @Test
    public void removesEveryOccurrence() {
        String[] out = ServerLauncher.withoutFlag(new String[] {FLAG, "-m", "m.gguf", FLAG}, FLAG);
        assertThat(out, arrayContaining("-m", "m.gguf"));
    }

    @Test
    public void isNoOpWhenAbsent() {
        assertThat(ServerLauncher.withoutFlag(new String[] {"-m", "m.gguf"}, FLAG), arrayContaining("-m", "m.gguf"));
    }

    @Test
    public void emptyArgsStayEmpty() {
        assertThat(ServerLauncher.withoutFlag(new String[] {}, FLAG), is(emptyArray()));
    }
}
