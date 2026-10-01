// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

package net.ladenthin.llama.consumertest;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import net.ladenthin.llama.GgufInspector;
import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.parameters.ModelParameters;
import net.ladenthin.llama.value.GgufMetadata;
import org.junit.Test;
import org.junit.runner.RunWith;

/**
 * On-emulator smoke of the shipped AAR: the binding must load via
 * {@code System.loadLibrary("jllama")} from the APK's native-lib dir (the AAR's
 * {@code jni/x86_64/} on the CI emulator), and both the pure-Java GGUF inspector and real
 * native inference must work on Android/bionic. The model file is adb-pushed by the
 * {@code test-android-emulator} CI job; every test self-skips when it is absent so the
 * fixture stays green on a bare emulator.
 */
@RunWith(AndroidJUnit4.class)
public class OnDeviceInferenceTest {

    /** Where the CI job adb-pushes the test GGUF (world-readable in /data/local/tmp). */
    private static final String MODEL_PATH = "/data/local/tmp/jllama-test-model.gguf";

    private static void assumeModelPresent() {
        assumeTrue("test model not pushed to " + MODEL_PATH, new File(MODEL_PATH).canRead());
    }

    @Test
    public void ggufInspectorReadsTheModelOnDevice() throws IOException {
        assumeModelPresent();

        GgufMetadata meta = GgufInspector.read(Paths.get(MODEL_PATH));

        assertTrue("tensor count must be positive, was " + meta.getTensorCount(), meta.getTensorCount() > 0);
        assertTrue("architecture key must be present: " + meta, meta.getArchitecture().isPresent());
    }

    @Test
    public void nativeInferenceGeneratesTokensOnDevice() {
        assumeModelPresent();

        // Loading LlamaModel exercises System.loadLibrary("jllama") + JNI_OnLoad's
        // FindClass resolution against the D8-dexed classes — the load-time surface
        // the build-only CI jobs cannot reach.
        try (LlamaModel model = new LlamaModel(
                new ModelParameters().setModel(MODEL_PATH).setCtxSize(512).setGpuLayers(0))) {
            String generated = model.complete(new InferenceParameters("Hello").withNPredict(8));

            assertFalse("expected generated tokens, got empty output", generated.isEmpty());
        }
    }
}
