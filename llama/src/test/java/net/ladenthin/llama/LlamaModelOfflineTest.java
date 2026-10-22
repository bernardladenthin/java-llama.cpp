// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import net.ladenthin.llama.args.ModelFlag;
import net.ladenthin.llama.exception.ModelUnavailableException;
import net.ladenthin.llama.loader.OfflineModelGuard;
import net.ladenthin.llama.parameters.ModelParameters;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Unit tests for the {@code OFFLINE} plumbing on {@link ModelParameters} and the paired
 * deterministic pre-check in {@link OfflineModelGuard}.
 *
 * <p>These tests do NOT load the native library &#x2014; they exercise pure Java logic: the
 * boolean-setter round-trip via {@link ModelParameters#hasFlag(ModelFlag)} and the guard that
 * promotes an offline-with-missing-local-model load into a typed
 * {@link ModelUnavailableException} before the native loader is ever called.</p>
 */
public class LlamaModelOfflineTest {

    /** Default constructor used by JUnit Jupiter. */
    public LlamaModelOfflineTest() {
        // no-op
    }

    @Test
    @DisplayName("setOffline(true) sets the OFFLINE flag")
    public void setOffline_true_setsFlag() {
        ModelParameters p = new ModelParameters().setOffline(true);
        assertTrue(p.hasFlag(ModelFlag.OFFLINE));
    }

    @Test
    @DisplayName("setOffline(false) clears the OFFLINE flag")
    public void setOffline_false_clearsFlag() {
        ModelParameters p = new ModelParameters().setOffline(true).setOffline(false);
        assertFalse(p.hasFlag(ModelFlag.OFFLINE));
    }

    @Test
    @DisplayName("hasFlag returns false by default")
    public void hasFlag_byDefault_returnsFalse() {
        assertFalse(new ModelParameters().hasFlag(ModelFlag.OFFLINE));
    }

    @Test
    @DisplayName("getModel returns null when no local path is set")
    public void getModel_unset_returnsNull() {
        org.junit.jupiter.api.Assertions.assertNull(new ModelParameters().getModel());
    }

    @Test
    @DisplayName("check: offline + missing local model -> ModelUnavailableException")
    public void check_offlineAndMissingModel_throws() {
        ModelParameters p = new ModelParameters().setOffline(true).setModel("/definitely/does/not/exist/model.gguf");

        ModelUnavailableException ex = assertThrows(ModelUnavailableException.class, () -> OfflineModelGuard.check(p));
        assertTrue(
                ex.getMessage() != null && ex.getMessage().contains("--offline"),
                "message should mention the --offline flag for caller diagnosis");
    }

    @Test
    @DisplayName("check: offline + a path the file system cannot represent -> ModelUnavailableException")
    public void check_offlineAndUnrepresentablePath_throwsModelUnavailable() {
        // A NUL character is invalid in a path on every platform, so Paths.get rejects it with an
        // InvalidPathException. Such a file cannot exist, which is exactly the case the guard
        // promises to report as the typed exception.
        ModelParameters p = new ModelParameters().setOffline(true).setModel("models/bad\u0000name.gguf");

        ModelUnavailableException ex = assertThrows(ModelUnavailableException.class, () -> OfflineModelGuard.check(p));
        assertTrue(
                ex.getMessage() != null && ex.getMessage().contains("--offline"),
                "message should mention the --offline flag for caller diagnosis");
    }

    @Test
    @DisplayName("check: offline + present local model -> no exception")
    public void check_offlineAndPresentModel_doesNotThrow(@TempDir Path tempDir) throws IOException {
        Path model = Files.createFile(tempDir.resolve("model.gguf"));
        ModelParameters p = new ModelParameters().setOffline(true).setModel(model.toString());

        assertDoesNotThrow(() -> OfflineModelGuard.check(p));
    }

    @Test
    @DisplayName("check: offline NOT set -> no exception even if local model is missing")
    public void check_offlineNotSet_doesNotThrow() {
        ModelParameters p = new ModelParameters().setModel("/definitely/does/not/exist/model.gguf");
        assertDoesNotThrow(() -> OfflineModelGuard.check(p));
    }

    @Test
    @DisplayName("check: offline + no local path (e.g. --hf-repo) -> falls through, no exception")
    public void check_offlineWithoutLocalPath_doesNotThrow() {
        ModelParameters p = new ModelParameters().setOffline(true).setHfRepo("ggml-org/some-model");
        assertDoesNotThrow(() -> OfflineModelGuard.check(p));
    }
}
