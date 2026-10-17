// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.loader;

import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Paths;
import net.ladenthin.llama.args.ModelFlag;
import net.ladenthin.llama.exception.ModelUnavailableException;
import net.ladenthin.llama.parameters.ModelParameters;

/**
 * Pure-Java guard that fails fast with a typed
 * {@link net.ladenthin.llama.exception.ModelUnavailableException} when the caller asked to run
 * {@link net.ladenthin.llama.args.ModelFlag#OFFLINE offline} (or via
 * {@link net.ladenthin.llama.parameters.ModelParameters#setOffline(boolean) setOffline(true)}) but
 * the configured local model file does not exist — i.e. the loader would have had to download a
 * replacement but is forbidden to.
 *
 * <p>Lives outside {@link net.ladenthin.llama.LlamaModel} so that unit tests can exercise the
 * check without triggering {@code LlamaModel}'s {@link LlamaLoader} static initializer (which
 * loads the JNI library and is not available in CPU-only / non-native test environments).</p>
 *
 * <h2>Why a deterministic pre-check and not an exception heuristic</h2>
 *
 * <p>Upstream's {@code --offline} flag ({@code common_params::offline}) makes the model-download
 * pipeline skip every download task; a present model loads, a missing one fails later inside the
 * native loader with a generic load error. Rather than pattern-match that native message (fragile),
 * this guard runs <em>before</em> the native call: if {@code OFFLINE} is set and a local
 * {@code --model} path is configured that does not exist on disk, it throws
 * {@link net.ladenthin.llama.exception.ModelUnavailableException} directly. This precisely flags the
 * common air-gapped case (a pre-staged path that is absent) without mislabelling unrelated load
 * failures. Loads driven purely by {@code --hf-repo} / {@code --model-url} have no local path to
 * pre-check here, so they fall through to whatever the native loader reports.</p>
 *
 * <p>The predecessor {@code SkipDownloadFailureTranslator} keyed on a parse-failure message because
 * {@code --skip-download} was never a registered upstream argument; that flag (and the
 * {@code common_skip_download_exception} it referenced) was removed in llama.cpp b9803 and replaced
 * here by the working {@code --offline} flag.</p>
 */
public final class OfflineModelGuard {

    private OfflineModelGuard() {
        // utility — not instantiable
    }

    /**
     * Throws {@link net.ladenthin.llama.exception.ModelUnavailableException} when {@code parameters}
     * has {@link net.ladenthin.llama.args.ModelFlag#OFFLINE} set and its configured local
     * {@code --model} path points at a file that does not exist; otherwise returns normally.
     *
     * @param parameters the parameters about to be passed to the native loader
     * @throws net.ladenthin.llama.exception.ModelUnavailableException when offline and the
     *         configured local model file is missing
     */
    public static void check(ModelParameters parameters) {
        if (!parameters.hasFlag(ModelFlag.OFFLINE)) {
            return;
        }
        String model = parameters.getModel();
        if (model != null && !model.isEmpty() && !localFileExists(model)) {
            throw new ModelUnavailableException(
                    "Model unavailable: --offline is set but the configured model file does not "
                            + "exist (no download attempted): " + model);
        }
    }

    /**
     * Whether {@code model} names an existing file. A path the file system cannot represent (a NUL
     * character, or a character Windows forbids) cannot name a file, so it counts as missing instead
     * of escaping as an {@link InvalidPathException}.
     *
     * @param model the configured local model path
     * @return {@code true} if the file exists
     */
    private static boolean localFileExists(String model) {
        try {
            return Files.exists(Paths.get(model));
        } catch (InvalidPathException e) {
            return false;
        }
    }
}
