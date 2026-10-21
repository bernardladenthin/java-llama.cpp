// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.loader;

import java.io.File;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.function.Function;
import net.ladenthin.llama.args.ModelFlag;
import net.ladenthin.llama.exception.ModelUnavailableException;
import net.ladenthin.llama.parameters.ModelParameters;
import org.jspecify.annotations.Nullable;

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
     * {@code --model} path points at a file that does not exist; otherwise returns normally. A path
     * the JVM's file-name encoding cannot represent (a non-ASCII name in a JVM started with the POSIX
     * locale) is not judged here but left to the native loader, which opens it with its UTF-8 bytes.
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
     * Whether {@code model} names an existing file, as far as the JVM can tell.
     *
     * @param model the configured local model path
     * @return {@code true} if the file exists or the JVM cannot decide (see {@link #countsAsMissing})
     */
    private static boolean localFileExists(String model) {
        return !countsAsMissing(
                model,
                fileNameEncoding(File.separatorChar == '\\', System.getProperty("sun.jnu.encoding")),
                Paths::get);
    }

    /**
     * The charset {@code Paths.get} encodes a file name with. On Windows none: {@code WindowsPath}
     * parses the UTF-16 string and never consults {@code sun.jnu.encoding}, so every rejection there
     * is the file system's, which UTF-8 (able to encode any well-formed string) expresses. Elsewhere
     * {@code UnixPath} encodes with {@code sun.jnu.encoding}.
     *
     * @param windows whether this is Windows
     * @param jnu     the value of {@code sun.jnu.encoding}, or {@code null}
     * @return the charset to test a rejected name against
     */
    static Charset fileNameEncoding(boolean windows, @Nullable String jnu) {
        return windows ? StandardCharsets.UTF_8 : jnuEncoding(jnu);
    }

    /**
     * The charset the JVM encodes file names with, resolved like the JDK does: the
     * {@code sun.jnu.encoding} property (set from the locale at startup, on Java 8 as well), and the
     * default charset when it is absent or names no supported charset.
     *
     * @param name the value of {@code sun.jnu.encoding}, or {@code null}
     * @return the file-name charset
     */
    static Charset jnuEncoding(@Nullable String name) {
        if (name != null) {
            try {
                return Charset.forName(name);
            } catch (IllegalArgumentException e) {
                // IllegalCharsetNameException and UnsupportedCharsetException: the JDK's fallback below
            }
        }
        return Charset.defaultCharset();
    }

    /**
     * Whether the guard reports {@code model} as missing.
     *
     * <ol>
     *   <li>A NUL character: missing. No file name can contain one, and native code would cut the path
     *       at it, so it is never handed on.</li>
     *   <li>Otherwise the file system decides through {@code parse} ({@code Paths.get}).</li>
     *   <li>When {@code parse} rejects the name with an {@link InvalidPathException}, it is missing only
     *       if {@code jnuEncoding} can encode it: then the JVM could spell the name and the file system
     *       refused it (a character Windows forbids). If the JVM's file-name encoding cannot represent
     *       it, the rejection says nothing about the file: on Linux {@code Paths.get} throws for every
     *       non-ASCII name in a JVM started with the POSIX locale (no {@code LANG}), while the native
     *       loader receives the UTF-8 bytes and opens the file. Such a path is left to the native
     *       loader, which reports a missing file itself.</li>
     * </ol>
     *
     * @param model       the configured local model path
     * @param jnuEncoding the charset the JVM encodes file names with
     * @param parse       turns the name into a path, {@code Paths::get} outside tests
     * @return {@code true} if the offline guard should report the model as missing
     */
    static boolean countsAsMissing(String model, Charset jnuEncoding, Function<String, Path> parse) {
        if (model.indexOf('\0') >= 0) {
            return true;
        }
        try {
            return !Files.exists(parse.apply(model));
        } catch (InvalidPathException e) {
            return jnuEncoding.newEncoder().canEncode(model);
        }
    }
}
