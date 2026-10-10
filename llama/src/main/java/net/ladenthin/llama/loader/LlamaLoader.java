// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.loader;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.FileTime;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.jar.JarFile;
import java.util.stream.Stream;
import java.util.zip.ZipEntry;
import lombok.ToString;
import org.jspecify.annotations.Nullable;

/**
 * Set the system property {@code net.ladenthin.llama.lib.path} appropriately
 * so that the library can find {@code *.dll}, {@code *.dylib} and
 * {@code *.so} files, according to the current OS (Windows, Linux, macOS).
 *
 * <p>The library files are extracted from the natives jars on the classpath: one directory per
 * backend, {@code net/ladenthin/llama/<os>/<arch>/<backend>/}. The directory of a {@link
 * #LIBRARY_BACKENDS library backend} ({@code cpu}, or {@code metal} on macOS) holds the {@code jllama}
 * library, with ggml's shared libraries and CPU modules next to it in a modular build; the directory
 * of a {@link #MODULE_BACKENDS module backend} (a GPU natives jar) holds one ggml backend module. All
 * of them are extracted into one directory, the library is loaded, and its {@code JNI_OnLoad} has ggml
 * load every module found there -- so adding a GPU natives jar to the classpath adds its backend, and
 * nothing in the jars overlaps. {@code net.ladenthin.llama.backend} narrows the modules (see {@link
 * #selectModules}).
 *
 * <p>Historically the loader also honoured a {@code net.ladenthin.llama.lib.name}
 * property that overrode the resolved library filename. Upstream removed the
 * code path that read it in {@code kherud/java-llama.cpp} commit {@code 6bb63e1}
 * (&quot;add ggml shared library to binding&quot;) when the loader was extended to
 * load multiple shared libraries (ggml + jllama) as separate files &mdash; the
 * single-name-override model is incompatible with that. The Javadoc mention
 * has since been a documentation lie in both upstream and this fork; it has
 * now been removed here, and the corresponding {@code getLibName()} getter
 * has been deleted from {@code LlamaSystemProperties}.
 *
 * <p>usage: call {@link #initialize()} before using the library.
 *
 * @author leo
 */
@SuppressWarnings("UseOfSystemOutOrSystemErr")
@ToString
public class LlamaLoader {

    /**
     * Private monitor guarding {@link #initialize()}. Synchronizing on this
     * dedicated object instead of {@code LlamaLoader.class} keeps the lock
     * private to this class, so untrusted code that can reach the public
     * {@code LlamaLoader} type cannot acquire the same intrinsic lock and
     * interfere with library initialization (SpotBugs
     * {@code USO_UNSAFE_STATIC_METHOD_SYNCHRONIZATION}).
     */
    private static final Object INITIALIZE_LOCK = new Object();

    private static boolean extracted = false;

    /** Whether {@link #initialize()} is running; only read and written under the lock. */
    private static boolean initializing = false;

    private static final LlamaSystemProperties systemProperties = new LlamaSystemProperties();
    private static final NativeLibraryPermissionSetter permissionSetter = new NativeLibraryPermissionSetter(System.err);

    /**
     * Classpath root of the bundled native libraries, without a leading slash because it is
     * resolved through the {@link ClassLoader} (see {@link #resource(String)}). Below it every
     * backend has its own directory, {@code <os>/<arch>/<backend>/}, and each natives jar
     * carries exactly one of them, so any combination of natives jars can share one classpath.
     * It must NOT be derived from this loader's own Java package, which moved to
     * {@code net.ladenthin.llama.loader} during the layered restructure.
     */
    static final String NATIVE_RESOURCE_BASE = "net/ladenthin/llama";

    /**
     * The backend directories that hold the {@code jllama} library itself: {@code metal} (the macOS
     * build, with Metal compiled in) and {@code cpu} (every other platform), tried in this order. A
     * classpath has one of them per platform, and everything -- the library, its sibling files and
     * every GPU module -- is extracted into that backend's directory.
     */
    static final Set<String> LIBRARY_BACKENDS =
            Collections.unmodifiableSet(new LinkedHashSet<>(Arrays.asList("metal", "cpu")));

    /**
     * The backend directories that hold a ggml backend module ({@code libggml-cuda.so}, {@code
     * ggml-vulkan.dll}, ...) and nothing the JVM loads itself: a GPU natives jar. Every one present on
     * the classpath is extracted next to the library, where ggml loads it at {@code JNI_OnLoad} in
     * its own fixed order (CUDA, HIP, SYCL, Vulkan, OpenCL, OpenVINO); a module whose vendor runtime
     * is missing loads nothing and registers no device, and llama.cpp counts a GPU that two backends
     * reach only once. The order here is the one of {@code natives.csv} and decides nothing.
     */
    static final Set<String> MODULE_BACKENDS = Collections.unmodifiableSet(
            new LinkedHashSet<>(Arrays.asList("cuda13", "rocm", "sycl", "vulkan", "opencl", "openvino")));

    /**
     * Optional file in a backend directory naming sibling files (one per line, {@code #}
     * comments allowed) to extract and load, in order, before that backend's library &mdash;
     * e.g. the OpenCL ICD loader bundled with OpenVINO on Windows, which Windows would not find
     * next to the library on its own.
     */
    static final String BACKEND_EXTRAS_FILE = "jllama-extras.txt";

    /**
     * Optional file in a backend directory naming sibling files (same format as {@link
     * #BACKEND_EXTRAS_FILE}) that are only extracted next to the library, never loaded by Java:
     * the shared ggml libraries the library finds through its {@code $ORIGIN} run path, and the
     * backend modules it loads itself from its own directory: one CPU module per instruction-set
     * level in a {@code JLLAMA_BACKEND_DL} build, of which ggml picks the best for the running CPU,
     * and the GPU module of every module backend on the classpath. Loading them from Java would defeat
     * exactly that choice. A module backend's directory has this file and no library.
     */
    static final String BACKEND_FILES_FILE = "jllama-files.txt";

    /**
     * File every natives directory carries, written by the build ({@code llama/CMakeLists.txt}):
     * {@code key=value} lines naming the build it came from. A module is extracted next to a library
     * of the same {@link #BUILD_KEY_LLAMA_CPP llama.cpp tag} only (see {@link #requireSameBuild}).
     */
    static final String BACKEND_BUILD_FILE = "jllama-build.txt";

    /** The key of {@link #BACKEND_BUILD_FILE} naming the llama.cpp tag the directory was built from. */
    static final String BUILD_KEY_LLAMA_CPP = "llama.cpp";

    /**
     * Prefix of the per-backend extraction subdirectory below the temp dir,
     * {@code jllama-backend-<backend>-<key>} (see {@link #extractionDirectoryName}). Deliberately
     * starts with {@code jllama} so {@link #shouldCleanPath(Path)} matches it during cleanup.
     */
    static final String BACKEND_TEMP_DIR_PREFIX = "jllama-backend-";

    /**
     * How long an extraction directory of <em>another</em> build is left alone after it was last
     * touched. A JVM touches its directory when it starts extracting into it, so a directory younger
     * than this may belong to a JVM that is still starting; an older one is a leftover of a build that
     * is not on this classpath, and the next start of any build removes it. The directories of the
     * builds on this classpath are never removed (see {@link #cleanup(Set)}).
     */
    static final Duration STALE_EXTRACTION_AGE = Duration.ofMinutes(10);

    /** Shader source a non-embedding Metal build ships next to its library. */
    private static final String METAL_SOURCE_FILE = "ggml-metal.metal";

    /** Static utility holder; not instantiable. */
    private LlamaLoader() {}

    /**
     * Loads the llama and jllama shared libraries
     */
    public static void initialize() {
        runOnceOnThisThread(LlamaLoader::load);
    }

    /**
     * Runs {@code body} under the initialization lock, unless this thread is already inside it.
     *
     * <p>Loading the library re-enters {@link #initialize()} on the same thread: the library's
     * {@code JNI_OnLoad} calls {@code GetFieldID} on {@code LlamaModel}, which initializes that class,
     * whose static block calls {@code initialize()}. The lock is reentrant, so without this guard the
     * nested call ran a second, complete load while the first was still inside {@code System.load}:
     * it deleted the extracted files and, with GPU module jars on the classpath, extracted every one
     * of them again over the library being loaded. Any class but {@code LlamaModel} as the first entry
     * point reached it -- {@code RpcServer} hung its start in the RPC smoke that way. Calls from other threads,
     * and later calls from this one, still run the body; only the nested one returns at once.
     *
     * @param body what to run
     */
    static void runOnceOnThisThread(Runnable body) {
        synchronized (INITIALIZE_LOCK) {
            if (initializing) {
                return;
            }
            initializing = true;
            try {
                body.run();
            } finally {
                initializing = false;
            }
        }
    }

    private static void load() {
        // only cleanup before the first extract
        if (!extracted) {
            cleanup(extractionDirectoriesOfThisClasspath());
        }
        loadNativeLibrary("jllama");
        extracted = true;
    }

    /**
     * Removes what earlier starts left in the temp directory, except what this start is about to
     * reuse: the per-build extraction directories of the natives on this classpath stay, so a second
     * start of the same build finds its files in place (every one is still compared with the jar byte
     * for byte before it is loaded, see {@link #extractFile}) and skips the copy -- 18 files, ~1 s with
     * an on-access scanner on Windows, on every JVM start before. What goes: the flat files of the
     * layout before 5.2.0 ({@code jllama*}, {@code llama*}, {@code ggml*} directly in the temp dir; on
     * Windows a loaded DLL was never removed at VM exit, bug #80) and the extraction directories of
     * other builds once they are {@link #STALE_EXTRACTION_AGE} old -- a younger one may belong to a
     * JVM that is still extracting. Every delete is best-effort: a file a running JVM has loaded is
     * locked on Windows and stays, which is the right outcome.
     *
     * @param keep the extraction directory names this classpath's backends use
     */
    private static void cleanup(Set<String> keep) {
        Instant now = Instant.now();
        try (Stream<Path> dirList = Files.list(getTempDir().toPath())) {
            dirList.filter(path -> isStaleExtraction(path, keep, now)).forEach(LlamaLoader::cleanPath);
        } catch (IOException e) {
            System.err.println("Failed to open directory: " + e.getMessage());
        }
    }

    /**
     * Whether {@link #cleanup(Set)} removes {@code path}: a {@link #shouldCleanPath named} temp entry
     * that is not a per-build extraction directory (the pre-5.2.0 flat layout), or an extraction
     * directory of a build not in {@code keep} whose last modification is more than
     * {@link #STALE_EXTRACTION_AGE} before {@code now}. A directory whose modification time cannot be
     * read is left alone.
     *
     * @param path an entry of the temp directory
     * @param keep the extraction directory names this classpath's backends use
     * @param now  the current time
     * @return whether to delete the entry
     */
    static boolean isStaleExtraction(Path path, Set<String> keep, Instant now) {
        if (!shouldCleanPath(path)) {
            return false;
        }
        Path fileNamePath = path.getFileName();
        String name = fileNamePath == null ? "" : fileNamePath.toString();
        if (!name.startsWith(BACKEND_TEMP_DIR_PREFIX) || !Files.isDirectory(path, LinkOption.NOFOLLOW_LINKS)) {
            return true;
        }
        if (keep.contains(name)) {
            return false;
        }
        try {
            return Files.getLastModifiedTime(path)
                    .toInstant()
                    .plus(STALE_EXTRACTION_AGE)
                    .isBefore(now);
        } catch (IOException e) {
            return false;
        }
    }

    /**
     * The extraction directory names this classpath uses -- one per library backend present, keyed by
     * the library and the modules selected with it -- computed before anything is extracted so the
     * cleanup can spare them.
     *
     * @return the directory names below {@link #getTempDir()}
     */
    static Set<String> extractionDirectoriesOfThisClasspath() {
        Set<String> names = new HashSet<>();
        String nativeResourcePath = getNativeResourcePath();
        List<String> modules = selectModules(presentModules(nativeResourcePath), systemProperties.getBackend());
        for (String backend : LIBRARY_BACKENDS) {
            URL library = resource(nativeResourcePath + "/" + backend + "/" + System.mapLibraryName("jllama"));
            if (library != null) {
                names.add(extractionDirectoryName(backend, library, moduleResources(nativeResourcePath, modules)));
            }
        }
        return names;
    }

    /**
     * The module backends on this classpath: every {@link #MODULE_BACKENDS} directory below {@code
     * nativeResourcePath} that has a {@link #BACKEND_FILES_FILE}.
     *
     * @param nativeResourcePath the platform's resource directory ({@link #getNativeResourcePath()})
     * @return the backend names, in {@link #MODULE_BACKENDS} order
     */
    static List<String> presentModules(String nativeResourcePath) {
        List<String> present = new ArrayList<>();
        for (String backend : MODULE_BACKENDS) {
            if (resource(nativeResourcePath + "/" + backend + "/" + BACKEND_FILES_FILE) != null) {
                present.add(backend);
            }
        }
        return present;
    }

    /**
     * Which of the module backends present on the classpath are extracted next to the library: every
     * one, unless the {@code net.ladenthin.llama.backend} property names some. The property is a
     * comma-separated list of backend names. A module name selects that module, and only the named
     * ones are extracted; a library name ({@code cpu}, {@code metal}) stands for no module at all, so
     * {@code cpu} runs a model on the CPU with every GPU jar still on the classpath. Naming a backend
     * that is not on the classpath, or no backend at all, fails loud -- the property exists to pin a
     * configuration, and a pin that cannot hold is an error, not a fallback. (On Android, where the
     * modules are loaded from the APK by name, the property has no effect.)
     *
     * @param present  the module backends on the classpath ({@link #presentModules})
     * @param property the property value, or {@code null} when unset
     * @return the module backends to extract, in {@link #MODULE_BACKENDS} order
     * @throws UnsatisfiedLinkError when the property names an unknown or absent backend
     */
    static List<String> selectModules(List<String> present, @Nullable String property) {
        if (property == null || property.trim().isEmpty()) {
            return present;
        }
        List<String> selected = new ArrayList<>();
        // String.split's "trailing empties dropped" quirk is benign here: empty entries are skipped.
        @SuppressWarnings("StringSplitter")
        final String[] names = property.split(",");
        for (String raw : names) {
            String name = raw.trim();
            if (name.isEmpty()) {
                continue;
            }
            if (LIBRARY_BACKENDS.contains(name)) {
                continue; // the library is always loaded; naming it only says "no module"
            }
            if (!MODULE_BACKENDS.contains(name)) {
                throw new UnsatisfiedLinkError(String.format(
                        "%s.backend names '%s', which is no backend: the library backends are %s, the GPU module"
                                + " backends %s",
                        LlamaSystemProperties.PREFIX, name, LIBRARY_BACKENDS, MODULE_BACKENDS));
            }
            if (!present.contains(name)) {
                throw new UnsatisfiedLinkError(String.format(
                        "%s.backend names '%s', but no natives jar with that backend is on the classpath for"
                                + " os.name=%s, os.arch=%s (present: %s) -- add net.ladenthin:llama:<version>:%s-<os>-<arch>",
                        LlamaSystemProperties.PREFIX, name, OSInfo.getOSName(), OSInfo.getArchName(), present, name));
            }
            if (!selected.contains(name)) {
                selected.add(name);
            }
        }
        List<String> ordered = new ArrayList<>();
        for (String backend : MODULE_BACKENDS) {
            if (selected.contains(backend)) {
                ordered.add(backend);
            }
        }
        return ordered;
    }

    /**
     * The list-file resources of the given modules, in order -- what keys the extraction directory
     * together with the library (see {@link #extractionDirectoryName(String, URL, Collection)}).
     */
    private static List<URL> moduleResources(String nativeResourcePath, Iterable<String> modules) {
        List<URL> resources = new ArrayList<>();
        for (String module : modules) {
            URL list = resource(nativeResourcePath + "/" + module + "/" + BACKEND_FILES_FILE);
            if (list != null) {
                resources.add(list);
            }
        }
        return resources;
    }

    static boolean shouldCleanPath(Path path) {
        Path fileNamePath = path.getFileName();
        if (fileNamePath == null) {
            return false;
        }
        String fileName = fileNamePath.toString();
        // "ggml" covers the ggml-metal.metal file that initialize() extracts on macOS — it was
        // never matched here, so a stale extracted copy accumulated in the temp dir forever.
        return fileName.startsWith("jllama") || fileName.startsWith("llama") || fileName.startsWith("ggml");
    }

    private static void cleanPath(Path path) {
        try {
            // Backend extractions live in per-backend subdirectories (BACKEND_TEMP_DIR_PREFIX),
            // so directories are cleaned recursively; each delete stays individually best-effort.
            if (Files.isDirectory(path, LinkOption.NOFOLLOW_LINKS)) {
                try (Stream<Path> entries = Files.list(path)) {
                    entries.forEach(LlamaLoader::cleanPath);
                }
            }
            Files.delete(path);
        } catch (IOException | RuntimeException e) {
            System.err.println("Failed to delete old native lib: " + e.getMessage());
        }
    }

    private static void loadNativeLibrary(String name) {
        List<String> triedPaths = new ArrayList<>();

        String nativeLibName = System.mapLibraryName(name);
        String nativeLibPath = systemProperties.getLibPath();
        if (nativeLibPath != null) {
            Path path = Paths.get(nativeLibPath, nativeLibName);
            if (loadNativeLibrary(path)) {
                return;
            } else {
                triedPaths.add(nativeLibPath);
            }
        }

        if (OSInfo.isAndroid()) {
            try {
                // loadLibrary can load directly from packed apk file automatically
                // if java-llama.cpp is added as code source
                System.loadLibrary(name);
                return;
            } catch (UnsatisfiedLinkError e) {
                // Carry the dlopen reason into the final error: "library not in the APK"
                // and "library present but a DT_NEEDED dependency is missing" are
                // indistinguishable without it (the latter shipped once — the Android .so
                // linked libomp.so/libc++_shared.so, which no device has).
                triedPaths.add("Directly from .apk/lib (" + e.getMessage() + ")");
            }
        }

        // Try to load the library from java.library.path
        String javaLibraryPath = System.getProperty("java.library.path", "");
        // String.split's "trailing empties dropped" quirk is benign here because
        // we explicitly skip empty entries with the isEmpty() check below.
        @SuppressWarnings("StringSplitter")
        final String[] ldPaths = javaLibraryPath.split(File.pathSeparator);
        for (String ldPath : ldPaths) {
            if (ldPath.isEmpty()) {
                continue;
            }
            Path path = Paths.get(ldPath, nativeLibName);
            if (loadNativeLibrary(path)) {
                return;
            } else {
                triedPaths.add(ldPath);
            }
        }

        // The natives jars: one directory per backend below <os>/<arch>/. The library comes from
        // the one library backend present; every selected module backend is extracted next to it.
        String nativeResourcePath = getNativeResourcePath();
        String library = null;
        for (String backend : LIBRARY_BACKENDS) {
            if (resource(nativeResourcePath + "/" + backend + "/" + nativeLibName) != null) {
                library = backend;
                break;
            }
            triedPaths.add(nativeResourcePath + "/" + backend);
        }
        if (library == null) {
            throw new UnsatisfiedLinkError(String.format(
                    "No native library found for os.name=%s, os.arch=%s, paths=[%s] -- add the natives jar of"
                            + " this platform (classifier cpu-<os>-<arch>, or metal-macos-aarch64; the pom"
                            + " net.ladenthin:llama-platform names them all) to the classpath, or on the module path"
                            + " together with --add-modules. A GPU natives jar alone is not enough: it holds only its"
                            + " backend module.",
                    OSInfo.getOSName(), OSInfo.getArchName(), String.join(File.pathSeparator, triedPaths)));
        }
        List<String> modules = selectModules(presentModules(nativeResourcePath), systemProperties.getBackend());
        if (!loadFromNativesJars(nativeResourcePath, library, modules)) {
            throw new UnsatisfiedLinkError(String.format(
                    "The native library of backend '%s' could not be loaded for os.name=%s, os.arch=%s"
                            + " (GPU modules: %s) -- see the messages above; paths=[%s]",
                    library,
                    OSInfo.getOSName(),
                    OSInfo.getArchName(),
                    modules.isEmpty() ? "none" : String.join(", ", modules),
                    String.join(File.pathSeparator, triedPaths)));
        }
    }

    /**
     * Loads native library using the given path and name of the library
     *
     * @param path path of the native library
     * @return true for successfully loading, otherwise false
     */
    public static boolean loadNativeLibrary(Path path) {
        if (!Files.exists(path)) {
            return false;
        }
        String absolutePath = path.toAbsolutePath().toString();
        try {
            System.load(absolutePath);
            return true;
        } catch (UnsatisfiedLinkError e) {
            System.err.println(e.getMessage());
            System.err.println("Failed to load native library: " + absolutePath + ". osinfo: "
                    + OSInfo.getNativeLibFolderPathForCurrentOS());
            return false;
        }
    }

    /**
     * Extracts one file from the classpath into {@code targetDirectory}, reusing a byte-identical
     * copy that is already there and replacing one that differs.
     *
     * <p>The reuse branch is what makes a second start of the same build cheap: {@link #cleanup(Set)}
     * leaves the build's own extraction directory in place, so every file an earlier start wrote is
     * found, compared with the jar byte for byte, and loaded without being written again. Nothing is
     * registered for deletion at exit any more -- the files are the cache of the next start.
     *
     * <p>Package-private rather than private so its reuse-vs-replace decision can be driven
     * directly (see {@code LlamaLoaderTest}) — the same convention the other testable statics in
     * this class follow.
     *
     * @param sourceDirectory the classpath resource folder holding {@code fileName}
     * @param fileName        the file to extract
     * @param targetDirectory the directory to extract into; must already exist
     * @return the extracted file, or {@code null} when the resource is absent or extraction failed
     */
    static @Nullable Path extractFile(String sourceDirectory, String fileName, String targetDirectory) {
        String nativeLibraryFilePath = sourceDirectory + "/" + fileName;

        Path extractedFilePath = Paths.get(targetDirectory, fileName);
        // Resolve the File once and reuse it (avoids repeated Path.toFile() calls).
        File extractedFile = extractedFilePath.toFile();

        try {
            // Fast path: a byte-identical copy already exists — extracted by a previous run or by a
            // concurrent JVM sharing this tmpdir. Reuse it rather than rewriting in place: replacing a
            // file another process has already loaded fails on Windows (the lib is locked), and an
            // in-place rewrite risks a partial file a concurrent loader could observe.
            if (Files.exists(extractedFilePath) && resourceMatchesFile(nativeLibraryFilePath, extractedFilePath)) {
                permissionSetter.apply(extractedFile);
                return extractedFilePath;
            }

            // Otherwise extract into a per-attempt unique temp file, verify it, then atomically move it
            // into place so a concurrent loader never observes a half-written library.
            Path tempFile = Files.createTempFile(Paths.get(targetDirectory), fileName + ".", ".tmp");
            try {
                try (InputStream reader = resourceAsStream(nativeLibraryFilePath)) {
                    if (reader == null) {
                        return null;
                    }
                    Files.copy(reader, tempFile, StandardCopyOption.REPLACE_EXISTING);
                }
                if (!resourceMatchesFile(nativeLibraryFilePath, tempFile)) {
                    System.err.println(String.format("Failed to write a native library file at %s", extractedFilePath));
                    return null;
                }
                moveIntoPlace(tempFile, extractedFilePath);
            } finally {
                // Best-effort cleanup: a no-op once moveIntoPlace consumed it, otherwise it removes the
                // temp file left behind if any step above bailed out. The delete must never throw out of
                // the finally block — that would mask the primary result/exception from the try — so the
                // IOException is swallowed (a leftover .tmp in java.io.tmpdir is harmless).
                try {
                    Files.deleteIfExists(tempFile);
                } catch (IOException ignored) {
                    // ignore: best-effort temp-file cleanup
                }
            }

            // Set executable (x) flag to enable Java to load the native library.
            permissionSetter.apply(extractedFile);

            System.err.println("[jllama] extracted '" + fileName + "' to '" + extractedFilePath + "'");
            return extractedFilePath;
        } catch (IOException e) {
            System.err.println(e.getMessage());
            return null;
        }
    }

    /**
     * Atomically replace {@code target} with {@code source}, falling back to a plain move when the
     * filesystem does not support atomic moves.
     */
    private static void moveIntoPlace(Path source, Path target) throws IOException {
        try {
            Files.move(source, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException atomicUnsupported) {
            Files.move(source, target, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    /** Whether the classpath resource at {@code resourcePath} is byte-identical to {@code file}. */
    static boolean resourceMatchesFile(String resourcePath, Path file) throws IOException {
        try (InputStream resource = resourceAsStream(resourcePath);
                InputStream onDisk = Files.newInputStream(file)) {
            if (resource == null) {
                return false;
            }
            return contentsEquals(resource, onDisk);
        }
    }

    /**
     * Parses a {@link #BACKEND_EXTRAS_FILE}.
     *
     * @param reader the file content
     * @return the listed file names in order; blank lines and {@code #} comments are skipped
     * @throws IOException when reading fails
     */
    static List<String> parseExtras(BufferedReader reader) throws IOException {
        List<String> extras = new ArrayList<>();
        String line;
        while ((line = reader.readLine()) != null) {
            String trimmed = line.trim();
            if (!trimmed.isEmpty() && !trimmed.startsWith("#")) {
                extras.add(trimmed);
            }
        }
        return extras;
    }

    /**
     * Reads a file list ({@link #BACKEND_EXTRAS_FILE} or {@link #BACKEND_FILES_FILE}) of a backend
     * directory, if it has one.
     *
     * @param backendResourcePath the backend's classpath directory
     * @param listFile            the list's file name
     * @return the listed file names, or an empty list when the backend has no such list
     * @throws IOException when the file exists but cannot be read
     */
    private static List<String> readFileList(String backendResourcePath, String listFile) throws IOException {
        InputStream stream = resourceAsStream(backendResourcePath + "/" + listFile);
        if (stream == null) {
            return Collections.emptyList();
        }
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8))) {
            return parseExtras(reader);
        }
    }

    /**
     * The three list files of one natives directory: what the JVM loads before the library ({@link
     * #BACKEND_EXTRAS_FILE}), what is only put next to it ({@link #BACKEND_FILES_FILE}) and the build it
     * came from ({@link #BACKEND_BUILD_FILE}).
     */
    private static final class NativesDirectory {
        final String backend;
        final String resourcePath;
        final List<String> extraFiles;
        final List<String> plainFiles;
        final Map<String, String> build;

        NativesDirectory(String backend, String resourcePath) throws IOException {
            this.backend = backend;
            this.resourcePath = resourcePath;
            this.extraFiles = readFileList(resourcePath, BACKEND_EXTRAS_FILE);
            this.plainFiles = readFileList(resourcePath, BACKEND_FILES_FILE);
            this.build = readBuild(resourcePath);
        }
    }

    /**
     * Extracts the library backend and the selected modules into one directory and loads the library.
     *
     * <p>Order: the library's extras are loaded first (on Windows {@code ggml-base.dll} and {@code
     * ggml.dll}, which must be resident before anything that imports them), then the modules' extras
     * (a vendor loader a module ships next to itself), then every plain file -- ggml's CPU modules and
     * the GPU modules -- is put in place without being loaded, and last the library, whose {@code
     * JNI_OnLoad} has ggml load the modules from that directory. Before a byte is written, every module
     * must come from the library's build ({@link #requireSameBuild}) and no two directories may ship a
     * file of the same name.
     *
     * @param nativeResourcePath the platform's resource directory
     * @param library            the library backend ({@link #LIBRARY_BACKENDS})
     * @param modules            the module backends to extract next to it ({@link #selectModules})
     * @return whether the library was loaded
     */
    private static boolean loadFromNativesJars(String nativeResourcePath, String library, List<String> modules) {
        String libraryFileName = System.mapLibraryName("jllama");
        NativesDirectory libraryDir;
        List<NativesDirectory> moduleDirs = new ArrayList<>();
        try {
            libraryDir = new NativesDirectory(library, nativeResourcePath + "/" + library);
            for (String module : modules) {
                moduleDirs.add(new NativesDirectory(module, nativeResourcePath + "/" + module));
            }
        } catch (IOException e) {
            // A list file that is on the classpath but cannot be read is a broken natives jar, not a
            // backend to skip: trying the next library backend would hide it.
            UnsatisfiedLinkError error =
                    new UnsatisfiedLinkError("Failed to read a file list of a natives directory of backend '" + library
                            + "': " + e.getMessage());
            error.initCause(e);
            throw error;
        }
        Map<String, String> claimed = new LinkedHashMap<>();
        claimed.put(libraryFileName, library);
        claimDistinct(claimed, libraryDir);
        for (NativesDirectory moduleDir : moduleDirs) {
            requireSameBuild(libraryDir.backend, libraryDir.build, moduleDir.backend, moduleDir.build);
            claimDistinct(claimed, moduleDir);
        }
        Path targetDirPath = getTempDir()
                .toPath()
                .resolve(extractionDirectoryName(
                        library,
                        resource(libraryDir.resourcePath + "/" + libraryFileName),
                        moduleResources(nativeResourcePath, modules)));
        try {
            Files.createDirectories(targetDirPath);
            // Mark the directory as in use before the first file is touched: cleanup() of a JVM
            // running another build spares a directory younger than STALE_EXTRACTION_AGE, so a start
            // that reuses an old directory is not swept from under it while it compares and loads.
            Files.setLastModifiedTime(targetDirPath, FileTime.from(Instant.now()));
        } catch (IOException e) {
            // The natives jars are the last place the loader looks, so there is nothing to fall back
            // to: say which directory could not be created (net.ladenthin.llama.tmpdir moves it).
            UnsatisfiedLinkError error = new UnsatisfiedLinkError(
                    "Failed to create the natives extraction directory " + targetDirPath + ": " + e.getMessage());
            error.initCause(e);
            throw error;
        }
        // Not registered for deleteOnExit, on purpose: the directory is keyed by the build (and the
        // modules chosen with it) and is reused by the next start of the same configuration, and by
        // other JVMs of it running now; a stale one is removed by a later start's cleanup() instead.
        String targetFolder = targetDirPath.toAbsolutePath().toString();
        List<NativesDirectory> all = new ArrayList<>();
        all.add(libraryDir);
        all.addAll(moduleDirs);
        for (NativesDirectory dir : all) {
            for (String extraFile : dir.extraFiles) {
                Path extraPath = extractFile(dir.resourcePath, extraFile, targetFolder);
                if (extraPath == null || !loadNativeLibrary(extraPath)) {
                    return false;
                }
            }
        }
        int placed = 0;
        for (NativesDirectory dir : all) {
            for (String plainFile : dir.plainFiles) {
                if (extractFile(dir.resourcePath, plainFile, targetFolder) == null) {
                    return false;
                }
                placed++;
            }
        }
        // Only a Metal build that does not embed its shader source ships ggml-metal.metal
        // (every CI build embeds it); ggml looks for it next to the library.
        if (resource(libraryDir.resourcePath + "/" + METAL_SOURCE_FILE) != null) {
            extractFile(libraryDir.resourcePath, METAL_SOURCE_FILE, targetFolder);
        }
        if (!extractAndLoadLibraryFile(libraryDir.resourcePath, libraryFileName, targetFolder)) {
            return false;
        }
        // The modules are extracted, never loaded from here (ggml loads them at JNI_OnLoad), and a
        // module ggml cannot load fails silently -- so at least say what is there (extractFile reports
        // each file it had to write; the others were reused from an earlier start). The native side
        // logs the backends and devices ggml ended up with.
        System.err.println("[jllama] native backend '" + library + "' loaded from " + targetFolder + " with "
                + (modules.isEmpty() ? "no GPU module" : "GPU module(s) " + String.join(", ", modules))
                + (placed == 0 ? "" : "; " + placed + " file(s) in place for ggml to load"));
        return true;
    }

    /**
     * Records the files {@code dir} ships under their owner, refusing a name another directory claimed:
     * all of them land in one extraction directory, so two jars shipping {@code libggml-sycl.so} would
     * overwrite each other.
     */
    private static void claimDistinct(Map<String, String> claimed, NativesDirectory dir) {
        List<String> files = new ArrayList<>(dir.extraFiles);
        files.addAll(dir.plainFiles);
        for (String file : files) {
            String owner = claimed.put(file, dir.backend);
            if (owner != null && !owner.equals(dir.backend)) {
                throw new UnsatisfiedLinkError(String.format(
                        "The natives jars of the backends '%s' and '%s' both ship a file named '%s'; they cannot"
                                + " share one classpath",
                        owner, dir.backend, file));
            }
        }
    }

    /**
     * Refuses to put a module next to a library of another build. The ggml ABI between {@code
     * libggml-base} and a backend module is that of one llama.cpp commit, so the two must record the
     * same {@link #BUILD_KEY_LLAMA_CPP} tag in their {@link #BACKEND_BUILD_FILE}; a module without the
     * file is refused too. Maven makes this hold by itself when every natives jar is taken at one
     * version of {@code net.ladenthin:llama}; this is the backstop for a classpath assembled by hand.
     *
     * @param library      the library backend
     * @param libraryBuild its build file, parsed
     * @param module       the module backend
     * @param moduleBuild  its build file, parsed
     * @throws UnsatisfiedLinkError when the tags differ or one is missing
     */
    static void requireSameBuild(
            String library, Map<String, String> libraryBuild, String module, Map<String, String> moduleBuild) {
        String want = libraryBuild.get(BUILD_KEY_LLAMA_CPP);
        String have = moduleBuild.get(BUILD_KEY_LLAMA_CPP);
        if (want != null && want.equals(have)) {
            return;
        }
        throw new UnsatisfiedLinkError(String.format(
                "The natives jar of the GPU backend '%s' was built from llama.cpp %s, the library of backend '%s'"
                        + " from llama.cpp %s: a backend module runs only next to the library of the same build --"
                        + " use one version of net.ladenthin:llama for every natives jar on the classpath",
                module, describeBuild(have), library, describeBuild(want)));
    }

    private static String describeBuild(@Nullable String tag) {
        return tag == null ? "an unknown build (no " + BACKEND_BUILD_FILE + ")" : tag;
    }

    /**
     * Parses a {@link #BACKEND_BUILD_FILE}: {@code key=value} lines; blank lines, {@code #} comments
     * and lines without {@code =} are skipped.
     *
     * @param reader the file content
     * @return the keys and values, in file order
     * @throws IOException when reading fails
     */
    static Map<String, String> parseBuild(BufferedReader reader) throws IOException {
        Map<String, String> build = new LinkedHashMap<>();
        for (String line : parseExtras(reader)) {
            int separator = line.indexOf('=');
            if (separator > 0) {
                build.put(
                        line.substring(0, separator).trim(),
                        line.substring(separator + 1).trim());
            }
        }
        return build;
    }

    /**
     * Reads the {@link #BACKEND_BUILD_FILE} of a natives directory, if it has one.
     *
     * @param backendResourcePath the backend's classpath directory
     * @return the parsed file, empty when the directory has none
     * @throws IOException when the file exists but cannot be read
     */
    private static Map<String, String> readBuild(String backendResourcePath) throws IOException {
        InputStream stream = resourceAsStream(backendResourcePath + "/" + BACKEND_BUILD_FILE);
        if (stream == null) {
            return Collections.emptyMap();
        }
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8))) {
            return parseBuild(reader);
        }
    }

    /**
     * Name of the directory a backend is extracted into: {@link #BACKEND_TEMP_DIR_PREFIX}, the
     * backend and a key derived from the build of its library ({@link #extractionKey(URL)}).
     *
     * <p>The directory is shared by every JVM using this temp dir, and a backend is many files
     * since {@link #BACKEND_FILES_FILE}: ggml's libraries and one CPU module per instruction-set
     * level. Keyed only by the backend, a JVM running another build of jllama would write its files
     * into a directory a running JVM loaded from. On Windows that replaces exactly the files the
     * first JVM did <em>not</em> lock -- the modules ggml scored and unloaded again, which modules
     * those are depends on the CPU -- and leaves a mixture of two builds behind for the next
     * start; on Linux the running JVM keeps its mapped files, but the extraction of a JVM that is
     * still starting is overtaken half-way. With the key, two builds never share a directory, and
     * two JVMs of the same build still share one (the byte-identical copy is reused, see
     * {@link #extractFile}).
     *
     * @param backend the backend directory name
     * @param library the backend's library resource, or {@code null} when absent
     * @return the directory name below {@link #getTempDir()}
     */
    static String extractionDirectoryName(String backend, @Nullable URL library) {
        return BACKEND_TEMP_DIR_PREFIX + backend + "-" + extractionKey(library);
    }

    /**
     * The extraction directory of a library backend together with the modules put next to it:
     * {@link #extractionDirectoryName(String, URL)} plus {@code -m<hash>} over the modules' own keys.
     * Keyed by the combination, so a start without a GPU jar that an earlier start had never finds that
     * start's module still lying there -- ggml loads every module in the directory, and a file a running
     * JVM has loaded cannot be removed on Windows. Each combination is extracted once and reused.
     *
     * @param backend the library backend
     * @param library the library resource, or {@code null} when absent
     * @param modules the list-file resources of the modules, in order (empty for none)
     * @return the directory name below {@link #getTempDir()}
     */
    static String extractionDirectoryName(String backend, @Nullable URL library, Collection<URL> modules) {
        if (modules.isEmpty()) {
            return extractionDirectoryName(backend, library);
        }
        StringBuilder keys = new StringBuilder();
        for (URL module : modules) {
            keys.append(module.toExternalForm())
                    .append('=')
                    .append(extractionKey(module))
                    .append(';');
        }
        return extractionDirectoryName(backend, library) + "-m"
                + Integer.toHexString(keys.toString().hashCode());
    }

    /**
     * A short key that changes whenever the build of a library resource changes, without reading
     * the library: for a resource inside a jar on disk its CRC-32 and size from the jar's central
     * directory, for a plain file its modification time and size, otherwise a hash of the URL.
     *
     * <p>Not a security measure -- {@link #extractFile} still compares every extracted file with
     * the resource byte for byte -- only what keeps two builds in two directories.
     *
     * @param library the resource URL, or {@code null}
     * @return the key, hexadecimal, {@code none} for a {@code null} URL
     */
    static String extractionKey(@Nullable URL library) {
        if (library == null) {
            return "none";
        }
        try {
            if ("jar".equals(library.getProtocol())) {
                // jar:<URL of the jar>!/<entry>, taken apart by hand: a JarURLConnection would open the
                // jar through the JVM's jar cache and keep it open.
                String form = library.toExternalForm();
                int separator = form.indexOf("!/");
                if (separator > 0) {
                    URL jarUrl = new URL(form.substring("jar:".length(), separator));
                    String entryName = form.substring(separator + 2);
                    if ("file".equals(jarUrl.getProtocol())) {
                        try (JarFile jar = new JarFile(new File(jarUrl.toURI()))) {
                            ZipEntry entry = jar.getEntry(entryName);
                            if (entry != null && entry.getCrc() >= 0 && entry.getSize() >= 0) {
                                return Long.toHexString(entry.getCrc()) + "-" + Long.toHexString(entry.getSize());
                            }
                        }
                    }
                }
            } else if ("file".equals(library.getProtocol())) {
                Path file = Paths.get(library.toURI());
                return Long.toHexString(Files.getLastModifiedTime(file).toMillis()) + "-"
                        + Long.toHexString(Files.size(file));
            }
        } catch (IOException | URISyntaxException | RuntimeException e) {
            // fall through: the URL itself still separates one jar from another
        }
        return Integer.toHexString(library.toExternalForm().hashCode());
    }

    /**
     * Extracts and loads the specified library file to the target folder
     *
     * @param libFolderForCurrentOS Library path.
     * @param libraryFileName       Library name.
     * @param targetFolder          Target folder.
     * @return whether the library was successfully loaded
     */
    private static boolean extractAndLoadLibraryFile(
            String libFolderForCurrentOS, String libraryFileName, String targetFolder) {
        Path path = extractFile(libFolderForCurrentOS, libraryFileName, targetFolder);
        if (path == null) {
            return false;
        }
        return loadNativeLibrary(path);
    }

    static boolean contentsEquals(InputStream in1, InputStream in2) throws IOException {
        if (!(in1 instanceof BufferedInputStream)) {
            in1 = new BufferedInputStream(in1);
        }
        if (!(in2 instanceof BufferedInputStream)) {
            in2 = new BufferedInputStream(in2);
        }

        int ch = in1.read();
        while (ch != -1) {
            int ch2 = in2.read();
            if (ch != ch2) {
                return false;
            }
            ch = in1.read();
        }
        int ch2 = in2.read();
        return ch2 == -1;
    }

    static File getTempDir() {
        String _override = systemProperties.getTmpDir();
        return new File(_override != null ? _override : System.getProperty("java.io.tmpdir"));
    }

    static String getNativeResourcePath() {
        return String.format("%s/%s", NATIVE_RESOURCE_BASE, OSInfo.getNativeLibFolderPathForCurrentOS());
    }

    /**
     * Looks a resource up through this class's {@link ClassLoader}, never through
     * {@link Class#getResource(String)}: on the module path the latter only sees this module's
     * own resources, while the natives live in separate jars (automatic modules).
     *
     * @param name the resource name, relative to the classpath root (no leading slash)
     * @return the resource URL, or {@code null} when absent
     */
    static @Nullable URL resource(String name) {
        return classLoader().getResource(name);
    }

    private static @Nullable InputStream resourceAsStream(String name) {
        return classLoader().getResourceAsStream(name);
    }

    private static ClassLoader classLoader() {
        ClassLoader loader = LlamaLoader.class.getClassLoader();
        return loader != null ? loader : ClassLoader.getSystemClassLoader();
    }
}
