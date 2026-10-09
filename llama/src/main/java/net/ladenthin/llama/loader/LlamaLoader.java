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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
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
 * backend, {@code net/ladenthin/llama/<os>/<arch>/<backend>/}, tried in {@link #BACKEND_PRIORITY}
 * order unless {@code net.ladenthin.llama.backend} forces one.
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
     * The backend directories tried, in this order, when no backend is forced: accelerators
     * first, then the CPU builds. The first one present on the classpath whose library loads
     * wins; a missing vendor runtime fails its load cleanly and the next one is tried. Every
     * build carries the CPU backend, so a GPU library that loads still runs a model on the CPU
     * with {@code -ngl 0}. {@code msvc} precedes {@code cpu} because nobody adds the MSVC
     * natives jar except to use it.
     */
    static final List<String> BACKEND_PRIORITY = Collections.unmodifiableList(Arrays.asList(
            "cuda13",
            "rocm",
            "sycl-fp16",
            "sycl-fp32",
            "sycl",
            "vulkan",
            "opencl",
            "openvino",
            "metal",
            "msvc",
            "cpu"));

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
     * backend modules it loads itself from its own directory (one CPU module per instruction-set
     * level in a {@code JLLAMA_CPU_VARIANTS} build, of which ggml picks the best for the running
     * CPU). Loading them from Java would defeat exactly that choice.
     */
    static final String BACKEND_FILES_FILE = "jllama-files.txt";

    /**
     * Prefix of the per-backend extraction subdirectory below the temp dir,
     * {@code jllama-backend-<backend>-<key>} (see {@link #extractionDirectoryName}). Deliberately
     * starts with {@code jllama} so {@link #shouldCleanPath(Path)} matches it during cleanup.
     */
    static final String BACKEND_TEMP_DIR_PREFIX = "jllama-backend-";

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
     * it deleted the extracted files and, with a multi-backend jar, probed every backend again and
     * extracted over the library being loaded. Any class but {@code LlamaModel} as the first entry
     * point reached it -- {@code RpcServer} hung its fat-jar start that way. Calls from other threads,
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
            cleanup();
        }
        loadNativeLibrary("jllama");
        extracted = true;
    }

    /**
     * Deleted old native libraries e.g. on Windows the DLL file is not removed on VM-Exit (bug #80)
     */
    private static void cleanup() {
        try (Stream<Path> dirList = Files.list(getTempDir().toPath())) {
            dirList.filter(LlamaLoader::shouldCleanPath).forEach(LlamaLoader::cleanPath);
        } catch (IOException e) {
            System.err.println("Failed to open directory: " + e.getMessage());
        }
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

        // The natives jars: one directory per backend below <os>/<arch>/. A forced backend is
        // the only candidate and fails loud; otherwise every backend present on the classpath
        // is tried in priority order.
        String forced = systemProperties.getBackend();
        List<String> candidates = forced != null ? Collections.singletonList(forced) : BACKEND_PRIORITY;
        String nativeResourcePath = getNativeResourcePath();
        Set<String> residentExtraFiles = new HashSet<>();
        for (String backend : candidates) {
            String backendResourcePath = nativeResourcePath + "/" + backend;
            if (resource(backendResourcePath + "/" + nativeLibName) == null) {
                continue;
            }
            if (tryLoadBackend(backendResourcePath, backend, residentExtraFiles)) {
                System.err.println("[jllama] using native backend '" + backend + "'");
                return;
            }
            triedPaths.add(backendResourcePath);
        }
        if (forced != null) {
            throw new UnsatisfiedLinkError(String.format(
                    "Forced native backend '%s' (%s.backend) could not be loaded for os.name=%s, os.arch=%s,"
                            + " paths=[%s]",
                    forced,
                    LlamaSystemProperties.PREFIX,
                    OSInfo.getOSName(),
                    OSInfo.getArchName(),
                    String.join(File.pathSeparator, triedPaths)));
        }

        throw new UnsatisfiedLinkError(String.format(
                "No native library found for os.name=%s, os.arch=%s, paths=[%s] -- add a natives jar for"
                        + " this platform (e.g. classifier cpu-<os>-<arch>) to the classpath, or on the module path together"
                        + " with --add-modules",
                OSInfo.getOSName(), OSInfo.getArchName(), String.join(File.pathSeparator, triedPaths)));
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
     * <p>Package-private rather than private so its reuse-vs-replace decision can be driven
     * directly (see {@code LlamaLoaderTest}) — the same convention the other testable statics in
     * this class follow. Going through {@link #initialize()} cannot reach that decision: the
     * cleanup pass it runs first deletes exactly the {@code jllama*} temp paths a test would have
     * to seed, so the branch is never taken.
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
                extractedFile.deleteOnExit();
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
            extractedFile.deleteOnExit();

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
     * Attempts to extract and load one backend from its resource directory into a per-backend
     * temp subdirectory (backends share file names, so they must not overwrite each other's
     * extractions).
     *
     * @param backendResourcePath the backend's classpath directory
     * @param backend             the backend directory name
     * @param residentExtraFiles  file names of extra modules already loaded by earlier failed
     *                            attempts; updated with this attempt's loaded extras. A native
     *                            module cannot be unloaded, and imports bind by module name, so a
     *                            backend declaring an extra file that is already resident from
     *                            another backend must be skipped to avoid cross-wiring.
     * @return whether the backend's main library was successfully loaded
     */
    private static boolean tryLoadBackend(String backendResourcePath, String backend, Set<String> residentExtraFiles) {
        List<String> extraFiles;
        List<String> plainFiles;
        try {
            extraFiles = readFileList(backendResourcePath, BACKEND_EXTRAS_FILE);
            plainFiles = readFileList(backendResourcePath, BACKEND_FILES_FILE);
        } catch (IOException e) {
            System.err.println("Failed to read a file list of " + backendResourcePath + ": " + e.getMessage());
            return false;
        }
        for (String extraFile : extraFiles) {
            if (residentExtraFiles.contains(extraFile)) {
                System.err.println("[jllama] skipping backend '" + backend + "': module '" + extraFile
                        + "' is already resident from a previously failed backend attempt");
                return false;
            }
        }
        String libraryFileName = System.mapLibraryName("jllama");
        Path targetDirPath = getTempDir()
                .toPath()
                .resolve(extractionDirectoryName(backend, resource(backendResourcePath + "/" + libraryFileName)));
        try {
            Files.createDirectories(targetDirPath);
        } catch (IOException e) {
            System.err.println("Failed to create backend temp directory " + targetDirPath + ": " + e.getMessage());
            return false;
        }
        // Registered before the contained files: File.deleteOnExit processing is LIFO, so the
        // files registered afterwards by extractFile are deleted first, then this directory.
        targetDirPath.toFile().deleteOnExit();
        String targetFolder = targetDirPath.toAbsolutePath().toString();
        for (String extraFile : extraFiles) {
            Path extraPath = extractFile(backendResourcePath, extraFile, targetFolder);
            if (extraPath == null || !loadNativeLibrary(extraPath)) {
                return false;
            }
            residentExtraFiles.add(extraFile);
        }
        for (String plainFile : plainFiles) {
            if (extractFile(backendResourcePath, plainFile, targetFolder) == null) {
                return false;
            }
        }
        // Only a Metal build that does not embed its shader source ships ggml-metal.metal
        // (every CI build embeds it); ggml looks for it next to the library.
        if (resource(backendResourcePath + "/" + METAL_SOURCE_FILE) != null) {
            extractFile(backendResourcePath, METAL_SOURCE_FILE, targetFolder);
        }
        return extractAndLoadLibraryFile(backendResourcePath, libraryFileName, targetFolder);
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
