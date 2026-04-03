/*--------------------------------------------------------------------------
 *  Copyright 2007 Taro L. Saito
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *--------------------------------------------------------------------------*/

package de.kherud.llama;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Stream;

import org.jetbrains.annotations.Nullable;

/**
 * Set the system properties, de.kherud.llama.lib.path, de.kherud.llama.lib.name, appropriately so that the
 * library can find *.dll, *.dylib and *.so files, according to the current OS (win, linux, mac).
 *
 * <p>The library files are automatically extracted from this project's package (JAR).
 *
 * <p>usage: call {@link #initialize()} before using the library.
 *
 * @author leo
 */
@SuppressWarnings("UseOfSystemOutOrSystemErr")
class LlamaLoader {

	private static boolean extracted = false;

	/**
	 * Loads the llama and jllama shared libraries
	 */
	static synchronized void initialize() throws UnsatisfiedLinkError {
		// only cleanup before the first extract
		if (!extracted) {
			cleanup();
		}
		if ("Mac".equals(OSInfo.getOSName())) {
			String nativeDirName = getNativeResourcePath();
			String tempFolder = getTempDir().getAbsolutePath();
			System.out.println(nativeDirName);
			Path metalFilePath = extractFile(nativeDirName, "ggml-metal.metal", tempFolder, false);
			if (metalFilePath == null) {
				System.err.println("'ggml-metal.metal' not found");
			}
		}

		// On Windows, extract OpenSSL/BoringSSL DLLs before loading jllama
		// This ensures jllama.dll can find its dependencies
		if ("Windows".equals(OSInfo.getOSName())) {
			String nativeDirName = getNativeResourcePath();
			String tempFolder = getTempDir().getAbsolutePath();
			extractOpenSSLDlls(nativeDirName, tempFolder);
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
		}
		catch (IOException e) {
			System.err.println("Failed to open directory: " + e.getMessage());
		}
	}

	static boolean shouldCleanPath(Path path) {
		String fileName = path.getFileName().toString();
		return fileName.startsWith("jllama") || fileName.startsWith("llama") ||
			   fileName.startsWith("libssl") || fileName.startsWith("libcrypto");
	}

	/**
	 * Extracts OpenSSL/BoringSSL library dependencies across all platforms
	 * These are required by jllama at runtime
	 *
	 * @param nativeDirName The native resource directory path
	 * @param tempFolder    The temporary directory for extraction
	 */
	private static void extractOpenSSLDlls(String nativeDirName, String tempFolder) {
		// Look for libssl and libcrypto libraries (from BoringSSL/LibreSSL)
		// Windows: .dll (runtime), .lib (import library)
		// Linux: .a (static), .so (shared)
		// macOS: .dylib (dynamic), .a (static)
		String[] sslLibraries = {
			// Windows DLLs
			"libssl-3-x64.dll",
			"libssl-3-x86.dll",
			"libssl-3.dll",
			"libcrypto-3-x64.dll",
			"libcrypto-3-x86.dll",
			"libcrypto-3.dll",
			// Windows import libraries
			"ssl.lib",
			"crypto.lib",
			// Linux static libraries
			"libssl.a",
			"libcrypto.a",
			// Linux shared libraries
			"libssl.so",
			"libcrypto.so",
			// macOS dynamic libraries
			"libssl.dylib",
			"libcrypto.dylib"
		};

		for (String libName : sslLibraries) {
			Path extractedPath = extractFile(nativeDirName, libName, tempFolder, false);
			if (extractedPath != null) {
				// Successfully extracted, don't spam errors for missing optional files
				continue;
			}
			// File not found is expected - only the libraries that were built will exist
		}
	}

	private static void cleanPath(Path path) {
		try {
			Files.delete(path);
		}
		catch (Exception e) {
			System.err.println("Failed to delete old native lib: " + e.getMessage());
		}
	}

	private static void loadNativeLibrary(String name) {
		List<String> triedPaths = new LinkedList<>();

		String nativeLibName = System.mapLibraryName(name);
		String nativeLibPath = System.getProperty("de.kherud.llama.lib.path");
		if (nativeLibPath != null) {
			Path path = Paths.get(nativeLibPath, nativeLibName);
			if (loadNativeLibrary(path)) {
				return;
			}
			else {
				triedPaths.add(nativeLibPath);
			}
		}

		if (OSInfo.isAndroid()) {
			try {
				// loadLibrary can load directly from packed apk file automatically
				// if java-llama.cpp is added as code source
				System.loadLibrary(name);
				return;
			}
			catch (UnsatisfiedLinkError e) {
				triedPaths.add("Directly from .apk/lib");
			}
		}

		// Try to load the library from java.library.path
		String javaLibraryPath = System.getProperty("java.library.path", "");
		for (String ldPath : javaLibraryPath.split(File.pathSeparator)) {
			if (ldPath.isEmpty()) {
				continue;
			}
			Path path = Paths.get(ldPath, nativeLibName);
			if (loadNativeLibrary(path)) {
				return;
			}
			else {
				triedPaths.add(ldPath);
			}
		}

		// As a last resort try load the os-dependent library from the jar file
		nativeLibPath = getNativeResourcePath();
		if (hasNativeLib(nativeLibPath, nativeLibName)) {
			// temporary library folder
			String tempFolder = getTempDir().getAbsolutePath();
			// Try extracting the library from jar
			if (extractAndLoadLibraryFile(nativeLibPath, nativeLibName, tempFolder)) {
				return;
			}
			else {
				triedPaths.add(nativeLibPath);
			}
		}

		throw new UnsatisfiedLinkError(
				String.format(
						"No native library found for os.name=%s, os.arch=%s, paths=[%s]",
						OSInfo.getOSName(),
						OSInfo.getArchName(),
						String.join(File.pathSeparator, triedPaths)
				)
		);
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
		}
		catch (UnsatisfiedLinkError e) {
			System.err.println(e.getMessage());
			System.err.println("Failed to load native library: " + absolutePath + ". osinfo: " + OSInfo.getNativeLibFolderPathForCurrentOS());
			return false;
		}
	}

	@Nullable
	private static Path extractFile(String sourceDirectory, String fileName, String targetDirectory, boolean addUuid) {
		String nativeLibraryFilePath = sourceDirectory + "/" + fileName;

		Path extractedFilePath = Paths.get(targetDirectory, fileName);

		try {
			// Extract a native library file into the target directory
			try (InputStream reader = LlamaLoader.class.getResourceAsStream(nativeLibraryFilePath)) {
				if (reader == null) {
					return null;
				}
				Files.copy(reader, extractedFilePath, StandardCopyOption.REPLACE_EXISTING);
			}
			finally {
				// Delete the extracted lib file on JVM exit.
				extractedFilePath.toFile().deleteOnExit();
			}

			// Set executable (x) flag to enable Java to load the native library
			extractedFilePath.toFile().setReadable(true);
			extractedFilePath.toFile().setWritable(true, true);
			extractedFilePath.toFile().setExecutable(true);

			// Check whether the contents are properly copied from the resource folder
			try (InputStream nativeIn = LlamaLoader.class.getResourceAsStream(nativeLibraryFilePath);
				 InputStream extractedLibIn = Files.newInputStream(extractedFilePath)) {
				if (!contentsEquals(nativeIn, extractedLibIn)) {
					throw new RuntimeException(String.format("Failed to write a native library file at %s", extractedFilePath));
				}
			}

			System.out.println("Extracted '" + fileName + "' to '" + extractedFilePath + "'");
			return extractedFilePath;
		}
		catch (IOException e) {
			System.err.println(e.getMessage());
			return null;
		}
	}

	/**
	 * Extracts and loads the specified library file to the target folder
	 *
	 * @param libFolderForCurrentOS Library path.
	 * @param libraryFileName       Library name.
	 * @param targetFolder          Target folder.
	 * @return whether the library was successfully loaded
	 */
	private static boolean extractAndLoadLibraryFile(String libFolderForCurrentOS, String libraryFileName, String targetFolder) {
		Path path = extractFile(libFolderForCurrentOS, libraryFileName, targetFolder, true);
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
		return new File(System.getProperty("de.kherud.llama.tmpdir", System.getProperty("java.io.tmpdir")));
	}

	static String getNativeResourcePath() {
		String packagePath = LlamaLoader.class.getPackage().getName().replace(".", "/");
		return String.format("/%s/%s", packagePath, OSInfo.getNativeLibFolderPathForCurrentOS());
	}

	private static boolean hasNativeLib(String path, String libraryName) {
		return LlamaLoader.class.getResource(path + "/" + libraryName) != null;
	}
}
