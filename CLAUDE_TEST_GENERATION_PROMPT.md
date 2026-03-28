# Prompt — Regenerate Unit Tests for java-llama.cpp

Use this prompt in a fresh Claude Code session to reproduce all AI-generated
unit tests in this repository.

---

## Prompt

You are working on the `java-llama.cpp` repository — Java JNI bindings for
llama.cpp.  Your task is to find untested Java code and write unit tests for it
**without requiring a model file or native library** (pure Java unit tests only).
Where internal logic is buried in `private` methods, promote those methods to
package-private first so they can be tested directly.

### Step 1 — Understand the existing test style

Read every file under `src/test/java/` and note:
- The test framework in use (JUnit 4 with `org.junit.Assert.*` static imports).
- `@BeforeClass` / `@AfterClass` for shared setup.
- `@Ignore` for tests that need a model file.
- The existing `PairTest` as the gold-standard style reference.
- The `@ClaudeGenerated` annotation defined in
  `src/test/java/de/kherud/llama/ClaudeGenerated.java` — apply it to every
  new test class with a `purpose` string explaining what is being validated.

### Step 2 — Read all untested production classes

Read every `.java` file under `src/main/java/de/kherud/llama/` including the
`args/` sub-package.  Identify code that has **no direct unit tests**:

| Class | Untested behaviour |
|---|---|
| `JsonParameters` / `InferenceParameters` | `toJsonString` escaping, all setters, `setMessages` role validation, collection builders |
| `LlamaOutput` | constructor byte→text decoding, field storage, `toString()` |
| `LlamaException` | message propagation, `RuntimeException` hierarchy |
| `OSInfo` | `translateOSNameToFolderName`, `translateArchNameToFolderName`, arch override property, `getNativeLibFolderPathForCurrentOS` |
| `LlamaLoader` | `shouldCleanPath`, `contentsEquals`, `getTempDir`, `getNativeResourcePath` |
| `ModelParameters` | priority/penalty validation, enum-based setters, sampler formatting, `isDefault`, `CliParameters.toString/toArray` |
| `args/PoolingType` | `getArgValue()` for each constant |
| `args/RopeScalingType` | `getArgValue()` for each constant |

### Step 3 — Promote private statics in `LlamaLoader`

In `src/main/java/de/kherud/llama/LlamaLoader.java` change the following four
methods from `private` to package-private (remove the `private` modifier):

- `shouldCleanPath(Path)`
- `contentsEquals(InputStream, InputStream)`
- `getTempDir()`
- `getNativeResourcePath()`

### Step 4 — Create the following test files

Create each file below.  Follow the existing style exactly: JUnit 4,
`import static org.junit.Assert.*`, no Mockito, no model file required.

#### `src/test/java/de/kherud/llama/InferenceParametersTest.java`
- `testConstructorSetsPrompt` — constructor stores prompt as JSON string
- `testConstructorWithEmptyPrompt`
- `testSetPromptOverrides`
- `testSetNPredict`, `testSetTemperature`, `testSetTopK`, `testSetTopP`,
  `testSetMinP`, `testSetTfsZ`, `testSetTypicalP`, `testSetRepeatLastN`,
  `testSetRepeatPenalty`, `testSetFrequencyPenalty`, `testSetPresencePenalty`,
  `testSetSeed`, `testSetNProbs`, `testSetMinKeep`, `testSetNKeep`,
  `testSetCachePrompt`, `testSetIgnoreEos`, `testSetPenalizeNl`,
  `testSetDynamicTemperatureRange`, `testSetDynamicTemperatureExponent`
  — each verifies the correct map key and string value
- `testSetInputPrefix`, `testSetInputSuffix`, `testSetGrammar`,
  `testSetPenaltyPromptString`, `testSetUseChatTemplate`, `testSetChatTemplate`
- `testSetMiroStatDisabled`, `testSetMiroStatV1`, `testSetMiroStatV2`,
  `testSetMiroStatTau`, `testSetMiroStatEta`
- `testSetStopStringsSingle`, `testSetStopStringsMultiple`, `testSetStopStringsEmpty`
- `testSetSamplersSingle`, `testSetSamplersMultiple`, `testSetSamplersMinP`, `testSetSamplersEmpty`
- `testSetTokenIdBias`, `testSetTokenIdBiasEmpty`
- `testSetTokenBias`, `testSetTokenBiasEmpty`
- `testDisableTokenIds`, `testDisableTokenIdsEmpty`
- `testDisableTokens`, `testDisableTokensEmpty`
- `testSetPenaltyPromptTokenIds`, `testSetPenaltyPromptTokenIdsEmpty`
- `testSetMessagesWithSystemAndUserMessages`,
  `testSetMessagesWithAssistantRole`,
  `testSetMessagesNoSystemMessage`,
  `testSetMessagesEmptySystemMessage`,
  `testSetMessagesInvalidRole` — expects `IllegalArgumentException`,
  `testSetMessagesInvalidRoleOther` — expects `IllegalArgumentException`
- `testToStringContainsPrompt`, `testToStringWithMultipleParams`
- `testToJsonStringEscapesBackslash`, `testToJsonStringEscapesDoubleQuote`,
  `testToJsonStringEscapesNewline`, `testToJsonStringEscapesTab`,
  `testToJsonStringEscapesCarriageReturn`, `testToJsonStringNull`,
  `testToJsonStringEscapesSlashAfterLt`
- `testBuilderChainingReturnsSameInstance`
- `testSetStreamTrue`, `testSetStreamFalse`
- `testSetTokenIdBiasMultiple`

#### `src/test/java/de/kherud/llama/LlamaOutputTest.java`
- `testTextFromBytes` — `new LlamaOutput("hello".getBytes(UTF_8), emptyMap, false)` → text == "hello"
- `testEmptyText`
- `testUtf8MultibyteText` — round-trip with "héllo wörld"
- `testProbabilitiesStored` — map with 2 entries
- `testEmptyProbabilities`
- `testStopFlagFalse`, `testStopFlagTrue`
- `testToStringReturnsText`, `testToStringEmptyText`

#### `src/test/java/de/kherud/llama/LlamaExceptionTest.java`
- `testMessageIsPreserved`
- `testIsRuntimeException`
- `testEmptyMessage`
- `testNullMessage`
- `testCanBeThrown` — use a `boolean caught` flag, **not** `return` inside catch
  (a `return` followed by `fail()` is unreachable and fails to compile under `-source 8`)

#### `src/test/java/de/kherud/llama/OSInfoTest.java`
Use `@Before`/`@After` to save and restore the system property
`"de.kherud.llama.osinfo.architecture"`.
- `testTranslateWindowsXP`, `testTranslateWindows10` → `"Windows"`
- `testTranslateMacOSX`, `testTranslateDarwin` → `"Mac"`
- `testTranslateAIX` → `"AIX"`
- `testTranslateLinuxOnNonMuslNonAndroid` — assert result is one of
  `"Linux"`, `"Linux-Musl"`, `"Linux-Android"` (environment-dependent)
- `testTranslateUnknownOsStripsNonWordChars` — `"Some Unknown OS!"` → `"SomeUnknownOS"`
- `testTranslateArchStripsDots` — `"sparc.64"` → `"sparc64"`
- `testTranslateArchStripsHyphens` — `"aarch-64"` → `"aarch64"`
- `testTranslateArchNoSpecialChars`, `testTranslateArchEmptyString`
- `testGetArchNameWithOverride`, `testGetArchNameWithoutOverrideReturnsNonEmpty`
- `testGetNativeLibFolderPathContainsSlash`, `testGetNativeLibFolderPathHasTwoParts`
- `testIsAndroidRuntimeFalseOnNonAndroid`

#### `src/test/java/de/kherud/llama/LlamaLoaderTest.java`
Use `@Before`/`@After` to save and restore `"de.kherud.llama.tmpdir"`.
- `testShouldCleanPathJllamaPrefix`, `testShouldCleanPathJllamaWithSuffix`
- `testShouldCleanPathLlamaPrefix`, `testShouldCleanPathLlamaWithSuffix`
- `testShouldCleanPathUnrelatedFile`, `testShouldCleanPathEmptyFilename`
- `testShouldCleanPathPartialMatchInMiddle` — `"myjllama.so"` → false
- `testShouldCleanPathCaseSensitive` — `"Jllama.so"` → false
- `testContentsEqualsIdenticalContent`, `testContentsEqualsBothEmpty`
- `testContentsEqualsDifferentContent`, `testContentsEqualsFirstLonger`,
  `testContentsEqualsSecondLonger`
- `testContentsEqualsAlreadyBuffered` — pass `BufferedInputStream` wrappers
- `testContentsEqualsDifferentAtFirstByte`, `testContentsEqualsSingleByteMatch`
- `testGetTempDirDefaultsToJavaIoTmpdir` — compare **`File` objects**, not raw
  strings (Windows `java.io.tmpdir` has a trailing backslash that `File` strips)
- `testGetTempDirUsesOverrideProperty` — set property to
  `new File(tmpdir, "llama-test-custom").getPath()` and compare `File` objects
- `testGetNativeResourcePathStartsWithSlash`,
  `testGetNativeResourcePathContainsPackage`,
  `testGetNativeResourcePathContainsOsAndArch`

#### `src/test/java/de/kherud/llama/ModelParametersTest.java`
- `testSetPriorityValid0`, `testSetPriorityValid3`
- `testSetPriorityNegative` — `@Test(expected = IllegalArgumentException.class)`
- `testSetPriorityTooHigh` — `@Test(expected = IllegalArgumentException.class)`
- `testSetPriorityBatchValid1`, `testSetPriorityBatchNegative`, `testSetPriorityBatchTooHigh`
- `testSetRepeatLastNValidZero`, `testSetRepeatLastNValidMinusOne`, `testSetRepeatLastNValid64`
- `testSetRepeatLastNTooLow` — `@Test(expected = RuntimeException.class)`
- `testSetDryPenaltyLastNValidMinusOne`, `testSetDryPenaltyLastNValidZero`
- `testSetDryPenaltyLastNTooLow` — `@Test(expected = RuntimeException.class)`
- `testSetSamplersSingle`, `testSetSamplersMultiple`, `testSetSamplersEmpty`
- `testSetSamplersAllLowercase` — iterate all `Sampler` values
- `testAddLoraScaledAdapter` — `"adapter.bin,0.5"`
- `testAddControlVectorScaled` — `"vec.bin,1.5"`
- `testSetControlVectorLayerRange` — `"2,10"`, `testSetControlVectorLayerRangeSameStartEnd`
- `testIsDefaultTrueWhenNotSet`, `testIsDefaultFalseWhenSet`, `testIsDefaultFalseAfterFlagOnly`
- `testSetPoolingType`, `testSetRopeScaling`, `testSetCacheTypeKLowercase`,
  `testSetCacheTypeVLowercase`, `testSetSplitModeLowercase`, `testSetNumaLowercase`,
  `testSetMirostatOrdinal`
- `testToStringContainsKey`, `testToStringFlagOnlyNoValue`, `testToStringEmpty`
- `testToArrayEmptyParametersHasOneElement` — `toArray()` always prepends `""`
- `testToArrayScalarParameterHasThreeElements` — `["", "--threads", "4"]`
- `testToArrayFlagOnlyHasTwoElements` — `["", "--embedding"]`
- `testToArrayMultipleParameters`
- `testBuilderChainingReturnsSameInstance`

#### `src/test/java/de/kherud/llama/args/PoolingTypeTest.java`
Test each of `UNSPECIFIED`, `NONE`, `MEAN`, `CLS`, `LAST`, `RANK` → their
`getArgValue()` string.  Also verify `values().length == 6`.
Import `de.kherud.llama.ClaudeGenerated`.

#### `src/test/java/de/kherud/llama/args/RopeScalingTypeTest.java`
Test each of `UNSPECIFIED`, `NONE`, `LINEAR`, `YARN2` (→ `"yarn"`),
`LONGROPE`, `MAX_VALUE` (→ `"maxvalue"`) → their `getArgValue()` string.
Also verify `values().length == 6`.
Import `de.kherud.llama.ClaudeGenerated`.

### Step 5 — Fix `LlamaModelTest` for environments without the model

Add `Assume.assumeTrue("Model file not found, skipping", new java.io.File("models/codellama-7b.Q2_K.gguf").exists())`
as the **first line** of `@BeforeClass setup()` in `LlamaModelTest`.
This turns a hard error into a graceful skip on machines without the model.
Also add `import org.junit.Assume;`.

### Step 6 — Verify compilation

Run `mvn test-compile` and fix any issues:
- Unreachable statements (e.g. `fail()` after a catch that always returns —
  use a `boolean caught` flag instead).
- Windows path comparisons: always compare `File` objects, never raw strings,
  because `java.io.tmpdir` may have a trailing backslash and forward-slash
  literals are rewritten to backslashes by `File`.

---

## Expected outcome

162 new tests across 8 new test classes, zero model-dependent failures.
All tests pass on Linux, macOS, and Windows with `mvn test` when the model
file is absent (LlamaModelTest is skipped, not errored).
