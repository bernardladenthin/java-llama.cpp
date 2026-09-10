// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.android.llmservice

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import java.io.File
import java.time.LocalTime
import java.time.temporal.ChronoUnit
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import net.ladenthin.llama.LlamaModel
import net.ladenthin.llama.kotlin.generateChatFlow
import net.ladenthin.llama.parameters.InferenceParameters
import net.ladenthin.llama.parameters.ModelParameters
import net.ladenthin.llama.value.ChatMessage
import net.ladenthin.llama.value.ContentPart

/**
 * File name of the SAF-picked model copied into `filesDir`. Transient working data — wiped on cold
 * start (and best-effort on close) by [LlmServiceApp] so nothing lingers between sessions.
 */
const val MODEL_COPY_NAME: String = "current-model.gguf"

/**
 * File name of the SAF-picked vision projector (mmproj) copied into `filesDir`. Same transient
 * lifecycle as [MODEL_COPY_NAME] — wiped on cold start and on [ChatViewModel.unloadModel].
 */
const val MMPROJ_COPY_NAME: String = "current-mmproj.gguf"

/**
 * Holds the loaded [LlamaModel] and drives streaming chat over the llama-kotlin
 * [generateChatFlow] facade. All native work runs on [Dispatchers.IO]; the UI observes
 * [uiState]. This is the whole "brain" of the app — the Compose layer is pure rendering.
 *
 * User-facing strings live in resources (localized); the UI resolves them and passes the
 * system prompt into [send], and maps [Notice] / [ErrorInfo] to localized text — so the
 * ViewModel stays free of the active locale.
 */
class ChatViewModel(application: Application) : AndroidViewModel(application) {

    /**
     * One chat turn. [text] grows token-by-token while an assistant reply streams. A user turn may
     * carry an image ([imageBytes] + [imageMimeType]) attached via the 🖼️ button; both are null for
     * every other turn. Images are session-transient — [SessionStore] persists text only.
     */
    data class Message(
        val role: String,
        val text: String,
        val imageBytes: ByteArray? = null,
        val imageMimeType: String? = null,
    )

    /** An image picked via the attach button, staged in memory until [send] consumes it. */
    data class PendingImage(val bytes: ByteArray, val mimeType: String, val displayName: String)

    /** Lifecycle of the model load. */
    enum class ModelState { NONE, LOADING, READY }

    /** One-shot user notice (mapped to a localized string + shown as a snackbar by the UI). */
    enum class Notice { SESSION_SAVED, SESSION_LOADED, NO_SESSION }

    /** Which localized error template the UI should format [ErrorInfo.detail] into. */
    enum class ErrorType { LOAD, GENERATION }

    /** A localized-at-the-UI error: the [type] selects the template, [detail] fills it in. */
    data class ErrorInfo(val type: ErrorType, val detail: String)

    /**
     * Sampling / generation knobs surfaced in the Settings sheet. The defaults are chosen to give
     * a coherent reply on small on-device models: a [repeatPenalty] > 1 with a non-zero
     * [repeatLastN] is what stops the degenerate "…spezifischen spezifischen…" repetition loop that
     * a bare greedy/no-penalty decode falls into (the reported SmolVLM-500M behaviour). All are
     * forwarded verbatim to [InferenceParameters]; see [GenerationSettings.DEFAULT].
     */
    data class GenerationSettings(
        val temperature: Float = 0.7f,
        val topK: Int = 40,
        val topP: Float = 0.95f,
        val minP: Float = 0.05f,
        val repeatPenalty: Float = 1.1f,
        val repeatLastN: Int = 64,
        val maxTokens: Int = 256,
    ) {
        companion object {
            /** The shipped defaults (also the "Reset" target). */
            val DEFAULT = GenerationSettings()
        }
    }

    /**
     * Model **load-time** parameters (unlike [GenerationSettings], which apply per request). Changing
     * these only takes effect when the model is (re)loaded — see [reloadModel].
     */
    data class ModelConfig(
        val threads: Int = DEFAULT_THREADS,
        val contextSize: Int = DEFAULT_CONTEXT_SIZE,
    ) {
        companion object {
            /** A portable CPU-thread default; users tune it in Settings. */
            const val DEFAULT_THREADS = 4

            /** Default context window (tokens). */
            const val DEFAULT_CONTEXT_SIZE = 2048

            /** The shipped defaults. */
            val DEFAULT = ModelConfig()
        }
    }

    /** Immutable snapshot the Compose UI renders. */
    data class UiState(
        val modelState: ModelState = ModelState.NONE,
        val modelName: String? = null,
        val messages: List<Message> = emptyList(),
        val generating: Boolean = false,
        val error: ErrorInfo? = null,
        val notice: Notice? = null,
        val settings: GenerationSettings = GenerationSettings.DEFAULT,
        val modelConfig: ModelConfig = ModelConfig.DEFAULT,
        /** Display name of the selected vision projector (mmproj), or null when text-only. */
        val visionModelName: String? = null,
        /** An image staged via the attach button, waiting for the next [send]. */
        val pendingImage: PendingImage? = null,
    )

    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    /**
     * Rolling in-app log (newest last), shown as a one-line strip at the bottom and in full in the
     * log viewer. Kept separate from [uiState] so per-token chat updates don't re-emit the whole
     * log list. Capped at [MAX_LOG_LINES]; entries are prefixed with a local wall-clock time.
     */
    private val _log = MutableStateFlow<List<String>>(emptyList())
    val log: StateFlow<List<String>> = _log.asStateFlow()

    private var model: LlamaModel? = null
    private var modelPath: String? = null
    private var mmprojPath: String? = null
    private var generation: Job? = null

    /**
     * Optional chat-template override (e.g. "chatml") for GGUFs that ship no template of their
     * own. Left null for real instruct models (they carry their own template); the instrumented
     * test sets it because the tiny CI draft model has none.
     */
    private var chatTemplate: String? = null

    private companion object {
        const val MAX_LOG_LINES = 500
    }

    /** Replaces the model load-time config (threads / context). Applies on the next [reloadModel]. */
    fun updateModelConfig(config: ModelConfig) {
        _uiState.update { it.copy(modelConfig = config) }
    }

    /**
     * Reopens the currently loaded model with the current [ModelConfig] (threads / context), so a
     * changed thread count or context size takes effect. No-op if no model is loaded or busy.
     */
    fun reloadModel() {
        val path = modelPath
        if (path == null || _uiState.value.generating || _uiState.value.modelState == ModelState.LOADING) {
            return
        }
        val name = _uiState.value.modelName ?: File(path).name
        val template = chatTemplate
        _uiState.update { it.copy(modelState = ModelState.LOADING, error = null) }
        viewModelScope.launch { openModel(path, name, template) }
    }

    /** Appends one timestamped line to the rolling [log] (trimmed to [MAX_LOG_LINES]). */
    private fun log(line: String) {
        val ts = LocalTime.now().truncatedTo(ChronoUnit.SECONDS)
        _log.update { (it + "$ts  $line").takeLast(MAX_LOG_LINES) }
    }

    /** Replaces the sampling settings (from the Settings sheet). */
    fun updateSettings(settings: GenerationSettings) {
        _uiState.update { it.copy(settings = settings) }
    }

    /** Restores the shipped default sampling settings. */
    fun resetSettings() {
        _uiState.update { it.copy(settings = GenerationSettings.DEFAULT) }
        log("Settings reset to defaults")
    }

    /** Clears the in-app log buffer. */
    fun clearLog() {
        _log.value = emptyList()
    }

    /**
     * Copies a SAF-picked GGUF (a `content://` URI, which llama.cpp cannot mmap) into app-internal
     * storage, then loads it by real path. Big models take a while to copy; the UI shows LOADING.
     */
    fun loadModelFromUri(uri: Uri, displayName: String) {
        _uiState.update { it.copy(modelState = ModelState.LOADING, error = null) }
        viewModelScope.launch {
            try {
                val path = withContext(Dispatchers.IO) { copyUriToInternal(uri, MODEL_COPY_NAME) }
                openModel(path, displayName, template = null)
            } catch (t: Throwable) {
                _uiState.update {
                    it.copy(modelState = ModelState.NONE, error = ErrorInfo(ErrorType.LOAD, t.message ?: "I/O error"))
                }
            }
        }
    }

    /**
     * Copies a SAF-picked vision projector (mmproj) GGUF into app-internal storage and records its
     * path. Takes effect the next time a model is (re)loaded — pick it before or after the main
     * model, in either order. No-op error path mirrors [loadModelFromUri].
     */
    fun loadMmprojFromUri(uri: Uri, displayName: String) {
        viewModelScope.launch {
            try {
                val path = withContext(Dispatchers.IO) { copyUriToInternal(uri, MMPROJ_COPY_NAME) }
                mmprojPath = path
                _uiState.update { it.copy(visionModelName = displayName, error = null) }
                log("Vision model set: $displayName")
            } catch (t: Throwable) {
                _uiState.update { it.copy(error = ErrorInfo(ErrorType.LOAD, t.message ?: "I/O error")) }
            }
        }
    }

    /** Clears the selected vision projector; the next model load is text-only again. */
    fun clearMmproj() {
        mmprojPath = null
        _uiState.update { it.copy(visionModelName = null) }
        log("Vision model cleared")
        viewModelScope.launch(Dispatchers.IO) {
            File(getApplication<Application>().filesDir, MMPROJ_COPY_NAME).delete()
        }
    }

    /**
     * Stages a SAF-picked image as the pending attachment for the next [send]. Read fully into
     * memory rather than copied to [android.content.Context.getFilesDir]: unlike the model/mmproj
     * GGUFs (which llama.cpp must mmap by real path),
     * [net.ladenthin.llama.value.ContentPart.imageBytes] consumes raw bytes directly.
     */
    fun attachImage(uri: Uri, displayName: String) {
        viewModelScope.launch {
            try {
                val resolver = getApplication<Application>().contentResolver
                val (mimeType, bytes) = withContext(Dispatchers.IO) {
                    val type = resolver.getType(uri) ?: "image/jpeg"
                    val data = resolver.openInputStream(uri)?.use { it.readBytes() }
                        ?: throw java.io.IOException("content resolver returned no stream for $uri")
                    type to data
                }
                _uiState.update { it.copy(pendingImage = PendingImage(bytes, mimeType, displayName), error = null) }
                log("Image attached: $displayName")
            } catch (t: Throwable) {
                _uiState.update { it.copy(error = ErrorInfo(ErrorType.LOAD, t.message ?: "I/O error")) }
            }
        }
    }

    /** Removes the pending image attachment before it's sent. */
    fun clearPendingImage() {
        _uiState.update { it.copy(pendingImage = null) }
    }

    /**
     * Loads a model already present at an absolute on-device path (no copy). Used by the
     * instrumented test against the adb-pushed model and by session restore.
     */
    fun loadModelFromPath(path: String, template: String? = null) {
        _uiState.update { it.copy(modelState = ModelState.LOADING, error = null) }
        viewModelScope.launch {
            openModel(path, File(path).name, template)
        }
    }

    private suspend fun openModel(path: String, displayName: String, template: String?) {
        val config = _uiState.value.modelConfig
        val mmproj = mmprojPath
        log("Loading model: $displayName")
        try {
            val loaded = withContext(Dispatchers.IO) {
                val parameters = ModelParameters()
                    .setModel(path)
                    .setCtxSize(config.contextSize)
                    .setThreads(config.threads)
                    .setGpuLayers(0) // CPU-only: portable across every device
                // The chat template is a LOAD-time option. It used to be passed per request via
                // InferenceParameters.withChatTemplate, which llama.cpp's request schema discarded
                // without a word -- the MODEL_PATH/CHAT_TEMPLATE test hook silently had no effect.
                if (template != null) {
                    parameters.setChatTemplate(template)
                }
                if (mmproj != null) {
                    // Mirrors the CPU-only + mmproj config validated by MultimodalIntegrationTest:
                    // no GPU device selection, no mmproj offload attempt.
                    parameters.setMmproj(mmproj).setDevices("none").setMmprojOffload(false)
                }
                LlamaModel(parameters)
            }
            withContext(Dispatchers.IO) { model?.close() }
            model = loaded
            modelPath = path
            chatTemplate = template
            _uiState.update {
                it.copy(modelState = ModelState.READY, modelName = displayName, error = null)
            }
            log(
                "Model ready: $displayName (ctx=${config.contextSize}, threads=${config.threads}, " +
                    "CPU${if (mmproj != null) ", vision" else ""})",
            )
        } catch (t: Throwable) {
            log("Load failed: ${t.message ?: "load failed"}")
            _uiState.update {
                it.copy(modelState = ModelState.NONE, error = ErrorInfo(ErrorType.LOAD, t.message ?: "load failed"))
            }
        }
    }

    /**
     * Sends a user turn and streams the assistant reply into [uiState]. No-op if busy or no model.
     *
     * @param userText the message typed by the user
     * @param systemPrompt the localized system prompt (resolved from resources by the UI)
     */
    fun send(userText: String, systemPrompt: String) {
        val text = userText.trim()
        val active = model
        val attachment = _uiState.value.pendingImage
        if ((text.isEmpty() && attachment == null) || active == null || _uiState.value.generating) {
            return
        }
        val userMessage = Message(
            role = "user",
            text = text,
            imageBytes = attachment?.bytes,
            imageMimeType = attachment?.mimeType,
        )
        _uiState.update { it.copy(pendingImage = null) }
        startGeneration(active, _uiState.value.messages + userMessage, systemPrompt)
    }

    /**
     * Regenerates the reply to the most recent user turn: drops the trailing assistant reply and
     * re-runs generation from the conversation up to and including that user turn. No-op if busy,
     * no model, or there is no user turn yet. Useful after tuning the sampling settings.
     *
     * @param systemPrompt the localized system prompt (resolved from resources by the UI)
     */
    fun regenerate(systemPrompt: String) {
        val active = model
        if (active == null || _uiState.value.generating) {
            return
        }
        val messages = _uiState.value.messages
        val lastUserIdx = messages.indexOfLast { it.role == "user" }
        if (lastUserIdx < 0) {
            return
        }
        log("Regenerating reply")
        startGeneration(active, messages.subList(0, lastUserIdx + 1).toList(), systemPrompt)
    }

    /**
     * Shared streaming path for [send] and [regenerate]. [history] must already end with the user
     * turn to answer; this appends the empty assistant message that grows as tokens arrive.
     */
    private fun startGeneration(active: LlamaModel, history: List<Message>, systemPrompt: String) {
        _uiState.update { it.copy(messages = history + Message("assistant", ""), generating = true, error = null) }

        val s = _uiState.value.settings
        generation = viewModelScope.launch {
            val chatMessages = mutableListOf<ChatMessage>()
            if (systemPrompt.isNotEmpty()) {
                chatMessages.add(ChatMessage("system", systemPrompt))
            }
            history.forEach { chatMessages.add(it.toChatMessage()) }
            // Apply the sampling knobs. repeatPenalty (> 1) + repeatLastN are the ones that break
            // the degenerate repetition loop small on-device models fall into with a bare decode.
            var params = InferenceParameters("")
                .withMessages(chatMessages)
                .withNPredict(s.maxTokens)
                .withTemperature(s.temperature)
                .withTopK(s.topK)
                .withTopP(s.topP)
                .withMinP(s.minP)
                .withRepeatPenalty(s.repeatPenalty)
                .withRepeatLastN(s.repeatLastN)

            log("Generating (temp=${s.temperature}, repeat=${s.repeatPenalty}/${s.repeatLastN}, maxTokens=${s.maxTokens})")
            val reply = StringBuilder()
            try {
                active.generateChatFlow(params)
                    .flowOn(Dispatchers.IO)
                    .collect { output ->
                        reply.append(output.text)
                        _uiState.update { state -> state.replaceLastAssistant(reply.toString()) }
                    }
                log("Reply complete (${reply.length} chars)")
            } catch (t: CancellationException) {
                // User pressed Stop: keep the partial reply, no error. Rethrow to honour cancellation.
                _uiState.update { state -> state.replaceLastAssistant(reply.toString()) }
                throw t
            } catch (t: Throwable) {
                log("Generation failed: ${t.message ?: "generation failed"}")
                _uiState.update { state ->
                    state.replaceLastAssistant(reply.toString())
                        .copy(error = ErrorInfo(ErrorType.GENERATION, t.message ?: "generation failed"))
                }
            } finally {
                _uiState.update { it.copy(generating = false) }
            }
        }
    }

    /** Cancels an in-flight generation, keeping whatever partial reply has streamed so far. */
    fun stopGeneration() {
        if (_uiState.value.generating) {
            log("Generation stopped")
            generation?.cancel()
        }
    }

    /** Clears the current conversation. No-op while generating (the UI disables it then). */
    fun clearChat() {
        if (_uiState.value.generating) {
            return
        }
        _uiState.update { it.copy(messages = emptyList(), error = null) }
        log("Chat cleared")
    }

    /**
     * Unloads the current model: cancels any generation, frees the native memory, and deletes the
     * copied working model file (privacy + storage). The chat history is kept (still saveable /
     * copyable), and the UI returns to the "no model loaded" state so a new model can be picked.
     * No-op if no model is loaded.
     */
    fun unloadModel() {
        if (model == null && _uiState.value.modelState == ModelState.NONE) {
            return
        }
        generation?.cancel()
        generation = null
        val toClose = model
        model = null
        modelPath = null
        chatTemplate = null
        mmprojPath = null
        _uiState.update {
            it.copy(
                modelState = ModelState.NONE,
                modelName = null,
                visionModelName = null,
                generating = false,
                error = null,
                pendingImage = null,
            )
        }
        log("Model unloaded")
        viewModelScope.launch(Dispatchers.IO) {
            toClose?.close()
            // Drop the copied working model + mmproj (frees storage; nothing lingers). A model
            // loaded by an absolute path (test / session restore) has no copy here, so this is then
            // a no-op; likewise when no vision projector was ever selected.
            File(getApplication<Application>().filesDir, MODEL_COPY_NAME).delete()
            File(getApplication<Application>().filesDir, MMPROJ_COPY_NAME).delete()
        }
    }

    /** Saves the current conversation (and its model path) to private local storage. */
    fun saveSession() {
        val snapshot = _uiState.value
        val pathAtSave = modelPath
        val nameAtSave = snapshot.modelName
        val messagesAtSave = snapshot.messages
        viewModelScope.launch {
            withContext(Dispatchers.IO) {
                SessionStore.save(getApplication(), pathAtSave, nameAtSave, messagesAtSave)
            }
            log("Session saved (${messagesAtSave.size} messages)")
            _uiState.update { it.copy(notice = Notice.SESSION_SAVED) }
        }
    }

    /** Loads the saved conversation from private local storage, re-opening its model if still present. */
    fun loadSession() {
        viewModelScope.launch {
            val saved = withContext(Dispatchers.IO) { SessionStore.load(getApplication()) }
            if (saved == null) {
                log("Load session: no saved chat")
                _uiState.update { it.copy(notice = Notice.NO_SESSION) }
                return@launch
            }
            log("Session loaded (${saved.messages.size} messages)")
            _uiState.update { it.copy(messages = saved.messages, notice = Notice.SESSION_LOADED) }
            val path = saved.modelPath
            if (path != null && File(path).canRead() && modelPath != path) {
                loadModelFromPath(path, chatTemplate)
            }
        }
    }

    /** Clears the one-shot [UiState.notice] after the UI has shown it. */
    fun consumeNotice() {
        _uiState.update { it.copy(notice = null) }
    }

    /** Converts one turn to the wire type: multimodal (text + image parts) when an image is attached. */
    private fun Message.toChatMessage(): ChatMessage {
        val bytes = imageBytes
        val mime = imageMimeType
        if (bytes != null && mime != null) {
            val parts = mutableListOf<ContentPart>()
            if (text.isNotBlank()) {
                parts.add(ContentPart.text(text))
            }
            parts.add(ContentPart.imageBytes(bytes, mime))
            return ChatMessage(role, parts)
        }
        return ChatMessage(role, text)
    }

    private fun UiState.replaceLastAssistant(newText: String): UiState {
        if (messages.isEmpty()) return this
        val updated = messages.toMutableList()
        updated[updated.lastIndex] = updated.last().copy(text = newText)
        return copy(messages = updated)
    }

    /** Copies a `content://` URI into `filesDir/targetName` (llama.cpp mmaps a real path, not a URI). */
    private fun copyUriToInternal(uri: Uri, targetName: String): String {
        val resolver = getApplication<Application>().contentResolver
        val target = File(getApplication<Application>().filesDir, targetName)
        resolver.openInputStream(uri).use { input ->
            requireNotNull(input) { "content resolver returned no stream for $uri" }
            target.outputStream().use { output -> input.copyTo(output, bufferSize = 1 shl 20) }
        }
        return target.absolutePath
    }

    override fun onCleared() {
        generation?.cancel()
        // Native memory is not GC-managed. viewModelScope is already cancelled here, so close on
        // a fresh daemon thread rather than the main thread (which the native close could block).
        val toClose = model
        model = null
        if (toClose != null) {
            kotlin.concurrent.thread(isDaemon = true, name = "llama-close") { toClose.close() }
        }
    }
}
