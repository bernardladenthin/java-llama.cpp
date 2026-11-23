// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.android.llmservice

import android.app.Activity
import android.content.Context
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AppCompatDelegate
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.clickable
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.SuggestionChip
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.core.os.LocaleListCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import java.util.Locale
import kotlin.math.roundToInt

/**
 * Single-screen KISS demo: pick a GGUF from the file system, then chat with it fully on-device.
 * All inference is in [ChatViewModel]; this file is UI + the SAF picker + the flag language
 * picker + Save/Load session buttons.
 */
class MainActivity : AppCompatActivity() {

    val chatViewModel: ChatViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Edge-to-edge is forced from Android 15 on (targetSdk 35+); enable it explicitly on every version
        // so window-inset handling is consistent. The bottom log strip then pads itself above the
        // system navigation bar (see LogStrip) instead of being drawn behind it.
        enableEdgeToEdge()

        // Test / automation hook: preload a model straight from an absolute path, bypassing the
        // interactive SAF picker (a separate system process the UI test can't drive). The shipping
        // UI never sets these extras.
        intent?.getStringExtra(EXTRA_MODEL_PATH)?.let { path ->
            chatViewModel.loadModelFromPath(path, intent?.getStringExtra(EXTRA_CHAT_TEMPLATE))
        }

        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    ChatScreen(chatViewModel)
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        // Prompt privacy cleanup when the app is actually closing (not on a config-change / locale
        // recreation). The cold-start wipe in LlmServiceApp is the reliable guarantee; this just
        // removes the copied model sooner. isFinishing is false on config changes, so the in-use
        // model survives a language switch.
        if (isFinishing) {
            LlmServiceApp.clearWorkingData(this)
        }
    }

    companion object {
        /** Absolute on-device GGUF path to preload (test hook). */
        const val EXTRA_MODEL_PATH = "net.ladenthin.android.llmservice.MODEL_PATH"

        /** Optional chat-template override for template-less GGUFs (test hook). */
        const val EXTRA_CHAT_TEMPLATE = "net.ladenthin.android.llmservice.CHAT_TEMPLATE"
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ChatScreen(viewModel: ChatViewModel) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val snackbarHostState = remember { SnackbarHostState() }

    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri != null) {
            viewModel.loadModelFromUri(uri, queryDisplayName(context, uri))
        }
    }
    val onChoose: () -> Unit = { picker.launch(arrayOf("*/*")) }

    // Vision projector (mmproj) picker — optional, sets or replaces the vision model for the next load.
    val mmprojPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri != null) {
            viewModel.loadMmprojFromUri(uri, queryDisplayName(context, uri))
        }
    }
    val onChooseMmproj: () -> Unit = { mmprojPicker.launch(arrayOf("*/*")) }

    // Attachment picker — stages one image or one video for the next Send; only usable once a
    // vision model is loaded (the attach button in Conversation is gated on that).
    val imagePicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri != null) {
            viewModel.attachImage(uri, queryDisplayName(context, uri))
        }
    }
    val onAttachImage: () -> Unit = { imagePicker.launch(arrayOf("image/*", "video/*")) }

    // One-shot notices (saved / loaded / none) -> snackbar. Resolve the localized text in
    // composable scope, then show it from the effect.
    val noticeText = state.notice?.let { stringResource(noticeRes(it)) }
    LaunchedEffect(state.notice) {
        if (noticeText != null) {
            snackbarHostState.showSnackbar(noticeText)
            viewModel.consumeNotice()
        }
    }

    var showSettings by remember { mutableStateOf(false) }
    var showLog by remember { mutableStateOf(false) }
    var showClearConfirm by remember { mutableStateOf(false) }
    var showQuitConfirm by remember { mutableStateOf(false) }

    Scaffold(
        topBar = {
            // Two-row app bar: the model name gets its own full-width line (so a long name is
            // readable / draggable), and all the action icons sit on a second line below it.
            Surface(shadowElevation = 3.dp) {
                Column(modifier = Modifier.fillMaxWidth().statusBarsPadding()) {
                    // Row 1: model name (horizontally scrollable — drag a long name into view; no
                    // auto-marquee wobble) + a ❌ button to unload the model when one is loaded.
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(
                            text = state.modelName ?: stringResource(R.string.app_name),
                            style = MaterialTheme.typography.titleLarge,
                            maxLines = 1,
                            softWrap = false,
                            modifier = Modifier.weight(1f)
                                .horizontalScroll(rememberScrollState())
                                .padding(horizontal = 16.dp, vertical = 10.dp),
                        )
                        if (state.modelState == ChatViewModel.ModelState.READY) {
                            IconButton(onClick = { viewModel.unloadModel() }) {
                                Text("❌", modifier = Modifier.testTag("unloadButton"))
                            }
                        }
                    }
                    // Row 2: all action icons on their own line.
                    Row(
                        modifier = Modifier.fillMaxWidth().padding(horizontal = 4.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        IconButton(
                            onClick = { viewModel.saveSession() },
                            enabled = state.messages.isNotEmpty(),
                        ) {
                            Text("💾", modifier = Modifier.testTag("saveButton"))
                        }
                        IconButton(onClick = { viewModel.loadSession() }) {
                            Text("📂", modifier = Modifier.testTag("loadButton"))
                        }
                        IconButton(
                            onClick = { showClearConfirm = true },
                            enabled = state.messages.isNotEmpty() && !state.generating,
                        ) {
                            Text("🗑", modifier = Modifier.testTag("clearButton"))
                        }
                        IconButton(onClick = { showSettings = true }) {
                            Text("⚙️", modifier = Modifier.testTag("settingsButton"))
                        }
                        LanguageMenu()
                    }
                }
            }
        },
        bottomBar = { LogStrip(viewModel = viewModel, onOpen = { showLog = true }) },
        snackbarHost = { SnackbarHost(snackbarHostState) },
    ) { padding ->
        Box(modifier = Modifier.fillMaxSize().padding(padding)) {
            val hasConversation = state.messages.isNotEmpty() || state.modelState == ChatViewModel.ModelState.READY
            when {
                // No model loaded (fresh start, or after unloadModel()/a failed load) always goes
                // to the picker screen — regardless of leftover chat history — so its ❌ quit
                // button and "choose a model" action are reachable. Leftover messages are kept in
                // state and simply reappear once a model becomes READY again (R3.8).
                state.modelState == ChatViewModel.ModelState.NONE ->
                    ChooseModelView(
                        error = errorText(state.error),
                        visionModelName = state.visionModelName,
                        onChoose = onChoose,
                        onChooseMmproj = onChooseMmproj,
                        onClearMmproj = viewModel::clearMmproj,
                        onRequestQuit = { showQuitConfirm = true },
                    )

                state.modelState == ChatViewModel.ModelState.LOADING && !hasConversation ->
                    LoadingView()

                else ->
                    Conversation(
                        state = state,
                        onSend = viewModel::send,
                        onStop = viewModel::stopGeneration,
                        onRegenerate = viewModel::regenerate,
                        onChoose = onChoose,
                        onAttachImage = onAttachImage,
                        onClearAttachment = viewModel::clearPendingImage,
                    )
            }
        }
    }

    if (showClearConfirm) {
        AlertDialog(
            onDismissRequest = { showClearConfirm = false },
            title = { Text(stringResource(R.string.clear_confirm_title)) },
            text = { Text(stringResource(R.string.clear_confirm_message)) },
            confirmButton = {
                TextButton(onClick = {
                    viewModel.clearChat()
                    showClearConfirm = false
                }) { Text(stringResource(R.string.action_clear)) }
            },
            dismissButton = {
                TextButton(onClick = { showClearConfirm = false }) { Text(stringResource(R.string.action_cancel)) }
            },
        )
    }
    if (showQuitConfirm) {
        AlertDialog(
            onDismissRequest = { showQuitConfirm = false },
            title = { Text(stringResource(R.string.quit_confirm_title)) },
            text = { Text(stringResource(R.string.quit_confirm_message)) },
            confirmButton = {
                TextButton(onClick = {
                    showQuitConfirm = false
                    // Wipe the transient working data (model copy + cache) — the saved session
                    // (session.json) is deliberately NOT touched — then close the app and drop it
                    // from Recents.
                    LlmServiceApp.clearWorkingData(context)
                    (context as? Activity)?.finishAndRemoveTask()
                }) { Text(stringResource(R.string.action_quit)) }
            },
            dismissButton = {
                TextButton(onClick = { showQuitConfirm = false }) { Text(stringResource(R.string.action_cancel)) }
            },
        )
    }
    if (showSettings) {
        SettingsDialog(
            settings = state.settings,
            modelConfig = state.modelConfig,
            modelLoaded = state.modelState == ChatViewModel.ModelState.READY,
            onChange = viewModel::updateSettings,
            onModelConfigChange = viewModel::updateModelConfig,
            onReset = viewModel::resetSettings,
            onReload = viewModel::reloadModel,
            onClose = { showSettings = false },
        )
    }
    if (showLog) {
        LogDialog(viewModel = viewModel, onClose = { showLog = false })
    }
}

/** The always-visible one-line log strip at the very bottom; tap to open the full [LogDialog]. */
@Composable
private fun LogStrip(viewModel: ChatViewModel, onOpen: () -> Unit) {
    val log by viewModel.log.collectAsStateWithLifecycle()
    val last = log.lastOrNull() ?: stringResource(R.string.log_empty)
    Surface(tonalElevation = 2.dp) {
        // navigationBarsPadding keeps the strip's content above the system nav bar (back / home /
        // recents) under edge-to-edge; the Surface background still fills to the screen edge behind it.
        Row(
            modifier = Modifier.fillMaxWidth().clickable(onClick = onOpen).testTag("logStrip")
                .navigationBarsPadding()
                .padding(horizontal = 12.dp, vertical = 6.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text("🧾", modifier = Modifier.padding(end = 8.dp))
            Text(
                text = last,
                style = MaterialTheme.typography.labelSmall,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.weight(1f),
            )
        }
    }
}

/** Full-screen log viewer with copy-all, save-as-txt (SAF) and clear; ✕ (top-right) closes it. */
@Composable
private fun LogDialog(viewModel: ChatViewModel, onClose: () -> Unit) {
    val context = LocalContext.current
    val clipboard = LocalClipboardManager.current
    val log by viewModel.log.collectAsStateWithLifecycle()
    val text = log.joinToString("\n")
    val copiedMsg = stringResource(R.string.toast_log_copied)
    val savedMsg = stringResource(R.string.toast_log_saved)

    val saver = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("text/plain")) { uri: Uri? ->
        if (uri != null) {
            context.contentResolver.openOutputStream(uri)?.use { it.write(text.toByteArray(Charsets.UTF_8)) }
            Toast.makeText(context, savedMsg, Toast.LENGTH_SHORT).show()
        }
    }

    Dialog(onDismissRequest = onClose, properties = DialogProperties(usePlatformDefaultWidth = false)) {
        Surface(modifier = Modifier.fillMaxSize().padding(12.dp), shape = MaterialTheme.shapes.large) {
            Column(modifier = Modifier.fillMaxSize().padding(12.dp)) {
                Row(modifier = Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = stringResource(R.string.log_title),
                        style = MaterialTheme.typography.titleLarge,
                        modifier = Modifier.weight(1f),
                    )
                    IconButton(onClick = onClose) { Text("✕", modifier = Modifier.testTag("logClose")) }
                }
                SelectionContainer(modifier = Modifier.weight(1f).fillMaxWidth()) {
                    Column(modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
                        Text(
                            text = text.ifEmpty { stringResource(R.string.log_empty) },
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                }
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.End,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    TextButton(onClick = { viewModel.clearLog() }) { Text(stringResource(R.string.log_clear)) }
                    TextButton(
                        onClick = {
                            clipboard.setText(AnnotatedString(text))
                            Toast.makeText(context, copiedMsg, Toast.LENGTH_SHORT).show()
                        },
                        enabled = text.isNotEmpty(),
                    ) {
                        Text(stringResource(R.string.log_copy_all))
                    }
                    TextButton(onClick = { saver.launch("llm-service-log.txt") }, enabled = text.isNotEmpty()) {
                        Text(stringResource(R.string.log_save_as))
                    }
                }
            }
        }
    }
}

/**
 * Settings sheet. The **sampling** knobs apply live (per request); the **model** knobs
 * (threads / context) are load-time and take effect via **Reload model**.
 */
@Composable
private fun SettingsDialog(
    settings: ChatViewModel.GenerationSettings,
    modelConfig: ChatViewModel.ModelConfig,
    modelLoaded: Boolean,
    onChange: (ChatViewModel.GenerationSettings) -> Unit,
    onModelConfigChange: (ChatViewModel.ModelConfig) -> Unit,
    onReset: () -> Unit,
    onReload: () -> Unit,
    onClose: () -> Unit,
) {
    Dialog(onDismissRequest = onClose) {
        Surface(shape = MaterialTheme.shapes.large, tonalElevation = 6.dp) {
            Column(modifier = Modifier.padding(20.dp).verticalScroll(rememberScrollState())) {
                Text(stringResource(R.string.settings_title), style = MaterialTheme.typography.titleLarge)
                Spacer(Modifier.height(12.dp))
                FloatSetting(R.string.settings_temperature, settings.temperature, 0f..2f) {
                    onChange(settings.copy(temperature = it))
                }
                IntSetting(R.string.settings_top_k, settings.topK, 0..100) { onChange(settings.copy(topK = it)) }
                FloatSetting(R.string.settings_top_p, settings.topP, 0f..1f) { onChange(settings.copy(topP = it)) }
                FloatSetting(R.string.settings_min_p, settings.minP, 0f..1f) { onChange(settings.copy(minP = it)) }
                FloatSetting(R.string.settings_repeat_penalty, settings.repeatPenalty, 1f..2f) {
                    onChange(settings.copy(repeatPenalty = it))
                }
                IntSetting(R.string.settings_repeat_last_n, settings.repeatLastN, 0..256) {
                    onChange(settings.copy(repeatLastN = it))
                }
                IntSetting(R.string.settings_max_tokens, settings.maxTokens, 16..1024) {
                    onChange(settings.copy(maxTokens = it))
                }

                // Load-time model knobs: only take effect on (re)load, so they carry an explicit
                // Reload button (enabled when a model is loaded).
                Spacer(Modifier.height(12.dp))
                Text(stringResource(R.string.settings_model_section), style = MaterialTheme.typography.titleMedium)
                IntSetting(R.string.settings_threads, modelConfig.threads, 1..16) {
                    onModelConfigChange(modelConfig.copy(threads = it))
                }
                IntSetting(R.string.settings_context, modelConfig.contextSize, 512..8192) {
                    onModelConfigChange(modelConfig.copy(contextSize = it))
                }

                Spacer(Modifier.height(8.dp))
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
                    TextButton(onClick = onReload, enabled = modelLoaded) {
                        Text(stringResource(R.string.settings_reload))
                    }
                    TextButton(onClick = onReset) { Text(stringResource(R.string.settings_reset)) }
                    TextButton(onClick = onClose) { Text(stringResource(R.string.action_close)) }
                }
            }
        }
    }
}

/** A labelled float slider showing its current value (2 decimals, locale-independent). */
@Composable
private fun FloatSetting(labelRes: Int, value: Float, range: ClosedFloatingPointRange<Float>, onChange: (Float) -> Unit) {
    Column(modifier = Modifier.padding(vertical = 4.dp)) {
        Text("${stringResource(labelRes)}: ${String.format(Locale.US, "%.2f", value)}", style = MaterialTheme.typography.bodyMedium)
        Slider(value = value, onValueChange = onChange, valueRange = range)
    }
}

/** A labelled integer slider (rounds the slider's float to the nearest int). */
@Composable
private fun IntSetting(labelRes: Int, value: Int, range: IntRange, onChange: (Int) -> Unit) {
    Column(modifier = Modifier.padding(vertical = 4.dp)) {
        Text("${stringResource(labelRes)}: $value", style = MaterialTheme.typography.bodyMedium)
        Slider(
            value = value.toFloat(),
            onValueChange = { onChange(it.roundToInt()) },
            valueRange = range.first.toFloat()..range.last.toFloat(),
        )
    }
}

@Composable
private fun LanguageMenu() {
    var expanded by remember { mutableStateOf(false) }
    val activeFlag = flagForActiveTags(AppCompatDelegate.getApplicationLocales().toLanguageTags())
    IconButton(onClick = { expanded = true }) {
        Text(activeFlag, modifier = Modifier.testTag("languageButton"))
    }
    DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
        for (language in APP_LANGUAGES) {
            DropdownMenuItem(
                text = { Text("${language.flag}  ${language.endonym}") },
                onClick = {
                    expanded = false
                    val locales =
                        if (language.tag.isEmpty()) {
                            LocaleListCompat.getEmptyLocaleList()
                        } else {
                            LocaleListCompat.forLanguageTags(language.tag)
                        }
                    // Recreates the activity in the chosen locale; the ViewModel (and the loaded
                    // model) survive the recreation, so the chat is not lost.
                    AppCompatDelegate.setApplicationLocales(locales)
                },
            )
        }
    }
}

@Composable
private fun LoadingView() {
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        CircularProgressIndicator()
        Text(stringResource(R.string.status_loading_model), modifier = Modifier.padding(top = 16.dp))
    }
}

@Composable
private fun ChooseModelView(
    error: String?,
    visionModelName: String?,
    onChoose: () -> Unit,
    onChooseMmproj: () -> Unit,
    onClearMmproj: () -> Unit,
    onRequestQuit: () -> Unit,
) {
    Box(modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier.fillMaxSize().padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            OfflineBadge(modifier = Modifier.padding(bottom = 16.dp))
            Text(text = stringResource(R.string.intro_pick_model), textAlign = TextAlign.Center)
            Button(
                onClick = onChoose,
                modifier = Modifier.padding(top = 24.dp).testTag("chooseModelButton"),
            ) {
                Text(stringResource(R.string.action_choose_model))
            }
            // Optional vision projector (mmproj): picked independently of the main model, applied on
            // the next load. Shows the selected file's name with a ✕ to clear it once set.
            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(top = 8.dp)) {
                TextButton(onClick = onChooseMmproj, modifier = Modifier.testTag("chooseMmprojButton")) {
                    Text(
                        if (visionModelName != null) {
                            stringResource(R.string.action_vision_model_set, visionModelName)
                        } else {
                            stringResource(R.string.action_choose_vision_model)
                        },
                    )
                }
                if (visionModelName != null) {
                    IconButton(onClick = onClearMmproj, modifier = Modifier.testTag("clearMmprojButton")) {
                        Text("✕")
                    }
                }
            }
            DeviceCard(modifier = Modifier.padding(top = 24.dp))
            if (error != null) {
                Text(text = error, color = MaterialTheme.colorScheme.error, modifier = Modifier.padding(top = 16.dp))
            }
        }
        // Quit + clean up (keeps saved chats) — a "close app" ❌ at the top-right of the main screen.
        IconButton(
            onClick = onRequestQuit,
            modifier = Modifier.align(Alignment.TopEnd).testTag("quitButton"),
        ) {
            Text("❌")
        }
    }
}

/** A small readiness card: free RAM / storage / battery — so the user can gauge if a model fits. */
@Composable
private fun DeviceCard(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    // One-shot read at composition; the picker screen is short-lived, so live updates aren't needed.
    val device = remember { DeviceInfo.read(context) }
    Surface(
        color = MaterialTheme.colorScheme.surfaceVariant,
        shape = MaterialTheme.shapes.medium,
        modifier = modifier,
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(stringResource(R.string.cd_device), style = MaterialTheme.typography.titleSmall)
            Spacer(Modifier.height(8.dp))
            Text(
                "${stringResource(R.string.device_ram)}: " +
                    stringResource(
                        R.string.device_free_of,
                        formatBytes(device.availMemBytes),
                        formatBytes(device.totalMemBytes),
                    ),
                style = MaterialTheme.typography.bodyMedium,
            )
            Text(
                "${stringResource(R.string.device_storage)}: ${formatBytes(device.availStorageBytes)}",
                style = MaterialTheme.typography.bodyMedium,
            )
            val battery =
                if (device.batteryPercent >= 0) {
                    val charging = if (device.charging) " (${stringResource(R.string.device_charging)})" else ""
                    "${device.batteryPercent}%$charging"
                } else {
                    "—"
                }
            Text(
                "${stringResource(R.string.device_battery)}: $battery",
                style = MaterialTheme.typography.bodyMedium,
            )
        }
    }
}

@Composable
private fun Conversation(
    state: ChatViewModel.UiState,
    onSend: (String, String) -> Unit,
    onStop: () -> Unit,
    onRegenerate: (String) -> Unit,
    onChoose: () -> Unit,
    onAttachImage: () -> Unit,
    onClearAttachment: () -> Unit,
) {
    val systemPrompt = stringResource(R.string.system_prompt)
    val context = LocalContext.current
    val clipboard = LocalClipboardManager.current
    val copiedMsg = stringResource(R.string.toast_copied)
    Column(modifier = Modifier.fillMaxSize()) {
        val listState = rememberLazyListState()
        LaunchedEffect(state.messages.size, state.messages.lastOrNull()?.text) {
            if (state.messages.isNotEmpty()) {
                listState.animateScrollToItem(state.messages.lastIndex)
            }
        }
        LazyColumn(
            state = listState,
            modifier = Modifier.weight(1f).fillMaxWidth().padding(horizontal = 12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            items(state.messages) { message -> MessageBubble(message) }
        }

        // Copy / Regenerate act on the last assistant reply, shown only when it's a finished,
        // non-empty reply and we're idle.
        val lastReply = state.messages.lastOrNull()?.takeIf { it.role == "assistant" && it.text.isNotBlank() }
        if (lastReply != null && !state.generating && state.modelState == ChatViewModel.ModelState.READY) {
            Row(
                modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp),
                horizontalArrangement = Arrangement.End,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                TextButton(
                    onClick = {
                        clipboard.setText(AnnotatedString(lastReply.text))
                        Toast.makeText(context, copiedMsg, Toast.LENGTH_SHORT).show()
                    },
                    modifier = Modifier.testTag("copyButton"),
                ) {
                    Text(stringResource(R.string.action_copy))
                }
                TextButton(
                    onClick = { onRegenerate(systemPrompt) },
                    modifier = Modifier.testTag("regenerateButton"),
                ) {
                    Text(stringResource(R.string.action_regenerate))
                }
            }
        }

        if (state.modelState != ChatViewModel.ModelState.READY) {
            NotReadyBanner(state.modelState, onChoose)
        }

        errorText(state.error)?.let { message ->
            Text(
                text = message,
                color = MaterialTheme.colorScheme.error,
                modifier = Modifier.padding(horizontal = 16.dp),
            )
        }

        val ready = state.modelState == ChatViewModel.ModelState.READY
        var draft by remember { mutableStateOf("") }

        // Prompt shortcut chips: quick starters that fill the input (localized). Shown only when
        // ready, idle, and the input is still empty, so they don't clutter an in-progress message.
        if (ready && !state.generating && draft.isBlank()) {
            Row(
                modifier = Modifier.fillMaxWidth().horizontalScroll(rememberScrollState())
                    .padding(horizontal = 12.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                for (chip in intArrayOf(R.string.chip_summarize, R.string.chip_explain, R.string.chip_ideas)) {
                    val label = stringResource(chip)
                    SuggestionChip(
                        onClick = { draft = "$label " },
                        label = { Text(label) },
                        modifier = Modifier.testTag("promptChip"),
                    )
                }
            }
        }

        // Pending image attachment: a small removable chip shown above the input once one is staged.
        state.pendingImage?.let { pending ->
            Row(
                modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                SuggestionChip(
                    onClick = {},
                    label = { Text("🖼️ ${pending.displayName}") },
                    modifier = Modifier.weight(1f, fill = false).testTag("pendingImageChip"),
                )
                IconButton(onClick = onClearAttachment, modifier = Modifier.testTag("removeAttachmentButton")) {
                    Text("✕")
                }
            }
        }

        Row(
            modifier = Modifier.fillMaxWidth().padding(12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            // Attach an image for the loaded vision model; hidden entirely when no vision projector
            // is loaded (attaching an image to a text-only model would just be ignored).
            if (state.visionModelName != null) {
                IconButton(
                    onClick = onAttachImage,
                    enabled = ready && !state.generating,
                    modifier = Modifier.testTag("attachImageButton"),
                ) {
                    Text("📎")
                }
            }
            OutlinedTextField(
                value = draft,
                onValueChange = { draft = it },
                modifier = Modifier.weight(1f).testTag("promptInput"),
                placeholder = { Text(stringResource(R.string.hint_message)) },
                enabled = ready && !state.generating,
            )
            if (state.generating) {
                // Swap Send for Stop while a reply streams, so a runaway/repetitive reply can be cut off.
                Button(
                    onClick = onStop,
                    modifier = Modifier.padding(start = 8.dp).testTag("stopButton"),
                ) {
                    Text(stringResource(R.string.action_stop))
                }
            } else {
                Button(
                    onClick = {
                        onSend(draft, systemPrompt)
                        draft = ""
                    },
                    enabled = ready && (draft.isNotBlank() || state.pendingImage != null),
                    modifier = Modifier.padding(start = 8.dp).testTag("sendButton"),
                ) {
                    Text(stringResource(R.string.action_send))
                }
            }
        }
    }
}

/** Small "fully on-device" chip reinforcing the app's zero-network, nothing-leaves-the-device posture. */
@Composable
private fun OfflineBadge(modifier: Modifier = Modifier) {
    Surface(
        color = MaterialTheme.colorScheme.secondaryContainer,
        shape = MaterialTheme.shapes.small,
        modifier = modifier,
    ) {
        Text(
            text = stringResource(R.string.badge_offline),
            style = MaterialTheme.typography.labelMedium,
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
        )
    }
}

@Composable
private fun NotReadyBanner(modelState: ChatViewModel.ModelState, onChoose: () -> Unit) {
    Row(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (modelState == ChatViewModel.ModelState.LOADING) {
            CircularProgressIndicator(modifier = Modifier.size(18.dp))
            Text(
                stringResource(R.string.status_loading_model),
                modifier = Modifier.padding(start = 12.dp),
            )
        } else {
            Text(stringResource(R.string.banner_model_not_loaded), modifier = Modifier.weight(1f))
            TextButton(onClick = onChoose) { Text(stringResource(R.string.action_choose_model)) }
        }
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun MessageBubble(message: ChatViewModel.Message) {
    val isUser = message.role == "user"
    val bubbleColor =
        if (isUser) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant
    val context = LocalContext.current
    val clipboard = LocalClipboardManager.current
    val copiedMsg = stringResource(R.string.toast_copied)
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start,
    ) {
        Surface(
            color = bubbleColor,
            shape = MaterialTheme.shapes.medium,
            modifier = Modifier
                .testTag(if (isUser) "userMessage" else "assistantMessage")
                // Long-press any bubble to copy its text (complements the last-reply Copy button).
                .combinedClickable(
                    onClick = {},
                    onLongClick = {
                        if (message.text.isNotBlank()) {
                            clipboard.setText(AnnotatedString(message.text))
                            Toast.makeText(context, copiedMsg, Toast.LENGTH_SHORT).show()
                        }
                    },
                ),
        ) {
            val prefix = when {
                message.imageBytes == null -> ""
                message.imageMimeType?.startsWith("video/") == true -> "🎞️ "
                else -> "🖼️ "
            }
            Text(text = prefix + message.text.ifEmpty { "…" }, modifier = Modifier.padding(12.dp))
        }
    }
}

@Composable
private fun errorText(error: ChatViewModel.ErrorInfo?): String? {
    if (error == null) return null
    val template =
        when (error.type) {
            ChatViewModel.ErrorType.LOAD -> R.string.error_load_model
            ChatViewModel.ErrorType.GENERATION -> R.string.error_generation
        }
    return stringResource(template, error.detail)
}

private fun noticeRes(notice: ChatViewModel.Notice): Int =
    when (notice) {
        ChatViewModel.Notice.SESSION_SAVED -> R.string.toast_session_saved
        ChatViewModel.Notice.SESSION_LOADED -> R.string.toast_session_loaded
        ChatViewModel.Notice.NO_SESSION -> R.string.toast_session_none
    }

private fun queryDisplayName(context: Context, uri: Uri): String {
    context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
        if (cursor.moveToFirst()) {
            val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (index >= 0) {
                return cursor.getString(index)
            }
        }
    }
    return "model.gguf"
}
