# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# Smoke test for an all-backends server fat jar on a GPU-less Windows runner —
# the PowerShell analogue of smoke-test-fatjar.sh (see there for what it proves).
#
# Usage: smoke-test-fatjar.ps1 -JarDir <dir> -JarGlob <glob> -Model <gguf> [-Port <p>]
# Server output is written to server-out.log / server-err.log in the working dir
# (uploaded by the CI job on failure).
param(
    [Parameter(Mandatory = $true)][string]$JarDir,
    [Parameter(Mandatory = $true)][string]$JarGlob,
    [Parameter(Mandatory = $true)][string]$Model,
    [int]$Port = 18080
)
$ErrorActionPreference = 'Stop'

function Dump-ServerLogs {
    foreach ($log in 'server-out.log', 'server-err.log') {
        if (Test-Path $log) {
            Write-Host "--- $log (tail) ---"
            Get-Content $log -Tail 50
        }
    }
}

$jars = @(Get-ChildItem -Path $JarDir -Filter $JarGlob -File)
if ($jars.Count -ne 1) {
    Write-Error "expected exactly 1 jar matching $JarGlob in $JarDir, got $($jars.Count)"
}
$jar = $jars[0].FullName
if (-not (Test-Path $Model)) { Write-Error "model file missing: $Model" }
Write-Host "smoke jar: $jar"

$proc = Start-Process java -PassThru -NoNewWindow `
    -RedirectStandardOutput server-out.log -RedirectStandardError server-err.log `
    -ArgumentList '-jar', $jar, '-m', $Model, '--host', '127.0.0.1', '--port', "$Port", '--chat-template', 'chatml'
try {
    # Poll /health until 200 (model loaded); 100 x 3 s = 5 min budget. An early server
    # exit (e.g. an UnsatisfiedLinkError the fallback chain failed to absorb) fails fast.
    $healthy = $false
    foreach ($i in 1..100) {
        if ($proc.HasExited) {
            Dump-ServerLogs
            Write-Error "server process exited before becoming healthy (exit code $($proc.ExitCode))"
        }
        try {
            $health = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -UseBasicParsing -TimeoutSec 5
            if ($health.StatusCode -eq 200) { $healthy = $true; break }
        } catch {
            # 503 while loading / connection refused before listening: keep polling.
        }
        Start-Sleep -Seconds 3
    }
    if (-not $healthy) {
        Dump-ServerLogs
        Write-Error "/health never returned 200"
    }
    Write-Host "health OK"

    $body = '{"messages":[{"role":"user","content":"Say hello."}],"max_tokens":16,"temperature":0}'
    $response = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" `
        -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 300
    if (-not $response.choices -or $response.choices.Count -lt 1 -or -not $response.choices[0].message) {
        Write-Error "malformed chat completion response: $($response | ConvertTo-Json -Depth 6 -Compress)"
    }
    Write-Host "chat completion OK: $($response.choices[0].message.content)"

    # The loader must have reported its backend decision (a chosen backend on a GPU
    # machine, the CPU fallback on a GPU-less runner) — this pins that the manifest was
    # actually read, i.e. the smoke really ran the multi-backend code path.
    $selection = Select-String -Path 'server-out.log', 'server-err.log' `
        -Pattern '\[jllama\] (using native backend|no manifest backend loadable)'
    if (-not $selection) {
        Dump-ServerLogs
        Write-Error "no backend-selection log line found - the backend manifest was not processed"
    }
    Write-Host "backend selection: $($selection[0].Line)"
    Write-Host "smoke test PASSED"
} finally {
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
}
