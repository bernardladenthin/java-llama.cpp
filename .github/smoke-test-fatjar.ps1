# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# Smoke test for an all-backends server fat jar on a GPU-less Windows runner —
# the PowerShell analogue of smoke-test-fatjar.sh (see there for what the two rounds prove:
# round 1 plain HTTP with the cached model, round 2 an HTTPS server behind a self-signed
# certificate serving a model it downloaded itself from an https:// URL).
#
# Usage: smoke-test-fatjar.ps1 -JarDir <dir> -JarGlob <glob> -Model <gguf> [-Port <p>]
# Server output is written to server-out.log / server-err.log (round 1) and server-tls-out.log /
# server-tls-err.log (round 2) in the working dir (uploaded by the CI job on failure).
param(
    [Parameter(Mandatory = $true)][string]$JarDir,
    [Parameter(Mandatory = $true)][string]$JarGlob,
    [Parameter(Mandatory = $true)][string]$Model,
    [int]$Port = 18080
)
$ErrorActionPreference = 'Stop'
$TlsPort = $Port + 1
$HttpsModelName = 'stories260K.gguf'

function Dump-ServerLogs([string[]]$Logs) {
    foreach ($log in $Logs) {
        if (Test-Path $log) {
            Write-Host "--- $log (tail) ---"
            Get-Content $log -Tail 50
        }
    }
}

# Poll /health until 200 (model loaded); 100 x 3 s = 5 min budget. An early server exit (e.g. an
# UnsatisfiedLinkError the fallback chain failed to absorb) fails fast. -SkipCertificateCheck
# accepts round 2's self-signed certificate and is a no-op over plain HTTP.
function Wait-Healthy($Proc, [string]$Base, [string[]]$Logs) {
    foreach ($i in 1..100) {
        if ($Proc.HasExited) {
            Dump-ServerLogs $Logs
            Write-Error "server process exited before becoming healthy ($Base, exit code $($Proc.ExitCode))"
        }
        try {
            $health = Invoke-WebRequest -Uri "$Base/health" -UseBasicParsing -SkipCertificateCheck -TimeoutSec 5
            if ($health.StatusCode -eq 200) { return }
        } catch {
            # 503 while loading / connection refused before listening: keep polling.
        }
        Start-Sleep -Seconds 3
    }
    Dump-ServerLogs $Logs
    Write-Error "$Base/health never returned 200"
}

# A chat completion with one valid choice, over $Base.
function Test-ChatCompletion([string]$Base) {
    $body = '{"messages":[{"role":"user","content":"Say hello."}],"max_tokens":16,"temperature":0}'
    $response = Invoke-RestMethod -Uri "$Base/v1/chat/completions" -SkipCertificateCheck `
        -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 300
    if (-not $response.choices -or $response.choices.Count -lt 1 -or -not $response.choices[0].message) {
        Write-Error "malformed chat completion response: $($response | ConvertTo-Json -Depth 6 -Compress)"
    }
    Write-Host "chat completion OK ($Base): $($response.choices[0].message.content)"
}

$jars = @(Get-ChildItem -Path $JarDir -Filter $JarGlob -File)
if ($jars.Count -ne 1) {
    Write-Error "expected exactly 1 jar matching $JarGlob in $JarDir, got $($jars.Count)"
}
$jar = $jars[0].FullName
if (-not (Test-Path $Model)) { Write-Error "model file missing: $Model" }
Write-Host "smoke jar: $jar"

$modelsCsv = Join-Path $PSScriptRoot 'models.csv'
$httpsModelUrl = (Get-Content $modelsCsv | Where-Object { $_ -like "$HttpsModelName,*" } | Select-Object -First 1)
if (-not $httpsModelUrl) { Write-Error "$HttpsModelName has no row in $modelsCsv (round 2 downloads it over HTTPS)" }
$httpsModelUrl = $httpsModelUrl.Substring($HttpsModelName.Length + 1)

# ---- Round 1: plain HTTP, the cached model -------------------------------------------------------
$proc = Start-Process java -PassThru -NoNewWindow `
    -RedirectStandardOutput server-out.log -RedirectStandardError server-err.log `
    -ArgumentList '-jar', $jar, '-m', $Model, '--host', '127.0.0.1', '--port', "$Port", '--chat-template', 'chatml'
try {
    Wait-Healthy $proc "http://127.0.0.1:$Port" @('server-out.log', 'server-err.log')
    Write-Host "health OK"
    Test-ChatCompletion "http://127.0.0.1:$Port"

    # The loader must have reported its backend decision (a chosen backend on a GPU
    # machine, the CPU fallback on a GPU-less runner) — this pins that the smoke really
    # ran the multi-backend code path.
    $selection = Select-String -Path 'server-out.log', 'server-err.log' `
        -Pattern '\[jllama\] using native backend'
    if (-not $selection) {
        Dump-ServerLogs @('server-out.log', 'server-err.log')
        Write-Error "no backend-selection log line found - the loader did not report a backend"
    }
    Write-Host "backend selection: $($selection[0].Line)"
} finally {
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
}

# ---- Round 2: HTTPS server + https:// model download ---------------------------------------------
# openssl comes with Git for Windows (usr\bin is not on PATH); the runner images have it.
$openssl = (Get-Command openssl -ErrorAction SilentlyContinue).Source
if (-not $openssl) { $openssl = Join-Path $env:ProgramFiles 'Git\usr\bin\openssl.exe' }
if (-not (Test-Path $openssl)) { Write-Error "no openssl CLI found to mint the TLS test certificate" }
& $openssl req -x509 -newkey rsa:2048 -nodes -keyout tls-key.pem -out tls-cert.pem -days 2 -subj '/CN=127.0.0.1' 2>$null
if ($LASTEXITCODE -ne 0 -or -not (Test-Path tls-cert.pem)) { Write-Error "could not create the self-signed TLS certificate" }
$env:LLAMA_CACHE = Join-Path (Get-Location) 'llama-cache'
if (Test-Path $env:LLAMA_CACHE) { Remove-Item -Recurse -Force $env:LLAMA_CACHE }
New-Item -ItemType Directory -Path $env:LLAMA_CACHE | Out-Null

# -m names where the download lands; with --model-url alone the server starts in ROUTER mode.
$proc = Start-Process java -PassThru -NoNewWindow `
    -RedirectStandardOutput server-tls-out.log -RedirectStandardError server-tls-err.log `
    -ArgumentList '-jar', $jar, '-m', (Join-Path $env:LLAMA_CACHE $HttpsModelName), '--model-url', $httpsModelUrl, `
        '--host', '127.0.0.1', '--port', "$TlsPort", `
        '--chat-template', 'chatml', '--ssl-key-file', 'tls-key.pem', '--ssl-cert-file', 'tls-cert.pem'
try {
    Wait-Healthy $proc "https://127.0.0.1:$TlsPort" @('server-tls-out.log', 'server-tls-err.log')
    Write-Host "HTTPS health OK"
    Test-ChatCompletion "https://127.0.0.1:$TlsPort"
    # The port must really speak TLS: a plain-HTTP request to it fails.
    $plainAnswered = $false
    try {
        Invoke-WebRequest -Uri "http://127.0.0.1:$TlsPort/health" -UseBasicParsing -TimeoutSec 10 | Out-Null
        $plainAnswered = $true
    } catch {
        # expected: the TLS server rejects the plain request
    }
    if ($plainAnswered) { Write-Error "the TLS port answered a plain-HTTP request - the server did not use the certificate" }
    Write-Host "plain HTTP on the TLS port refused: OK"
    # The model came over HTTPS into this run's cache, not from the runner's model directory.
    $downloaded = @(Get-ChildItem -Path $env:LLAMA_CACHE -Recurse -File -Filter '*.gguf')
    if ($downloaded.Count -lt 1) { Write-Error "no .gguf under $env:LLAMA_CACHE - the https:// download did not happen" }
    Write-Host "https:// model download OK: $($downloaded[0].Name)"
    Write-Host "smoke test PASSED"
} finally {
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
}
