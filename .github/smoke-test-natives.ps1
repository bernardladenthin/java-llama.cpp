# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# Smoke test for a smoke set (the classes jar, its dependencies, the CPU natives jar and every GPU
# module jar of this platform, see .github/package-smoke-sets.sh) on a GPU-less Windows runner --
# the PowerShell analogue of smoke-test-natives.sh (see there for what the two rounds prove:
# round 1 plain HTTP with the cached model, round 2 an HTTPS server behind a self-signed
# certificate serving a model it downloaded itself from an https:// URL).
#
# Round 3 (the natives directory through net.ladenthin.llama.lib.path, no natives jar on the
# classpath) is the Windows half of what smoke-test-natives.sh explains: this is the platform where
# the bare System.load used to fail, because jllama.dll imports ggml.dll and ggml-base.dll and
# Windows does not look next to the DLL for them -- the loader's directory branch preloads them
# from the directory's jllama-extras.txt, exactly as the jar branch does.
#
# Usage: smoke-test-natives.ps1 -SetDir <dir> -Model <gguf> [-Port <p>]
# Server output is written to server-out.log / server-err.log (round 1), server-tls-out.log /
# server-tls-err.log (round 2) and server-dir-out.log / server-dir-err.log (round 3) in the working
# dir (uploaded by the CI job on failure).
param(
    [Parameter(Mandatory = $true)][string]$SetDir,
    [Parameter(Mandatory = $true)][string]$Model,
    [int]$Port = 18080
)
$ErrorActionPreference = 'Stop'
$TlsPort = $Port + 1
$DirPort = $Port + 2
$HttpsModelName = 'stories260K.gguf'
$Main = 'net.ladenthin.llama.server.ServerLauncher'

# The GPU module backends whose jars are in the set, from natives.csv: the loader line must name each.
function Get-SetModules([string]$Dir) {
    $list = Join-Path $PSScriptRoot 'natives.csv'
    $modules = @()
    foreach ($line in Get-Content $list) {
        if ($line -match '^#' -or $line -match '^classifier,' -or -not $line.Trim()) { continue }
        $cols = $line.Split(',')
        if ($cols[4] -ne 'module') { continue }
        # An opt-in module (natives.csv column 6) is deliberately left out by the loader until
        # net.ladenthin.llama.backend names it, so it is not expected in the loader line.
        if ($cols.Length -gt 5 -and $cols[5] -eq 'yes') { continue }
        if (Get-ChildItem -Path $Dir -Filter "llama-*-$($cols[0]).jar" -File) {
            $modules += ($cols[1] -split '/')[-1]
        }
    }
    return $modules
}

# The rows of natives.csv as objects (classifier, dir, lib, kind).
function Get-NativesRows() {
    $list = Join-Path $PSScriptRoot 'natives.csv'
    $rows = @()
    foreach ($line in Get-Content $list) {
        if ($line -match '^#' -or $line -match '^classifier,' -or -not $line.Trim()) { continue }
        $cols = $line.Split(',')
        $rows += [pscustomobject]@{ Classifier = $cols[0]; Dir = $cols[1]; Lib = $cols[2]; Kind = $cols[4] }
    }
    return $rows
}

# The classpath of the set without its natives jars (the classes jar and its runtime dependencies):
# what a consumer has who ships the natives as a directory and names it with lib.path.
function Get-ClasspathWithoutNatives([string]$Dir) {
    $classifiers = @(Get-NativesRows | ForEach-Object { $_.Classifier })
    $jars = @(Get-ChildItem -Path $Dir -Filter '*.jar' -File | Where-Object {
        $base = $_.BaseName
        -not ($classifiers | Where-Object { $base.EndsWith("-$_") })
    } | ForEach-Object { $_.FullName })
    if ($jars.Count -lt 1) { Write-Error "no non-natives jar in the smoke set $Dir" }
    return ($jars -join ';')
}

# Unpacks the natives jars of the set into ONE natives directory the way a consumer does by hand
# (the bash twin is smoke_natives_directory in smoke-lib.sh): the library jar's directory, every
# module's files copied flat into it, the modules' jllama-extras.txt / jllama-files.txt lines
# appended to the library's, their jllama-build.txt and jllama-sbom.json dropped.
function New-NativesDirectory([string]$Dir, [string]$Out) {
    if (Test-Path $Out) { Remove-Item -Recurse -Force $Out }
    New-Item -ItemType Directory -Path $Out | Out-Null
    $rows = Get-NativesRows
    $libDir = $null
    foreach ($row in ($rows | Where-Object { $_.Kind -eq 'library' })) {
        $jar = Get-ChildItem -Path $Dir -Filter "llama-*-$($row.Classifier).jar" -File | Select-Object -First 1
        if (-not $jar) { continue }
        if ($libDir) { Write-Error "two library jars in the smoke set: $libDir and $($row.Classifier)" }
        $zip = Join-Path $Out "$($row.Classifier).zip"
        Copy-Item $jar.FullName $zip
        Expand-Archive -Path $zip -DestinationPath (Join-Path $Out 'library') -Force
        $libDir = Join-Path (Join-Path $Out 'library') ("net/ladenthin/llama/" + $row.Dir)
        if (-not (Test-Path (Join-Path $libDir $row.Lib))) { Write-Error "$($jar.Name) holds no net/ladenthin/llama/$($row.Dir)/$($row.Lib)" }
    }
    if (-not $libDir) { Write-Error "no library jar in the smoke set $Dir" }
    foreach ($row in ($rows | Where-Object { $_.Kind -eq 'module' })) {
        $jar = Get-ChildItem -Path $Dir -Filter "llama-*-$($row.Classifier).jar" -File | Select-Object -First 1
        if (-not $jar) { continue }
        $zip = Join-Path $Out "$($row.Classifier).zip"
        Copy-Item $jar.FullName $zip
        $modRoot = Join-Path $Out $row.Classifier
        Expand-Archive -Path $zip -DestinationPath $modRoot -Force
        $modDir = Join-Path $modRoot ("net/ladenthin/llama/" + $row.Dir)
        if (-not (Test-Path (Join-Path $modDir $row.Lib))) { Write-Error "$($jar.Name) holds no net/ladenthin/llama/$($row.Dir)/$($row.Lib)" }
        foreach ($f in Get-ChildItem -Path $modDir -File) {
            switch ($f.Name) {
                { $_ -in 'jllama-build.txt', 'jllama-sbom.json' } { }
                { $_ -in 'jllama-extras.txt', 'jllama-files.txt' } {
                    $lines = Get-Content $f.FullName | Where-Object { $_.Trim() -and -not $_.StartsWith('#') }
                    Add-Content -Path (Join-Path $libDir $f.Name) -Value $lines
                }
                default {
                    $target = Join-Path $libDir $f.Name
                    if (Test-Path $target) { Write-Error "$($row.Classifier) ships $($f.Name), which the library directory already holds" }
                    Copy-Item $f.FullName $target
                }
            }
        }
    }
    return (Resolve-Path $libDir).Path
}

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

if (-not (Test-Path $SetDir)) { Write-Error "smoke set directory missing: $SetDir" }
$SetDir = (Resolve-Path $SetDir).Path
$jars = @(Get-ChildItem -Path $SetDir -Filter '*.jar' -File)
if ($jars.Count -lt 1) { Write-Error "no jars in the smoke set $SetDir" }
$cp = Join-Path $SetDir '*'
if (-not (Test-Path $Model)) { Write-Error "model file missing: $Model" }
$setModules = @(Get-SetModules $SetDir)
Write-Host "smoke set: $SetDir ($($jars.Count) jars; GPU module jars: $($setModules -join ', '))"

$modelsCsv = Join-Path $PSScriptRoot 'models.csv'
$httpsModelUrl = (Get-Content $modelsCsv | Where-Object { $_ -like "$HttpsModelName,*" } | Select-Object -First 1)
if (-not $httpsModelUrl) { Write-Error "$HttpsModelName has no row in $modelsCsv (round 2 downloads it over HTTPS)" }
$httpsModelUrl = ($httpsModelUrl -split ',')[3]  # filename,feature,sha256,url

# ---- Round 1: plain HTTP, the cached model -------------------------------------------------------
$proc = Start-Process java -PassThru -NoNewWindow `
    -RedirectStandardOutput server-out.log -RedirectStandardError server-err.log `
    -ArgumentList '-cp', "`"$cp`"", $Main, '-m', $Model, '--host', '127.0.0.1', '--port', "$Port", '--chat-template', 'chatml'
try {
    Wait-Healthy $proc "http://127.0.0.1:$Port" @('server-out.log', 'server-err.log')
    Write-Host "health OK"
    Test-ChatCompletion "http://127.0.0.1:$Port"

    # The loader must have reported its backend decision (a chosen backend on a GPU
    # machine, the CPU fallback on a GPU-less runner) — this pins that the smoke really
    # ran the multi-backend code path.
    $selection = Select-String -Path 'server-out.log', 'server-err.log' `
        -Pattern "\[jllama\] native backend '[^']*' loaded "
    if (-not $selection) {
        Dump-ServerLogs @('server-out.log', 'server-err.log')
        Write-Error "no loader line found - the loader did not report the backend"
    }
    $loaderLine = $selection[0].Line
    Write-Host "loader: $loaderLine"
    foreach ($module in $setModules) {
        if ($loaderLine -notmatch "GPU module\(s\) ([a-z0-9-]+, )*$module(,|;|$)") {
            Write-Error "the loader did not put the $module module in place: $loaderLine"
        }
    }
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
    -ArgumentList '-cp', "`"$cp`"", $Main, '-m', (Join-Path $env:LLAMA_CACHE $HttpsModelName), '--model-url', $httpsModelUrl, `
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
} finally {
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
}

# ---- Round 3: the natives directory through lib.path, no natives jar on the classpath -----------
$nativesDir = New-NativesDirectory $SetDir (Join-Path (Get-Location) 'natives-dir')
$fileCount = @(Get-ChildItem -Path $nativesDir -File).Count
Write-Host "natives directory: $nativesDir ($fileCount files)"
if (-not (Select-String -Path (Join-Path $nativesDir 'jllama-build.txt') -Pattern '^llama.cpp=' -Quiet)) {
    Write-Error "the unpacked directory has no jllama-build.txt"
}
$cpNoNatives = Get-ClasspathWithoutNatives $SetDir
$proc = Start-Process java -PassThru -NoNewWindow `
    -RedirectStandardOutput server-dir-out.log -RedirectStandardError server-dir-err.log `
    -ArgumentList '-cp', "`"$cpNoNatives`"", "-Dnet.ladenthin.llama.lib.path=$nativesDir", $Main, '-m', $Model, `
        '--host', '127.0.0.1', '--port', "$DirPort", '--chat-template', 'chatml'
try {
    Wait-Healthy $proc "http://127.0.0.1:$DirPort" @('server-dir-out.log', 'server-dir-err.log')
    Write-Host "health OK (lib.path)"
    Test-ChatCompletion "http://127.0.0.1:$DirPort"
    $selection = Select-String -Path 'server-dir-out.log', 'server-dir-err.log' `
        -Pattern "\[jllama\] native backend '[^']*' loaded "
    if (-not $selection) {
        Dump-ServerLogs @('server-dir-out.log', 'server-dir-err.log')
        Write-Error "no loader line found - the loader did not report the backend (lib.path)"
    }
    $loaderLine = $selection[0].Line
    Write-Host "loader: $loaderLine"
    if ($loaderLine -notlike "*loaded from $nativesDir*") {
        Write-Error "the loader did not load from the lib.path directory ${nativesDir}: $loaderLine"
    }
    foreach ($module in $setModules) {
        if ($loaderLine -notmatch "GPU module\(s\) ([a-z0-9-]+, )*$module(,|;|$)") {
            Write-Error "the loader did not find the $module module next to the library: $loaderLine"
        }
    }
    Write-Host "lib.path round OK"
    Write-Host "smoke test PASSED"
} finally {
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
}
