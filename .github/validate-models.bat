@echo off
REM Validate that all required model files exist and are valid GGUF files
REM GGUF files start with magic bytes: 0x47 0x47 0x55 0x46 ("GGUF")

setlocal enabledelayedexpansion

set "MODELS=models\codellama-7b.Q2_K.gguf" "models\jina-reranker-v1-tiny-en-Q4_0.gguf" "models\AMD-Llama-135m-code.Q2_K.gguf"

echo Validating model files...
for %%M in (%MODELS%) do (
  if not exist "%%M" (
    echo ERROR: Model not found: %%M
    exit /b 1
  )

  REM Check file size using PowerShell
  for /f %%S in ('powershell -Command "(Get-Item '%%M').Length"') do set "size=%%S"

  if !size! lss 4 (
    echo ERROR: Model file too small (likely corrupted^): %%M (size: !size! bytes^)
    exit /b 1
  )

  REM Check GGUF magic bytes using PowerShell: 47475546 in hex = GGUF in ASCII
  for /f %%H in ('powershell -Command "[System.BitConverter]::ToString((Get-Content '%%M' -Encoding Byte -ReadCount 4)[0]) -replace '-',''"') do set "magic=%%H"

  if not "!magic!"=="47475546" (
    echo ERROR: Invalid GGUF magic bytes in %%M (got: !magic!, expected: 47475546^)
    exit /b 1
  )

  echo OK: %%M ^(!size! bytes^)
)

echo All models validated successfully!
exit /b 0
