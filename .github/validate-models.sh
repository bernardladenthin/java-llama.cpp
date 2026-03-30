#!/bin/bash
# Validate that all required model files exist and are valid GGUF files
# GGUF files start with magic bytes: 0x47 0x47 0x55 0x46 ("GGUF")

set -e

MODELS=(
  "models/codellama-7b.Q2_K.gguf"
  "models/jina-reranker-v1-tiny-en-Q4_0.gguf"
  "models/AMD-Llama-135m-code.Q2_K.gguf"
)

echo "Validating model files..."
for model in "${MODELS[@]}"; do
  if [[ ! -f "$model" ]]; then
    echo "ERROR: Model not found: $model"
    exit 1
  fi

  # Check file size (must be > 4 bytes for magic header)
  size=$(stat -f%z "$model" 2>/dev/null || stat -c%s "$model" 2>/dev/null)
  if [[ $size -lt 4 ]]; then
    echo "ERROR: Model file too small (likely corrupted): $model (size: $size bytes)"
    exit 1
  fi

  # Check GGUF magic bytes: 0x47 0x47 0x55 0x46
  magic=$(xxd -p -l 4 "$model")
  if [[ "$magic" != "47475546" ]]; then
    echo "ERROR: Invalid GGUF magic bytes in $model (got: $magic, expected: 47475546)"
    exit 1
  fi

  echo "✓ $model ($(numfmt --to=iec-i --suffix=B $size 2>/dev/null || echo $size bytes))"
done

echo "All models validated successfully!"
