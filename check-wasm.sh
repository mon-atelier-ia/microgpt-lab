#!/usr/bin/env bash
# Warns if wasm-pkg is stale (Rust source changed since last build).
# Run in pre-push hook or CI.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/model-rs"
HASH_FILE="$SCRIPT_DIR/app/wasm-pkg/.source-hash"

if [ ! -f "$HASH_FILE" ]; then
  echo "⚠ wasm-pkg/.source-hash missing — run ./build-wasm.sh"
  exit 1
fi

CURRENT=$(find "$MODEL_DIR/src" "$MODEL_DIR/crates" \( -name '*.rs' -o -name 'Cargo.toml' \) \
  | sort | xargs sha256sum | sha256sum | cut -d' ' -f1)
STORED=$(cat "$HASH_FILE")

if [ "$CURRENT" != "$STORED" ]; then
  echo "⚠ wasm-pkg is stale — Rust source changed since last build. Run ./build-wasm.sh"
  exit 1
fi

echo "✓ wasm-pkg is up to date"
