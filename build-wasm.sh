#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/model-rs"
OUT_DIR="$SCRIPT_DIR/app/wasm-pkg"

echo "Building microgpt-wasm..."
cd "$MODEL_DIR"
wasm-pack build crates/microgpt-wasm --target web --out-dir "$OUT_DIR"
rm -f "$OUT_DIR/.gitignore"

# Store source hash for staleness detection.
find "$MODEL_DIR/src" "$MODEL_DIR/crates" \( -name '*.rs' -o -name 'Cargo.toml' \) \
  | sort | xargs sha256sum | sha256sum | cut -d' ' -f1 > "$OUT_DIR/.source-hash"

SIZE=$(wc -c < "$OUT_DIR/microgpt_wasm_bg.wasm")
echo "Done. WASM output: $OUT_DIR ($SIZE bytes)"
