#!/usr/bin/env sh
# Build the wasm shell into web/johari.wasm.

set -eu
unset RUSTFLAGS CARGO_ENCODED_RUSTFLAGS
cd "$(dirname "$0")/.."
mode="${1:-release}"
crate_features="${FEATURES:-}"
name="johari${crate_features:+.$crate_features}"
names=
case "$mode" in
  release) ;;
  profiling)
    name="$name.profiling"
    names=1
    ;;
  *) echo "usage: $0 [release|profiling]" >&2; exit 2 ;;
esac
command -v wasm-opt >/dev/null 2>&1 || { echo 'wasm-opt not found: install binaryen' >&2; exit 1; }
artifacts="target/wasm-artifact/$name"

cargo build --release -p johari-wasm --target wasm32-unknown-unknown -Z build-std=core,alloc \
  -Z unstable-options --artifact-dir "$artifacts" \
  --config 'target.wasm32-unknown-unknown.rustflags=["-Zunstable-options", "-Cpanic=immediate-abort"]' \
  ${crate_features:+--features "$crate_features"}
wasm-opt -Os -ocimfs=1000 --low-memory-unused --converge ${names:+-g} --strip-target-features --strip-producers \
  "$artifacts/johari_wasm.wasm" -o "web/$name.wasm"
printf '%s bytes (%s gzipped): web/%s\n' "$(wc -c < "web/$name.wasm" | tr -d ' ')" "$(gzip -9 -c "web/$name.wasm" | wc -c | tr -d ' ')" "$name.wasm"
