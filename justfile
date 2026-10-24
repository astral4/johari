set positional-arguments

# List the recipes
default:
    @just --list

# Build the shipped module into web/johari.wasm
[group('build')]
build:
    scripts/build-wasm.sh

# Build the module with phase hooks and fit diagnostics into web/johari.bench.wasm
[group('build')]
build-bench:
    FEATURES=bench scripts/build-wasm.sh

# Build the module with function names and list its largest functions
[group('build')]
size *args:
    scripts/build-wasm.sh profiling
    twiggy top web/johari.profiling.wasm "$@"

# Run tests
[group('check')]
test: build
    cargo test
    node --test

# Check formatting, docs, and lints
[group('check')]
lint:
    cargo fmt --check
    rustfmt --check scripts/fuel.rs
    RUSTDOCFLAGS=-Dwarnings cargo doc -p johari-core --document-private-items --no-deps
    cargo clippy -p johari-core --all-targets
    cargo clippy -p johari-core --all-targets --features bench
    cargo clippy -p johari-wasm --target wasm32-unknown-unknown
    cargo clippy -p johari-wasm --target wasm32-unknown-unknown --features bench
    cargo clippy -Zscript --manifest-path scripts/fuel.rs

# Run all tests and checks
[group('check')]
check: lint test

# Time a simulated session's calls on the shipped module
[group('perf')]
bench *args: build
    node web/bench.mjs "$@"

# Time each phase of a step on the bench module
[group('perf')]
phases *args: build-bench
    node web/phases.mjs "$@"

# Count instructions per phase and per call (pass `web/johari.wasm` for calls on the shipped module)
[group('perf')]
fuel *args: build build-bench
    cargo -Zscript scripts/fuel.rs "$@"

# Print a simulated session step by step to diff against another build
[group('compare')]
trace *args: build-bench
    node web/replay.mjs trace "$@"

# Summarize simulated sessions over many seeds to compare distributions between builds
[group('compare')]
sweep *args: build-bench
    node web/replay.mjs sweep "$@"
