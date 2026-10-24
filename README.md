# Johari

Johari is a tool for creating personal ranked lists of favorite items.

## How it works

Johari approaches list-making as a Bayesian inference problem. Every item has a hidden score for how much you like them, and each answer you give is a noisy glimpse of those scores. Sessions start with a brief "bucket" stage where you sort each item into tiers or skip them. This gives the model a rough picture of your preferences before any comparisons are shown.

Then, the pairwise comparison stage refines that picture. For each head-to-head answer, the model assumes you perceive the gap between the two items' scores. You pick one of the options if the gap clears a learned indifference threshold, or answer "about the same" if it doesn't. A small proportion of answers is taken to be "careless" (random selection that doesn't relate to true preferences), so inconsistencies can't distort the entire list.

Instead of asking pairs at random, the model calculates what an answer to each pair not yet shown would be worth, then asks the one whose answer is expected to do the most for the final list. A mistake about items toward the top costs more than one about items near the bottom, so the model prioritizes getting the top correct. The application automatically stops asking once the value of the best remaining pair drops below a threshold.

Credible intervals are computed for each item's rank in the final list. items whose estimated scores are sufficiently close are treated as tied at the same rank.

## Development

The model is implemented in Rust and compiled to WebAssembly for a website frontend.

- `crates/johari-core`: core functionality as a `no_std` library; includes the model, fit, selection, and ranking.
- `crates/johari-wasm`: a thin `cdylib` ABI wrapper with panic handling and memory allocation.
- `web/`: `johari.js` for wrapping exports; `worker.js` for hosting the module in a Web Worker; `client.js` for the main-thread client.

Building the project requires [rustup/Cargo](https://www.rust-lang.org/tools/install) and [binaryen](https://github.com/WebAssembly/binaryen) (`wasm-opt`). Running tests and benchmarks additionally requires [Node](https://nodejs.org/en/download). Use [just](https://github.com/casey/just) and see [justfile](justfile) for basic commands and workflows.

```console
just --list    # list all recipes
just build     # build for wasm32 into web/johari.wasm
just check     # run tests; check formatting, docs, and lints
just bench     # time a simulated session's calls
just phases    # time each phase of a step
just fuel      # count instructions per phase and per call
just trace     # trace a simulated session step by step to diff against another build
just sweep     # summarize simulated sessions over 20 seeds to compare distributions
just size      # list the module's largest functions with twiggy
```
