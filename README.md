# Johari

Johari is a tool for creating personal ranked lists of favorite Touhou characters.

## Usage

Follow the uv [installation instructions](https://docs.astral.sh/uv/getting-started/installation/), then set up the environment from the project root:

```console
uv sync
```

Then, run the program:

```console
uv run johari
```

## Configuration

- `--roster`: Use a custom roster of characters. Must be a file path. Default: [`touhou_windows.json`](src/johari/data/touhou_windows.json)
- `--seed`: Set a specific seed for reproducible results. Must be a non-negative integer. Default: `0`
- `--stop-threshold`: Set the threshold for automatically stopping the pairwise comparison stage. Higher values reduce the stage duration but result in higher uncertainty around ranks. Must be a floating-point number. Default: `0.03`

## How it works

Johari treats list-making as a Bayesian inference problem. Every character has a hidden score for how much you like them, and each answer you give is a noisy glimpse of those scores. A session opens with a quick "bucket" stage where you sort each character into "meh", "fine", or "love", or skip anyone you don't recognize. The model interprets those marks as an ordinal probit and learns cutpoints as it goes. This gives it a rough picture of your tastes before a single comparison is shown.

Then, the pairwise comparison stage refines that picture. Each head-to-head answer follows a Rao-Kupper comparison model. The formulation is a Bradley-Terry style contest with a learned indifference threshold (so "about the same" is a real answer), mixed with a small careless-answer rate so inconsistent stated preferences don't distort the entire list. After every answer, the posterior over all scores is refitted by a damped Newton solver and summarized with a Laplace approximation, with gradients and Hessian computed via JAX. Instead of asking pairs at random, Johari draws samples from that posterior, assembles many candidate pairs, and scores each by how much it is expected to shrink a top-weighted measure of ranking error. In other words, a mistake towards the top of the list costs more than one near the bottom, so the model prioritizes getting the top correct. Candidates are priced by reweighting the posterior draws under each possible answer, so looking one step ahead never requires a refit. Once several proposals in a row are not worth asking according to a predetermined threshold, the session stops on its own.

The final list is the ordering that minimizes the expected number of top-weighted inversions, found by local search starting from the expected-rank order. Characters are shown as tied when the model believes their gap falls inside your own indifference threshold. Every entry also comes with an 80% confidence interval on its rank.
