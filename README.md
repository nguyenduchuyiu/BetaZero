# BetaZero

BetaZero is a proof-search framework for Lean 4 that combines large language
models with an AND/OR graph search. It generates tactics and proof skeletons,
verifies them against a persistent Lean REPL, and scores environment and
dependency rewards. Supervised fine-tuning of the tactic and skeleton adapters
lives under `data_mining/`.

## Project Layout

```
gammazero/   Core library (search graph, rollout, policy, env, utils)
configs/     YAML configs for runs and models
problems/    Lean theorems to roll out
repl/        Lean REPL workspace (Mathlib + AST/expr dump executables)
outputs/     Run logs, checkpoints, rollout JSON
analysis/    Post-hoc metrics and figures
```

## Requirements

- Python 3.12+
- Lean 4 toolchain with `lake` on PATH
- Git LFS for binary assets (adapters, datasets)
- A CUDA-capable GPU when running vLLM or local PEFT training

## Setup

### 1. Install Lean 4 via elan

`elan` is the Lean version manager; it installs `lean` and `lake` and pins the
toolchain declared by `repl/lean-toolchain`.

```bash
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y
source $HOME/.elan/env
```

Verify the install:

```bash
lean --version
lake --version
```

### 2. Install Python dependencies and pull LFS assets

```bash
git lfs install
git lfs pull
pip install -r requirements.txt
```

### 3. Build the Lean REPL workspace

```bash
cd repl
lake update
lake exe cache get
lake build
lake build dump_ast_server
lake build dump_expr_server
```

`lake exe cache get` downloads the prebuilt Mathlib oleans, which is
significantly faster than compiling from source.

## Running a Rollout

The entry point is `run_rollout.py`. It loads a config, starts a policy server
(vLLM, DeepSeek API, or Gemini API), and runs best-first AND/OR search on each
input theorem.

Single file:

```bash
python3 run_rollout.py \
    --config configs/api.yaml \
    --lean-file problems/aime_1984_p5.lean \
    --policy gemini \
    --out-json outputs/rollouts/
```

Directory of theorems:

```bash
python3 run_rollout.py \
    --config configs/api.yaml \
    --lean-dir problems/miniF2F-test \
    --policy gemini \
    --out-json outputs/rollouts/miniF2F-test
```

Flags:

- `--config` path to a YAML config under `configs/`
- `--lean-file` / `--lean-dir` mutually exclusive input selection
- `--policy` one of `vllm`, `deepseek`, `gemini`
- `--adapter` adapter directory (default: `adapter-deepseek-qwen-7b`)
- `--out-json` output file or directory for rollout graphs

API keys for hosted policies are read from environment variables
(`DEEPSEEK_API_KEY`, `GOOGLE_API_KEY`). Place them in `.env` or export them
before running.

## Monitoring

Training and rollout metrics are written to `outputs/runs/<run_name>`.

```bash
tensorboard --logdir outputs/runs/
```

## Visualizing Rollouts

A static HTML viewer renders saved rollout graphs.

```bash
./serve.sh
```

Then open `http://localhost:1234/and_or_graph.html` and load any JSON file from
`outputs/rollouts/`.
