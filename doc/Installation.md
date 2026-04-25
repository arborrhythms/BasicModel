# Installation and Usage

## Prerequisites

- **Python 3.12** with a virtual environment at `.venv/`
- **PyTorch** with MPS support (Apple Silicon) or CUDA
- **pandoc** (optional, for PDF generation via `make doc_pdf`)

## Setup

```bash
make install   # creates .venv with dependencies
```

This installs all Python dependencies (including PyTorch) into a
project-local virtual environment. All Makefile targets invoke Python
through `.venv/bin/python`, so there is no need to activate the
environment manually.

---

## Makefile Targets

### Training

| Target | Description |
|---|---|
| `make train` | Full training (Phase 1 embeddings + Phase 2 model), logged to `output/logs/` |
| `make train_micro` | Small run: 500 docs, 1 random shard, logged |
| `make train_remote` | Full training on `arbormini.local` via SSH |
| `make train_micro_remote` | Micro training on `arbormini.local` via SSH |

### Models

| Target | Description |
|---|---|
| `make run` | Run `BasicModel.py` with the config specified by `XML1` |
| `make xor` | Run with `data/XOR_exact.xml` |
| `make simple` | Run with `data/simple.xml` |
| `make ergodic` | Run with `data/ergodic.xml` |
| `make tomatoes` | Run with `data/tomatoes.xml` |
| `make mnist` | Run with `data/mnist.xml` |
| `make compare` | Compare two models side-by-side using `XML1` and `XML2` |
| `make SigmaPi` | Run `SigmaPi.py` |
| `make SymPercept` | Run `SymPercept.py` |
| `make SPNN` | Run `SPNN.py` |

### Testing and Benchmarks

| Target | Description |
|---|---|
| `make test` | Run unit tests (forces `BASICMODEL_DEVICE=cpu`) |
| `make bench` | Run training benchmarks (baseline, no env tweaks) |

### Documentation

| Target | Description |
|---|---|
| `make doc_pdf` | Generate `BasicModel.pdf` from doc chapters via pandoc |

The PDF is assembled from the following chapters in order:
`README.md`, `doc/Architecture.md`, `doc/BasicModel.md`, `doc/Spaces.md`,
`doc/Language.md`, `doc/Logic.md`, `doc/Ergodic.md`, `doc/Training.md`,
`doc/MachineMinds.md`, `doc/Params.md`, `doc/Installation.md`.

### Utilities

| Target | Description |
|---|---|
| `make clean` | Remove generated files (`BasicModel.pdf`) |
| `make all` | Default target; alias for `make xor` |

---

## Makefile Variables

All variables can be overridden on the command line,
e.g. `make run XML1=data/ergodic.xml`.

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `data/BasicModel.xml` | XML config file for training targets |
| `XML1` | `data/simple.xml` | Primary config for `make run` and `make compare` |
| `XML2` | `data/ergodic-only.xml` | Secondary config for `make compare` |
| `TRAIN_HOST` | *(empty)* | Remote training host; set automatically by `train_remote` targets |
| `TRAIN_USER` | `arogers` | SSH user for remote training |
| `TRAIN_KEY` | `~/.ssh/id_ed25519_arbormini` | SSH private key file |
| `TRAIN_DIR` | `~/WikiOracle/basicmodel` | Working directory on the remote host |

---

## train.py Options

`train.py` orchestrates the full training pipeline: Phase 1
(embeddings) followed by Phase 2 (model training).

### General

| Flag | Default | Description |
|---|---|---|
| `--model`, `-m` | `data/BasicModel.xml` | XML config file |
| `--max-docs` | *(from XML config)* | Override `maxDocs` -- limits the number of Wikipedia documents processed |
| `--num-shards` | *(from XML config)* | Override `numShards` -- number of embedding shards to train |
| `--random-shards` | off | Pick random shard indices instead of sequential, for variety across runs |
| `--force-embeddings` | off | Retrain embeddings even if the output file already exists |
| `--log [FILE]` | off | Capture stdout/stderr to a `.log` file in `output/logs/`. Omit the filename for an auto-generated timestamped name |
| `--profile` | off | Run Phase 2 under `cProfile`; writes a `.prof` file to `output/profiles/` and prints the top-30 cumulative-time functions |

### Remote Execution (SSH)

| Flag | Default | Description |
|---|---|---|
| `--host` | *(none)* | Remote host (e.g. `arbormini.local`). When set, training runs remotely instead of locally |
| `--user` | `arogers` | SSH user |
| `--key-file` | `~/.ssh/id_ed25519_arbormini` | SSH private key |
| `--remote-dir` | `~/WikiOracle/basicmodel` | Working directory on the remote machine |

---

## Environment Variables

| Variable | Description |
|---|---|
| `BASIC_MAX_DOCS` | Override `maxDocs` in `BasicModel.py` (set automatically by `train.py` when `--max-docs` is given) |
| `BASIC_NUM_SHARDS` | Override `numShards` in `BasicModel.py` (set automatically by `train.py` when `--num-shards` is given) |
| `BASICMODEL_DEVICE` | Force a specific device, e.g. `cpu`. Used by `make test` to avoid GPU-dependent failures |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | MPS memory allocation limit. Set to `0.0` by `train.py` to disable the high-watermark and allow PyTorch to use all available unified memory |
| `PYTHONUNBUFFERED` | Disable Python output buffering. Set to `1` by `train.py` so log output streams in real time |

---

## Remote Training (ArborMini)

The `train_remote` and `train_micro_remote` targets automate the
full remote-training workflow. Under the hood they set
`TRAIN_HOST=arbormini.local` and delegate to `train.py --host`.

### Workflow

1. **Rsync** syncs the local project tree to `arbormini.local`:

   ```bash
   rsync -av --progress \
     -e "ssh -i ~/.ssh/id_ed25519_arbormini" \
     --exclude .venv \
     --exclude __pycache__ \
     --exclude .DS_Store \
     --exclude output/ \
     ./  arogers@arbormini.local:~/WikiOracle/basicmodel/
   ```

   The exclusions keep the transfer fast: `.venv/` (the remote has its
   own), `__pycache__/` (bytecode), `.DS_Store` (macOS metadata), and
   `output/` (embeddings, logs, profiles).

2. **SSH** connects to ArborMini and runs `train.py` with the same
   flags that were passed locally (minus `--host`):

   ```bash
   ssh -i ~/.ssh/id_ed25519_arbormini -t arogers@arbormini.local \
     "cd ~/WikiOracle/basicmodel && PYTHONUNBUFFERED=1 PYTHONPATH=bin .venv/bin/python bin/train.py --model data/BasicModel.xml ..."
   ```

3. **Output stays remote.** Profile data (`.prof` files) and training
   logs remain on ArborMini under `~/WikiOracle/basicmodel/output/`.

### SSH Key Setup

Remote targets require passwordless SSH access. Generate and copy a
key if you have not already:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_arbormini
ssh-copy-id -i ~/.ssh/id_ed25519_arbormini arogers@arbormini.local
```

### Quick Start

```bash
# Fast sanity check: 500 docs, 1 shard, on ArborMini
make train_micro_remote

# Full training run on ArborMini
make train_remote
```
