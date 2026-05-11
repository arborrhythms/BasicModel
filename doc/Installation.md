# Installation and Usage

## Prerequisites

- **Python 3.12** with a virtual environment at `.venv/`
- **PyTorch** with MPS (Apple Silicon) or CUDA
- **pandoc** (optional, for PDF generation via `make doc_pdf`)

## Setup

```bash
make install   # creates .venv with dependencies
```

All Makefile targets invoke Python through `.venv/bin/python`, so no manual
activation is required.

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
| `make run` | Run `BasicModel.py` with the config in `XML1` |
| `make xor` | `data/XOR_exact.xml` |
| `make simple` | `data/simple.xml` |
| `make ergodic` | `data/ergodic.xml` |
| `make tomatoes` | `data/tomatoes.xml` |
| `make mnist` | `data/mnist.xml` |
| `make compare` | Side-by-side compare using `XML1` and `XML2` |
| `make SigmaPi` / `SymPercept` / `SPNN` | Run respective scripts |

### Testing and Documentation

| Target | Description |
|---|---|
| `make test` | Unit tests (forces `BASICMODEL_DEVICE=cpu`) |
| `make bench` | Training benchmarks (baseline) |
| `make doc_pdf` | Generate `BasicModel.pdf` via pandoc |

PDF chapters in order: `README.md`, `doc/Architecture.md`, `doc/BasicModel.md`,
`doc/Spaces.md`, `doc/Language.md`, `doc/Logic.md`, `doc/Ergodic.md`,
`doc/Training.md`, `doc/MachineMinds.md`, `doc/Params.md`, `doc/Installation.md`.

### Utilities

| Target | Description |
|---|---|
| `make clean` | Remove generated files (`BasicModel.pdf`) |
| `make all` | Default; alias for `make xor` |

---

## Makefile Variables

Override on the command line, e.g. `make run XML1=data/ergodic.xml`.

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `data/BasicModel.xml` | XML config for training |
| `XML1` | `data/simple.xml` | Primary config for `make run` / `compare` |
| `XML2` | `data/ergodic-only.xml` | Secondary config for `make compare` |
| `TRAIN_HOST` | *(empty)* | Remote training host |
| `TRAIN_USER` | `arogers` | SSH user |
| `TRAIN_KEY` | `~/.ssh/id_ed25519_arbormini` | SSH private key |
| `TRAIN_DIR` | `~/WikiOracle/basicmodel` | Working directory on remote |

---

## train.py Options

`train.py` orchestrates Phase 1 (embeddings) followed by Phase 2 (model
training).

### General

| Flag | Default | Description |
|---|---|---|
| `--model`, `-m` | `data/BasicModel.xml` | XML config |
| `--max-docs` | *(from XML)* | Override `maxDocs` |
| `--num-shards` | *(from XML)* | Override `numShards` |
| `--random-shards` | off | Pick random shard indices |
| `--force-embeddings` | off | Retrain embeddings even if file exists |
| `--log [FILE]` | off | Capture stdout/stderr to `.log` in `output/logs/`. Omit filename for timestamped name |
| `--profile` | off | Run Phase 2 under `cProfile`; writes `.prof` to `output/profiles/` |

### Remote Execution (SSH)

| Flag | Default | Description |
|---|---|---|
| `--host` | *(none)* | Remote host (e.g. `arbormini.local`); enables remote run |
| `--user` | `arogers` | SSH user |
| `--key-file` | `~/.ssh/id_ed25519_arbormini` | SSH private key |
| `--remote-dir` | `~/WikiOracle/basicmodel` | Working directory on remote |

---

## Environment Variables

| Variable | Description |
|---|---|
| `BASIC_MAX_DOCS` | Override `maxDocs` |
| `BASIC_NUM_SHARDS` | Override `numShards` |
| `BASICMODEL_DEVICE` | Force device, e.g. `cpu`. Used by `make test` |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | MPS memory limit; set to `0.0` by `train.py` |
| `PYTHONUNBUFFERED` | Set to `1` by `train.py` for real-time log streaming |

---

## Remote Training (ArborMini)

`train_remote` and `train_micro_remote` set `TRAIN_HOST=arbormini.local` and
delegate to `train.py --host`.

### Workflow

1. **Rsync** the project tree to `arbormini.local`, excluding `.venv/`,
   `__pycache__/`, `.DS_Store`, `output/`.

2. **SSH** and run `train.py` remotely with the same flags (minus `--host`):

   ```bash
   ssh -i ~/.ssh/id_ed25519_arbormini -t arogers@arbormini.local \
     "cd ~/WikiOracle/basicmodel && PYTHONUNBUFFERED=1 PYTHONPATH=bin .venv/bin/python bin/train.py ..."
   ```

3. **Output stays remote** under `~/WikiOracle/basicmodel/output/`.

### SSH Key Setup

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_arbormini
ssh-copy-id -i ~/.ssh/id_ed25519_arbormini arogers@arbormini.local
```

### Quick Start

```bash
# Sanity check: 500 docs, 1 shard, on ArborMini
make train_micro_remote

# Full run on ArborMini
make train_remote
```
