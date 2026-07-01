# Installation and Usage

## Prerequisites

- **Python 3.12** with a virtual environment at `.venv/`
- **PyTorch** with MPS (Apple Silicon) or CUDA
- **pandoc** (optional, for PDF generation via `make doc`)

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
| `make train_micro` | Small run: max 1,000,000 docs across 10 random shards, logged |

### Models

| Target | Description |
|---|---|
| `make run` | Run `Models.py` with the config in `XML1` |
| `make xor` | `data/MM_xor.xml` |
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
| `make doc` | Generate `BasicModel.pdf` via pandoc |

PDF chapters in order: `README.md`, `doc/Installation.md`, `doc/Architecture.md`,
`doc/BasicModel.md`, `doc/Spaces.md`, `doc/STM.md`, `doc/Language.md`,
`doc/Mereology.md`, `doc/Logic.md`, `doc/Reasoning.md`, `doc/Training.md`,
`doc/Ergodic.md`, `doc/MachineMinds.md`, `doc/Params.md`.

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
| `MODEL` | `data/MM_20M_legacy.xml` | XML config for training |
| `XML1` | `data/simple.xml` | Primary config for `make run` / `compare` |
| `XML2` | `data/ergodic-only.xml` | Secondary config for `make compare` |

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
| `--compile-target` | `gpu` | `gpu` runs normal training with accelerator torch.compile defaults; `mlx` exports an ExecuTorch/MLX `.pte` and skips Phase 2 training |
| `--compile-mode` | *(env/runtime default)* | Forwarded as `MODEL_COMPILE_MODE` |
| `--mlx-output` | `output/mlx/<model>.pte` | Destination for `--compile-target mlx` |
| `--random-shards` | off | Pick random shard indices |
| `--force-embeddings` | off | Retrain embeddings even if file exists |
| `--log [FILE]` | off | Capture stdout/stderr to `.log` in `output/logs/`. Omit filename for timestamped name |
| `--profile` | off | Run Phase 2 under `cProfile`; writes `.prof` to `output/profiles/` |

### Remote Execution (SSH)

| Flag | Default | Description |
|---|---|---|
| `--host` | *(none)* | Remote host; enables remote run |
| `--user` | `arogers` | SSH user |
| `--key-file` | *(none)* | Optional SSH key file; uses SSH config/agent when omitted |
| `--remote-dir` | `~/WikiOracle/basicmodel` | Working directory on remote |

---

## Environment Variables

| Variable | Description |
|---|---|
| `BASIC_MAX_DOCS` | Override `maxDocs` |
| `BASIC_NUM_SHARDS` | Override `numShards` |
| `BASICMODEL_DEVICE` | Force device, e.g. `cpu`. Used by `make test` |
| `BASICMODEL_PYTHON` | Python executable for train.py subprocesses; overrides the in-project `.venv` lookup |
| `MODEL_COMPILE` | torch.compile backend selector (`auto`, `none`, `inductor`, `eager`, `aot_eager`) |
| `MODEL_COMPILE_MODE` | torch.compile mode (`default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`) |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | MPS memory limit; set to `0.0` by `train.py` |
| `PYTHONUNBUFFERED` | Set to `1` by `train.py` for real-time log streaming |

---

## Private Remote Training

Machine-specific SSH targets, LAN hostnames, and local training shortcuts should
live in the ignored top-level `Makefile.local`. The public `train.py --host`
mode remains available for generic SSH execution when a caller supplies the
host, user, optional key, and remote directory explicitly.
