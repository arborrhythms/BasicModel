# Installation and Usage

## Prerequisites

- **Python 3.12** with a virtual environment at `.venv/`
- **PyTorch** with MPS (Apple Silicon) or CUDA
- **pandoc** (optional, for PDF generation via `make doc`)

## Conceptual Orientation

The installable artifact is not only an LLM training stack. It runs BasicModel's
explicit architecture: LLM-like prediction/reconstruction objectives,
Formal Concept Analysis-like part/whole concept order, and DisCoCat-like typed
grammar composition. The generated PDF includes the detailed explanation in
[Architecture.md](Architecture.md), [Mereology.md](Mereology.md), and
[Language.md](Language.md).

## Setup

```bash
make install   # creates .venv with dependencies
```

Makefile targets invoke the project virtual environment (`.venv/bin/python` on
Unix, `.venv/Scripts/python.exe` on Windows), so no manual activation is
required.

**Tested build (Alec, 2026-11-01): the latest stable releases, unpinned.**
`requirements.txt` names packages without versions, and `make install`
takes whatever PyPI serves as stable; what an upgrade breaks is fixed in
the code rather than pinned around. The suite is the criterion, and it
last ran on torch 2.15.0, torchvision 0.29.2, executorch 1.5.1 and pytest
9.2.0 on Python 3.12. Verify an environment with
`.venv/bin/python -m pip check`. The parent WikiOracle checkout's
`requirements.txt` includes this file, so one `pip install --upgrade -r
requirements.txt` there brings both to the same releases.

Note that `make test` depends on the venv stamp
(`.venv/.installed-bin`): when `requirements.txt` is newer than the stamp,
the target rebuilds `.venv` from scratch before testing, which upgrades
every unpinned package. Run `make -n test` first if that is not wanted.

The earlier tested path, the PyTorch 2.14 nightly pair
(`torch==2.14.0.dev20260722` with `torchvision==0.29.0.dev20260723` from
the nightly CPU index), compiled the complete W=512 tensor peer loop with
raw `max-autotune`, differentiated backward, and runtime word counts
through the configured sentence capacity without recompilation. That
measurement has not been repeated on the stable builds.

**MLX export on this stack (2026-11-01).** With torch 2.15.0 and executorch
1.5.1, `RUN_SLOW=1 pytest test/test_mlx_export.py` reports 3 passed and 2
xfailed, but the two xfails (D2, the `.pte` lowering; D3, runtime parity)
no longer fail for the recorded reason, the dedup'd constant shared across
the three delegated combine subgraphs at sO=3. They stop earlier, in
executorch's `to_edge_transform_and_lower` verifier: `SpecViolationError:
Parameter output aten_mean_dim does not point to a parameter that exists`,
the mutated parameter being `symbolSpace._owned_bases.what.W`. The exported
core writes a parameter during forward, which this verifier rejects. Whether
torch 2.15's export, executorch 1.5.1's verifier or the item 2.5 working
tree exposed it is not established; the upstream shared-constant limitation
is therefore unverified on 1.5.1. The non-strict xfails hide this, since
pytest reports an xfail whatever the failure.

---

## Makefile Targets

### Training

| Target | Description |
|---|---|
| `make train` | Full training (Phase 1 embeddings + Phase 2 model), logged to `output/logs/`; pins `--data text` (overrides the XML `<dataset>`). On macOS the process tree is capped at 60% of physical RAM by default. |
| `make train_micro` | Smoke run: max 1,000 docs, one random shard, ten batches, logged; also pins `--data text`, caps `--num-epochs 1`, and uses the same memory bounds as `make train`. |
| `make preflight` | Fast FineWeb launch/config/data/checkpoint gates |
| `make preflight_full` | Preflight plus one real N=64 optimizer step |

### Models

| Target | Description |
|---|---|
| `make run` | Run `Models.py` with the config named by `XML1` |
| `make xor` | `data/MM_xor.xml` |
| `make mnist` | `data/mnist.xml`, the non-text case; needs the MNIST files (below) |
| `make ergodic` | `data/ergodic.xml`: the MNIST model with `ErgodicLayer` exploration and per-neuron certainty on; `data/ergodic-only.xml`, the same without certainty, is its comparison arm |
| `make compare` | Side-by-side compare of `XML1` and `XML2`; writes an HTML report |
| `make SigmaPi` / `SymPercept` / `SPNN` | Run the legacy demos under `bin/etc/` with `--demo` |

The MNIST training file `data/mnist_train.csv` is a Git LFS object. Fetch it
with `git lfs pull --include=data/mnist_train.csv` before an MNIST run; the
loader names any MNIST file whose LFS content is not checked out.

### Testing and Documentation

| Target | Description |
|---|---|
| `make test` | The default suite through the bounded runner (`test/test_report.py`, `BASICMODEL_DEVICE=cpu`); the commit gate |
| `make testp` | Same suite, parallel via pytest-xdist (`TEST_JOBS`); faster iteration, no bounded run |
| `make test_all` | Same suite plus slow-gated tests (`RUN_SLOW=1`) |
| `make test_slow_weekly` | The weekly slow tier on a frozen copy of the source (`test/slow_weekly.py`) |
| `make bench` | Training benchmarks (baseline) |
| `make bench_local` | Reconstruction fidelity/timing bench, local (cpu+eager for reproducibility) |
| `make bench_sync` | rsync the working tree to the ArborStudio remote (excludes venvs, caches, large regenerable data) |
| `make bench_remote` | `bench_sync` then run the reconstruction bench on ArborStudio's native path |
| `make bench_pull` | rsync bench result JSON/profile files back from ArborStudio to `output/` |
| `make doc` | Generate `BasicModel.pdf` via pandoc |

PDF chapters in order: `README.md`, `doc/Installation.md`, `doc/Architecture.md`,
`doc/Componentization.md`, `doc/BasicModel.md`, `doc/Spaces.md`, `doc/STM.md`,
`doc/Language.md`, `doc/Orders.md`, `doc/Mereology.md`, `doc/Logic.md`,
`doc/Reasoning.md`, `doc/Training.md`, `doc/GradientFlow.md`, `doc/Ergodic.md`,
`doc/MachineMinds.md`, `doc/Params.md`, `doc/FutureWork.md`,
`doc/DevelopmentHistory.md`.

`make bench_local` / `bench_remote` set `MODEL` to `data/MM_ladder.xml` (a
compact grammar fixture) rather than the FineWeb default. Bench variables:

| Variable | Default | Description |
|---|---|---|
| `EPOCHS` | `3` | Epoch count passed to `recon_bench.py` |
| `REMOTE_COMPILE` | `auto` | `MODEL_COMPILE` value used on `bench_remote` |
| `ARBOR_USER` | `arogers` | SSH user for the ArborStudio remote |
| `ARBOR_HOST` | `ArborStudio.local` | ArborStudio hostname (LAN) |
| `ARBOR_KEY` | `~/.ssh/id_ed25519_arborstudio` | SSH key for the remote |
| `ARBOR_DEST` | `~/WikiOracle/basicmodel` | Remote working directory |
| `ARBOR_TUNNEL` | `0` | Set to `1` to route off-LAN via the wikiOracle.org reverse tunnel instead of direct LAN SSH |

### Utilities

| Target | Description |
|---|---|
| `make clean` | Remove generated files (`BasicModel.pdf` and `output/*`) |
| `make all` | Default; alias for `make xor` |

---

## Makefile Variables

Override on the command line, e.g.
`make compare XML1=data/ergodic.xml XML2=data/ergodic-only.xml`.

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `data/BasicModel.xml` | XML config for `make train`; `data/MM_ladder.xml` for the bench targets |
| `XML1` | *(none)* | Config for `make run`, and the first config for `make compare` |
| `XML2` | *(none)* | Second config for `make compare` |
| `TRAIN_ARGS` | *(empty)* | Extra `train.py` flags for `make train` |
| `TRAIN_MEMORY_PERCENT` | `60` | macOS process-tree memory limit for `make train`, as a percentage of physical RAM |
| `TEST_ARGS` | *(empty)* | Selection and runner flags passed to the test targets |

---

## train.py Options

`train.py` orchestrates Phase 1 (embeddings) followed by Phase 2 (model
training).

### General

| Flag | Default | Description |
|---|---|---|
| `--model`, `-m` | `data/BasicModel.xml` | XML config |
| `--data` | *(from XML)* | Override dataset/data source selector |
| `--max-docs` | *(from XML)* | Override `maxDocs` |
| `--num-shards` | *(from XML)* | Override `numShards` |
| `--num-epochs` | *(from XML)* | Override training epochs |
| `--batch-size` | *(from XML)* | Override `batchSize`; forwarded as `BASIC_BATCH_SIZE` |
| `--max-tokens` | *(from XML)* | Override token budget |
| `--batches` | *(from XML)* | Cap Phase 2 batches |
| `--max-seconds` | off | Stop training cleanly after this many wall-clock seconds, between completed batches |
| `--test [N]` | off | Run evaluation passes; optional `N` caps test batches |
| `--compile-target` | `gpu` | `gpu` runs normal training with accelerator torch.compile defaults; `mlx` exports an ExecuTorch/MLX `.pte` and skips Phase 2 training |
| `--compile-mode` | *(env/runtime default)* | Forwarded as `MODEL_COMPILE_MODE` |
| `--mlx-output` | `output/mlx/<model>.pte` | Destination for `--compile-target mlx` |
| `--random-shards` | off | Pick random shard indices |
| `--force-embeddings` | off | Retrain embeddings even if file exists |
| `--latent-vector-size` | *(model/vector size)* | Phase 1 SBOW latent size before optional projection |
| `--embed-lr` | *(embed default)* | Phase 1 SBOW learning rate |
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
| `BASIC_DATASET` | Override dataset/data source selector |
| `BASIC_MAX_DOCS` | Override `maxDocs` |
| `BASIC_NUM_SHARDS` | Override `numShards` |
| `BASIC_NUM_EPOCHS` | Override `numEpochs` |
| `BASIC_BATCH_SIZE` | Override `batchSize` |
| `BASIC_MAX_TOKENS` | Override token budget |
| `BASIC_MAX_BATCHES` | Cap Phase 2 batches |
| `BASIC_MAX_SECONDS` | Stop training after this many wall-clock seconds |
| `BASIC_RANDOM_SHARDS` | Set to `1` to pick random shard indices |
| `BASIC_RUN_TEST` | Request evaluation passes; optional value caps test batches |
| `BASICMODEL_DEVICE` | Force device, e.g. `cpu`. Used by `make test` and the `bench_*` targets |
| `BASICMODEL_PYTHON` | Python executable for train.py subprocesses; overrides the in-project `.venv` lookup |
| `MODEL_COMPILE` | torch.compile backend selector (`auto`, `none`, `inductor`, `eager`, `aot_eager`) |
| `MODEL_COMPILE_MODE` | torch.compile mode (`default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`). When unset, MPS uses `default`. PyTorch 2.12/2.13 MPS safely reduce CUDA-graph-bearing requests to their non-CUDAGraph equivalent because their reverse `while_loop` scheduler can crash while releasing a Metal command buffer. PyTorch 2.14+ MPS and real CUDA retain the requested mode unchanged; MPS still skips actual CUDA-only graph capture. |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | Hard MPS allocator watermark; defaults to `0.60` while preserving an explicit caller value |
| `PYTORCH_MPS_LOW_WATERMARK_RATIO` | Soft MPS allocator watermark; defaults to `0.50` while preserving an explicit caller value |
| `TRAIN_MEMORY_PERCENT` | macOS `make train`/`make train_micro` process-tree memory limit as a percentage of physical RAM; defaults to `60` |
| `BASICMODEL_MPS_IOBUF` | MPS inductor fusion cap (`max_fusion_unique_io_buffers`, default 24) — keeps fused Metal kernels under the hardware 31-buffer kernel-arg limit; lower (e.g. 12) for wide configs |
| `BASICMODEL_MPS_FUSE` | Optional MPS inductor `max_fusion_size` (nodes per fusion); unset leaves the torch default |
| `PYTHONUNBUFFERED` | Set to `1` by `train.py` for real-time log streaming |
| `RUN_SLOW` | Set to `1` to also run slow-gated tests; used by `make test_all` |
| `RUN_SLOW_FRACTION` | Fraction of the repeated work that budgeted slow fixtures run; `0.5` by default under `RUN_SLOW=1` |
| `RUN_TRAINED` | Set to `1` to run the trained tier, which requires registered phase artifacts ([Testing](Testing.md)) |
| `TEST_JOBS` | pytest-xdist worker count for `make testp` (e.g. `auto`) |

Additional env vars read directly by `Models.py`/`util.py` (not surfaced as
`train.py` flags): `BASIC_SEED` (overrides the training seed), `BASIC_CHECKPOINT_EVERY_BATCHES`
(checkpoint frequency in batches, default off), `BASIC_PROFILE` (enable
`cProfile` around Phase 2, also set by `train.py --profile`), `MODEL_AMP`
(autocast mode: `off`/`fp16`/`bf16`), `MODEL_DEBUG` (gates verbose debug
asserts/stat prints).

---

## Private Remote Training

There is no `Makefile.local` include mechanism today: the `bench_local`/`bench_sync`/
`bench_remote`/`bench_pull` targets hardcode one machine-specific SSH target
(`ARBOR_HOST`/`ARBOR_USER`/`ARBOR_KEY`/`ARBOR_DEST`, all overridable on the
command line — see the Bench Variables table above) directly in the Makefile.
The public `train.py --host` mode remains available for generic SSH execution
when a caller supplies the host, user, optional key, and remote directory
explicitly.
