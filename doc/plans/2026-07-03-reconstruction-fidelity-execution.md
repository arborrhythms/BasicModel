# Reconstruction Fidelity + Config Matrix + Speed — Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax. **Alec does ALL git writes** — no commit steps
> below; each GATE is where he commits. Design doc (APPROVED):
> [2026-07-03-reconstruction-fidelity-design.md](2026-07-03-reconstruction-fidelity-design.md).

**Goal:** Exact round-trip reconstruction for MM_20M_xor and (post
meronomy/meronomy switch) MM_20M_grammar, a canonical config-matrix test
suite, and a production-throughput speed pass on ArborStudio — with one
shared harness so every debug run records a benchmark sample.

**Architecture:** New `bin/recon_bench.py` harness (build config, run $N$
epochs via `Models.runEpoch`, decode reconstruction to text via
`reconstruct_data(text=True)`, emit a JSON run record with fidelity +
timing). Probes on MM_20M_xor root-cause the three deferred symptoms; fixes
land TDD in `test/test_reconstruction_roundtrip.py`. Remote runs reuse the
parent `Makefile.local` `ARBORSTUDIO_*` rsync/ssh machinery (native path —
no iCloud, no spaces).

**Tech stack:** torch (eager local / `MODEL_COMPILE=auto` candidate on
ArborStudio), pytest, torch.profiler, rsync/ssh.

**House rules for every task:** no git writes; comments one-liners;
targeted pytest only (`PYTHONPATH=test .venv/bin/python -m pytest ...`);
`make test` only at gates on a quiet tree; Inf/NaN fail loud; LaTeX math in
docs; probe scripts in scratchpad unless promoted deliberately.

---

## Task 1 — The shared harness: `bin/recon_bench.py`

**Files:**
- Create: `bin/recon_bench.py`
- Test: `test/test_recon_bench.py` (new)
- Read first: `test/bench_throughput.py` (timing-loop pattern),
  `test/_sweep_chunking.py` (config-rewrite pattern), `bin/Models.py`
  `runEpoch` (~4990: returns `(output_loss, reconstruction_loss,
  all_predictions, last_reconstruction)`), `bin/Spaces.py:5199`
  `reconstruct_data(decoded, text=False)` and the InputSpace variant at
  ~11903 (`reconstruct_data(text=True)`) — pick the site that renders the
  reconstruction to a string; `_d3_reconstruction_loss` (bin/Models.py
  ~9261) for the nearest-row word-match metric to reuse.

- [ ] **1.1 Write the failing test** (`test/test_recon_bench.py`):

```python
import json
from recon_bench import run_config, RunRecord


def test_run_record_schema_smoke(tmp_path):
    """One epoch on the tiny xor fixture yields a complete run record."""
    rec = run_config("data/MM_xor_fixture.xml", epochs=1, seed=0,
                     out_dir=str(tmp_path))
    assert isinstance(rec, RunRecord)
    d = json.loads((tmp_path / rec.filename).read_text())
    for key in ("config", "seed", "epochs", "wall_s_per_epoch",
                "output_loss", "recon_loss", "exact_match_rate",
                "where_recovery", "channel_losses", "device", "compile_mode",
                "host", "timestamp"):
        assert key in d, key
    assert 0.0 <= d["exact_match_rate"] <= 1.0
    assert d["wall_s_per_epoch"] > 0


def test_exact_match_rate_is_1_on_identity():
    """Feeding the decoded target back as reconstruction scores 1.0."""
    from recon_bench import exact_match_rate
    assert exact_match_rate(["a b c", "d e"], ["a b c", "d e"]) == 1.0
    assert exact_match_rate(["a b c"], ["a b X"]) == 0.0
```

(Use the actual small fixture the repo has — `data/MM_xor_fixture.xml`
exists; verify it runs an epoch quickly, else substitute `data/xor.xml` and
note it.)

- [ ] **1.2 Run to verify failure:**
  `PYTHONPATH=test:bin .venv/bin/python -m pytest test/test_recon_bench.py -x -q`
  — expect ImportError.

- [ ] **1.3 Implement `bin/recon_bench.py`:**

```python
"""Shared fidelity+timing harness (2026-07-03 recon plan): every debug run
is a benchmark sample. CLI: recon_bench.py <config.xml> [--epochs N]
[--seed S] [--out output/] [--profile]."""
import argparse, dataclasses, json, os, socket, time


@dataclasses.dataclass
class RunRecord:
    config: str
    seed: int
    epochs: int
    wall_s_per_epoch: float
    output_loss: float
    recon_loss: float
    exact_match_rate: float
    where_recovery: float
    channel_losses: dict
    device: str
    compile_mode: str
    host: str
    timestamp: str

    @property
    def filename(self):
        base = os.path.splitext(os.path.basename(self.config))[0]
        return f"recon_{base}_{self.timestamp}.json"


def exact_match_rate(targets, decoded):
    """Fraction of presentations whose decoded reconstruction is IDENTICAL."""
    if not targets:
        return 0.0
    hits = sum(1 for t, d in zip(targets, decoded) if t == d)
    return hits / len(targets)
```

plus `run_config(config, epochs, seed, out_dir, profile=False)`:
seed torch/python RNGs; build the model the way `test/bench_throughput.py`
does (ModelFactory from the config path); loop epochs calling
`m.runEpoch(...)` timing each with `time.perf_counter()`; after the last
epoch decode `last_reconstruction` to text via the InputSpace
`reconstruct_data(text=True)` path and collect the target texts for the
same rows (the unmasked `_ar_embedded` source — mirror how
`_d3_reconstruction_loss` obtains its target); compute `exact_match_rate`;
`where_recovery` = the fraction of reconstructed `.where` spans that match
the true tiling spans (reuse `_d3_reconstruction_loss`'s nearest-row
machinery; if spans are unavailable pre-Task-4, record `-1.0` and a
`"where_recovery_note": "pre-fix placeholder"` field — the Task 4 probes
replace it); `channel_losses` = the what/where/when components if exposed,
else `{}` with a note; `timestamp` = `time.strftime("%Y%m%d-%H%M%S")`
taken ONCE at start; write the JSON to `out_dir`. `--profile` wraps the
epoch loop in `torch.profiler.profile` (pattern in `bin/train.py:52`) and
writes `key_averages().table(sort_by="self_cpu_time_total", row_limit=25)`
next to the JSON.

- [ ] **1.4 Run tests to verify pass** (same command as 1.2). Also run the
  CLI once by hand:
  `PYTHONPATH=bin .venv/bin/python bin/recon_bench.py data/MM_xor_fixture.xml --epochs 2 --out output/`
  and eyeball the JSON.

---

## Task 2 — Remote mode: ArborStudio round-trip

**Files:**
- Modify: `Makefile` (basicmodel — add `bench_local`, `bench_remote`,
  `bench_pull` targets)
- Read first: parent `../Makefile.local` (`ARBORSTUDIO_USER/HOST/KEY_FILE/
  DEST/SYNC_OPTS`), parent `Makefile` sync targets (~lines 90-175).

- [ ] **2.1 Add Make targets** (basicmodel/Makefile, following its existing
  style; `MODEL ?= data/MM_20M_grammar.xml`, `EPOCHS ?= 3`):

```make
ARBOR_SSH  = ssh -i ~/.ssh/id_ed25519_arborstudio arogers@ArborStudio.local
ARBOR_DEST ?= ~/WikiOracle/basicmodel

bench_local:
	PYTHONPATH=bin $(VENV_PYTHON) bin/recon_bench.py $(MODEL) --epochs $(EPOCHS) --out output/

bench_sync:
	rsync -av --progress -e "ssh -i ~/.ssh/id_ed25519_arborstudio" \
	  --exclude .venv --exclude output --exclude .git --exclude .claude \
	  ./ arogers@ArborStudio.local:$(ARBOR_DEST)/

bench_remote: bench_sync
	$(ARBOR_SSH) "cd $(ARBOR_DEST) && make install_if_needed && \
	  PYTHONPATH=bin .venv/bin/python bin/recon_bench.py $(MODEL) --epochs $(EPOCHS) --out output/"

bench_pull:
	rsync -av -e "ssh -i ~/.ssh/id_ed25519_arborstudio" \
	  "arogers@ArborStudio.local:$(ARBOR_DEST)/output/recon_*.json" output/
```

(`install_if_needed`: reuse/alias the existing `make install` guarded by a
`.venv` existence check — read the current install target and mirror it;
if the remote venv story turns out different, STOP and report
NEEDS_CONTEXT rather than inventing one.)

- [ ] **2.2 Smoke the remote path** (requires the host up; if
  `ArborStudio.local` is unreachable, report it and mark this step
  deferred-not-failed): `make bench_remote MODEL=data/MM_xor_fixture.xml
  EPOCHS=1` then `make bench_pull` — a `recon_*.json` with
  `"host": "ArborStudio..."` lands in `output/`. Record whether
  `MODEL_COMPILE=auto` works there (no path-space CppCompileError on the
  native path) — just note it; the speed pass exploits it in Task 8.

---

## Task 3 — Baselines (GATE 1 after this)

- [ ] **3.1 Local baselines**: `make bench_local` for
  `data/MM_20M_xor.xml`, `data/MM_20M_grammar.xml` (pre-switch),
  `data/MM_20M_legacy.xml` — pick `EPOCHS` so each run stays under ~15 min
  (start with 3; use `BASIC_MAX_BATCHES` if a single epoch is too long,
  and record the cap in the JSON's config field notes).
- [ ] **3.2 Remote baselines**: same three via `make bench_remote` (skip
  gracefully if host unreachable; Gate 1 then carries local-only baselines
  with a note).
- [ ] **3.3 Record** a baseline table (config x host: s/epoch, recon_loss,
  exact_match_rate) in this plan's EXECUTION NOTES section as it develops.

**GATE 1 (Alec commits):** harness + targets + `test/test_recon_bench.py`
green + baseline table recorded.

---

## Task 4 — XOR probes: root-cause the three symptoms

**Files:**
- Create (scratchpad, NOT repo): `probe_recon_magnitude.py`,
  `probe_where_recovery.py`, `probe_decode_granularity.py`
- Read first: `bin/Models.py` `_d3_reconstruction_loss` (~9261-9360),
  the loss assembly where `reconstruction_scale`/`what_scale`/`where_scale`
  /`when_scale` multiply in (grep those names in bin/Models.py +
  bin/Spaces.py), Architecture.md sec B, and the `.where` endpoint-sum
  bracket encoding (`WhereEncoding.decode_span`, doc/Spaces.md).

All probes drive `m.runEpoch` (house pattern) on `data/MM_20M_xor.xml`,
seed 0, and print structured numbers.

- [ ] **4.1 Magnitude probe**: per-epoch, print output_loss,
  recon_loss RAW (pre-scale) and SCALED (post `reconstruction_scale` and
  per-channel scales), and the ratio scaled_recon/output. Also print the
  VARIANCE of the recon loss across batches — a near-constant recon loss
  indicates dead signal, not mis-scaling. Verdict to report: `dead-signal`
  vs `mis-scaled` vs `healthy-but-small`.
- [ ] **4.2 `.where` probe**: for one settled batch, decode the
  reconstructed `.where` brackets (`WhereEncoding.decode_span`) for every
  part/whole percept and compare with the TRUE tiling spans of the input
  (serial tiling = running sum of part sizes). Print: fraction of exact
  span matches, fraction within +-1 position (the aliasing signature), and
  the distribution of errors. Verdict: `aliasing` (off-by-small) vs
  `collapsed` (all-same/zero) vs `unlearned` (random).
- [ ] **4.3 Granularity probe**: decode `last_reconstruction` at THREE
  granularities — slab-level (`reconstruct_data` default), word-level
  (nearest PS codebook row per word slot, the `_d3` metric machinery), and
  byte-level if the config exposes it — print all three next to the target
  text. Verdict: which granularity is closest to round-trip today, and
  where the mismatch enters (encoder slab boundaries vs decoder row
  choice).
- [ ] **4.4 Findings**: write the three verdicts + evidence into this
  plan's EXECUTION NOTES. **CHECKPOINT — STOP and present the verdicts +
  proposed fix selection (from the Task 5 decision tree) to Alec before
  implementing Task 5.**

---

## Task 5 — Fixes, TDD (decision tree; GATE 2 after this)

**Files:**
- Create: `test/test_reconstruction_roundtrip.py`
- Modify: per branch below (bin/Models.py loss assembly, bin/Spaces.py
  WS `.where` production / decode path)

The round-trip pin that defines DONE for this task (write FIRST, red):

```python
def test_mm20m_xor_exact_roundtrip():
    """THE bar (Alec 2026-07-03): decoded reconstruction == input, exactly."""
    rec = run_config("data/MM_20M_xor.xml", epochs=EPOCHS_PINNED, seed=0,
                     out_dir=SCRATCH)
    assert rec.exact_match_rate == 1.0
```

`EPOCHS_PINNED` is chosen from the Gate-1 baseline (smallest count that
the fixed pipeline converges in, plus 25% margin); the test is marked
`@pytest.mark.slow` / RUN_SLOW-gated if it exceeds ~30 s, with a FAST
sibling that pins the post-fix single-epoch trajectory (exact_match_rate
strictly increasing epoch-over-epoch on 3 epochs).

Branch on the Task-4 verdicts (implement ONLY the indicated branches):

- [ ] **5a. If `mis-scaled`**: rebalance — make the recon channel scales
  config-visible and set defaults so scaled recon sits within a decade of
  the output loss at init (e.g. `reconstructionScale` default 0.5 -> the
  probe-derived value; XSD comment updated, no `--`). Pin: raw/scaled
  ratio test asserting the intended proportion on the fixture.
- [ ] **5b. If `dead-signal`**: trace WHERE the gradient dies (detached
  tensor, EMA-only path, or clone-boundary) — fix the connection, pin with
  a grads-flow test (`loss.backward(); assert param.grad.abs().sum() > 0`
  for the reconstruction-path parameters).
- [ ] **5c. If `.where` `aliasing`/`collapsed`**: implement the sec-B
  type-tiling reading — WS supplies the tiling TYPE sequence
  (word/space/punct); absolute placement is the RUNNING SUM of part sizes
  computed at decode; `.where` loss compares type sequence + part sizes,
  not absolute coordinates. Pin: span-recovery test (all spans exact on
  the fixture after training).
- [ ] **5d. If granularity mismatch**: decode at word granularity via PS
  codebook rows in `reconstruct_data(text=True)`'s path (promote the
  `_d3` nearest-row machinery from metric-only to the decode); pin with a
  granularity test (decoded token count == input word count).
- [ ] **5.5 Green the round-trip pin**; update `where_recovery` in
  `bin/recon_bench.py` to the real (post-fix) span-match metric and drop
  the `-1.0` placeholder path.
- [ ] **5.6 Regression sweep**:
  `PYTHONPATH=test .venv/bin/python -m pytest test/test_reconstruction_roundtrip.py test/test_recon_bench.py test/test_mereology_word_binding.py test/test_conceptualize.py -q`
  then targeted files any modified loss code touches (grep callers).

**GATE 2 (Alec commits):** XOR exact round-trip green + probes' findings
in EXECUTION NOTES.

---

## Task 6 — MM_20M_grammar -> meronomy/meronomy

**Files:**
- Modify: `data/MM_20M_grammar.xml`
- Read first: `data/MM_20M_xor.xml` (the meronomy shape that works:
  PS nVectors 65536 / nDim 1024 / nOutput 8; WS 1024/1024/8 butterfly
  true), current grammar dims (PS 8/12/1024; WS 1028/8/8 butterfly false,
  serial, symbolicOrder 1, grammar XML), and the grammar-XML coupling
  (`useGrammar`, `<learning>`, `exploreTemperature`).

- [ ] **6.1 Reshape**: set `synthesis=meronomy`, `analysis=meronomy`; adopt
  the XOR-proven PS/WS dims adapted to serial mode — KEEP
  `butterfly=false` (the ~1M-param square fold; the 114s-build note in the
  config header is why) and KEEP serial+grammar+symbolicOrder=1; document
  every dimension decision in the config comments (no `--`).
- [ ] **6.2 Build + smoke**: model builds, one epoch runs, recon lines
  appear: `make bench_local MODEL=data/MM_20M_grammar.xml EPOCHS=1`.
- [ ] **6.3 Round-trip pin**: add `test_mm20m_grammar_exact_roundtrip`
  (same shape as the xor pin, RUN_SLOW-gated as needed) — drive to green;
  if grammar-specific blockers surface (serial-mode decode path
  differences), report the specifics and iterate — do NOT weaken the bar.

---

## Task 7 — Canonical config matrix (GATE 3 after this)

**Files:**
- Create: `test/test_config_matrix.py`
- Create (variants as small config files): `data/matrix/` —
  `MM_20M_xor_noraise.xml`, `MM_20M_grammar_stack.xml`,
  `MM_20M_grammar_reading.xml`, `MM_20M_grammar_twopass.xml`
  (each a copy of its parent with ONE flag flipped; header comment names
  the flip; generate by copy at authoring time, not runtime, so `make
  test` shows real files).

- [ ] **7.1 The matrix test** (fast smokes; parametrized):

```python
MATRIX = [
    "data/MM_20M_grammar.xml",         # predominant path (meronomy pair)
    "data/MM_20M_xor.xml",             # parallel mereology, sO=0
    "data/MM_20M_legacy.xml",          # bpe/byte back-compat
    "data/matrix/MM_20M_xor_noraise.xml",      # mereologyRaise off
    "data/matrix/MM_20M_grammar_stack.xml",    # subsymbolicStack on
    "data/matrix/MM_20M_grammar_reading.xml",  # readingAttention on
    "data/matrix/MM_20M_grammar_twopass.xml",  # learning+exploreTemperature
    "data/MM_sparse_concept.xml",      # sO=3 parallel (wave; smoke ONLY)
]

@pytest.mark.parametrize("cfg", MATRIX)
def test_config_builds_runs_and_reconstructs(cfg):
    """One capped epoch: builds, runs, finite losses, recon decodes."""
    rec = run_config(cfg, epochs=1, seed=0, out_dir=SCRATCH,
                     max_batches=SMOKE_BATCHES)
    assert math.isfinite(rec.output_loss) and math.isfinite(rec.recon_loss)
    assert rec.exact_match_rate >= 0.0     # decode path executed
```

(`max_batches` plumbs to `runEpoch(max_batches=...)` — add the passthrough
to `run_config` in this task if Task 1 didn't. `SMOKE_BATCHES` sized so
the whole matrix stays under ~60 s in `make test`; measure and record.)

- [ ] **7.2 RUN_SLOW tier**: full round-trip assertions for xor + grammar
  (reuse the Task 5/6 pins — don't duplicate; this step just confirms the
  gating so `make test` stays fast and `make test_all` runs the bar).
- [ ] **7.3 `make test` on a quiet tree** — green is the gate criterion.

**GATE 3 (Alec commits):** matrix + round-trips green; `make test` green.

---

## Task 8 — Speed pass on ArborStudio (GATE 4 after this)

**Files:**
- Modify: as the profile dictates (each optimization is its own mini
  red-green: benchmark number before/after + round-trip suite green)
- Read first: Gate-1/3 accumulated run records in `output/`.

- [ ] **8.1 Profile**: `make bench_remote MODEL=data/MM_20M_grammar.xml
  EPOCHS=3` with `--profile`; pull the top-25 ops table. Also capture
  `MODEL_COMPILE=auto` vs `eager` wall-clock there (native path — expected
  unblocked; if CppCompileError persists even without the path space,
  record it and stay eager).
- [ ] **8.2 Optimization ladder** (apply in order, STOP when production
  throughput is acceptable to Alec; each rung: measure -> change -> re-run
  `test/test_reconstruction_roundtrip.py` + the matrix fast tier ->
  record before/after in the run records):
  1. compile mode (`MODEL_COMPILE=auto` / `reduce-overhead`) on ArborStudio;
  2. prefetch (`<numWorkers>` > 0 — the `_TickPrefetcher` path);
  3. batch-size sweep (the harness makes this a loop over run records);
  4. profiler-directed hot-spot fixes (unknowable until 8.1 — each gets
     its own before/after record and a one-line note in EXECUTION NOTES;
     if a fix requires touching numerics, it ALSO needs the round-trip
     suite green to land);
  5. MPS-eager probe (compile-incompatible; only adopt if it beats the
     best CPU number at equal fidelity).
- [ ] **8.3 The speed report**: before/after epochs-per-hour for
  MM_20M_grammar on ArborStudio + the ladder log, appended to EXECUTION
  NOTES.

**GATE 4 (Alec commits; plan done):** speed report recorded; `make test`
green (final full run, quiet tree).

---

## EXECUTION NOTES (append during execution, house style)

Baseline table (Task 3), probe verdicts (Task 4), fix branches taken
(Task 5), grammar reshape decisions (Task 6), matrix timing (Task 7),
speed ladder (Task 8), deviations as numbered items.

---

## Self-review (writer, 2026-07-03)

- Spec coverage: design decisions 1-5 map to Tasks 5/6 (bar), 7 (matrix),
  8 (throughput), 6 (defaults), 1-3 (conjunction harness); gates match the
  spec's four gates. Parked items (wave, Task 11) appear in no task —
  correct.
- No placeholders: probe steps carry full instrumentation specs; the one
  evidence-contingent region (Task 5) is a decision tree with concrete
  implementations per branch and an explicit Alec CHECKPOINT at 4.4;
  `where_recovery`'s pre-fix `-1.0` sentinel is declared and retired at 5.5.
- Type consistency: `run_config(config, epochs, seed, out_dir, profile,
  max_batches)` and `RunRecord`/`exact_match_rate` are used identically in
  Tasks 1, 5, 7; Make targets named `bench_local/bench_sync/bench_remote/
  bench_pull` throughout.
