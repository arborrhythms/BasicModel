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

- [x] **1.1 Write the failing test** (`test/test_recon_bench.py`):

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

- [x] **1.2 Run to verify failure:**
  `PYTHONPATH=test:bin .venv/bin/python -m pytest test/test_recon_bench.py -x -q`
  — expect ImportError.

- [x] **1.3 Implement `bin/recon_bench.py`:**

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

- [x] **1.4 Run tests to verify pass** (same command as 1.2). Also run the
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

- [x] **2.1 Add Make targets** (basicmodel/Makefile, following its existing
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

- [x] **2.2 Smoke the remote path** (requires the host up; if
  `ArborStudio.local` is unreachable, report it and mark this step
  deferred-not-failed): `make bench_remote MODEL=data/MM_xor_fixture.xml
  EPOCHS=1` then `make bench_pull` — a `recon_*.json` with
  `"host": "ArborStudio..."` lands in `output/`. Record whether
  `MODEL_COMPILE=auto` works there (no path-space CppCompileError on the
  native path) — just note it; the speed pass exploits it in Task 8.

---

## Task 3 — Baselines (GATE 1 after this)

- [x] **3.1 Local baselines**: `make bench_local` for
  `data/MM_20M_xor.xml`, `data/MM_20M_grammar.xml` (pre-switch),
  `data/MM_20M_legacy.xml` — pick `EPOCHS` so each run stays under ~15 min
  (start with 3; use `BASIC_MAX_BATCHES` if a single epoch is too long,
  and record the cap in the JSON's config field notes).
- [x] **3.2 Remote baselines**: same three via `make bench_remote` (skip
  gracefully if host unreachable; Gate 1 then carries local-only baselines
  with a note).
- [x] **3.3 Record** a baseline table (config x host: s/epoch, recon_loss,
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
- [x] **5b. If `dead-signal`**: trace WHERE the gradient dies (detached
  tensor, EMA-only path, or clone-boundary) — fix the connection, pin with
  a grads-flow test (`loss.backward(); assert param.grad.abs().sum() > 0`
  for the reconstruction-path parameters).
- [x] **5c. If `.where` `aliasing`/`collapsed`**: implement the sec-B
  type-tiling reading — WS supplies the tiling TYPE sequence
  (word/space/punct); absolute placement is the RUNNING SUM of part sizes
  computed at decode; `.where` loss compares type sequence + part sizes,
  not absolute coordinates. Pin: span-recovery test (all spans exact on
  the fixture after training).
- [x] **5d. If granularity mismatch**: decode at word granularity via PS
  codebook rows in `reconstruct_data(text=True)`'s path (promote the
  `_d3` nearest-row machinery from metric-only to the decode); pin with a
  granularity test (decoded token count == input word count).
- [x] **5.5 Green the round-trip pin**; update `where_recovery` in
  `bin/recon_bench.py` to the real (post-fix) span-match metric and drop
  the `-1.0` placeholder path.
- [x] **5.6 Regression sweep**:
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

- [x] **8.1 Profile**: `make bench_remote MODEL=data/MM_20M_grammar.xml
  EPOCHS=3` with `--profile`; pull the top-25 ops table. Also capture
  `MODEL_COMPILE=auto` vs `eager` wall-clock there (native path — expected
  unblocked; if CppCompileError persists even without the path space,
  record it and stay eager). [DONE: top-3 per config recorded; NB the
  harness `MODEL_COMPILE`-inert finding — `--compiled-step` flag added.]
- [x] **8.2 Optimization ladder** (apply in order, STOP when production
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
  [DONE: compile modes measured (grammar: no win, PYTHON-LENGTH recompile
  census names `_per_word_body_step` `out_slot` churn hitting cache limit;
  xor: +12% ro but numerics drift); prefetch + batch = dev-scale no-ops;
  MPS both configs BLOCKED (grammar generator-device crash, xor MPS OOM/NaN)
  — all RECOMMENDATIONS, no shipped-config/numerics edits, gate stays 24.]
- [x] **8.3 The speed report**: before/after epochs-per-hour for
  MM_20M_grammar on ArborStudio + the ladder log, appended to EXECUTION
  NOTES. [DONE: grammar 1.124→1.482 s/epoch = 2430 eph (arch switch, not
  regression); xor 4524→5085 eph best; ladder table + CUDA outlook recorded.]

**GATE 4 (Alec commits; plan done):** speed report recorded; `make test`
green (final full run, quiet tree).

---

## EXECUTION NOTES (append during execution, house style)

Baseline table (Task 3), probe verdicts (Task 4), fix branches taken
(Task 5), grammar reshape decisions (Task 6), matrix timing (Task 7),
speed ladder (Task 8), deviations as numbered items.

### Task 3 baselines (2026-07-03, local: ArborBook, cpu/eager, seed 0)

All three configs use the small `xor` dataset (4 rows) despite the
`MM_20M_*` filename convention — probes (`--epochs 1 --max-batches 2`) came
back in 1-2s/epoch, so the full baseline used **epochs=3, uncapped
batches** for every config; no `--max-batches` cap was needed anywhere.

| config | epochs x batches | wall s/epoch (steady) | output_loss | recon_loss | exact_match | decoded rows | notes |
|---|---|---|---|---|---|---|---|
| `data/MM_20M_xor.xml` | 3 x uncapped (1 batch/epoch) | 1.449 | 0.178018 | 0.0 | 0.0 | 4 | `recon_loss` exactly 0.0 — the lossIn-never-populates finding (recon channel is dead on this config) |
| `data/MM_20M_grammar.xml` (pre-meronomy-switch) | 3 x uncapped (1 batch/epoch) | 1.363 | 0.0 | 0.268371 | 0.0 | 4 | `output_loss` exactly 0.0 — inverse of the xor symptom (output channel is dead here); recon channel IS live |
| `data/MM_20M_legacy.xml` | 3 x uncapped (1 batch/epoch) | 0.197 | 0.125736 | 0.004399 | 0.0 | 4 | both channels live and small-but-nonzero; fastest config (~7x xor/grammar) — smaller embedding-dim build path |

JSON files (repo `output/`):
- `output/recon_MM_20M_xor_20260703-162937_32827.json` (baseline; probe:
  `output/recon_MM_20M_xor_20260703-162902_32681.json`)
- `output/recon_MM_20M_grammar_20260703-162953_32891.json` (baseline;
  probe: `output/recon_MM_20M_grammar_20260703-162916_32736.json`)
- `output/recon_MM_20M_legacy_20260703-163006_33070.json` (baseline;
  probe: `output/recon_MM_20M_legacy_20260703-162926_32776.json`)

Note: `output/recon_MM_20M_legacy_20260703-161203_27151.json` is a
pre-existing artifact from earlier ad hoc exploration (epochs=1, not part
of this Task 3 baseline set) — left in place, not counted in the table
above; its loss values (output_loss 0.174975, recon_loss 0.004521) are
close to but not identical to the epochs=3 baseline above, consistent with
one warm-up-only epoch vs three.

`exact_match_rate` is 0.0 across all three configs — this is EXPECTED
pre-fix per the plan; not a new finding.

Remote (ArborStudio) baselines: see "Task 3 remote baselines (ArborStudio
via tunnel)" below — access restored via the wikiOracle.org reverse tunnel
(`ARBOR_TUNNEL=1`); Gate 1 now carries both local and remote baselines.

**Observations for Task 4:**
1. The xor config (`MM_20M_xor.xml`) and the grammar config
   (`MM_20M_grammar.xml`) show COMPLEMENTARY dead channels: xor has
   `recon_loss == 0.0` (recon channel dead) while grammar has
   `output_loss == 0.0` (output channel dead). Both are candidates for the
   `dead-signal` verdict in Task 4.1's magnitude probe — worth checking
   whether this is the SAME root cause manifesting on two different wiring
   paths (xor = parallel mereology sO=0; grammar = serial + symbolicOrder=1
   pre-meronomy-switch) or two distinct bugs.
2. `legacy` is the only config where both output_loss and recon_loss are
   simultaneously nonzero and finite, and by far the fastest to build/run —
   useful as the healthy-baseline reference point for Task 4's magnitude
   probe (`healthy-but-small` calibration).
3. `decoded_rows` is 4 for every config — matches the full xor dataset row
   count (no truncation to `_MAX_EVAL_BATCH`=512), so the decode path
   executed cleanly (no `decode_note` degradation) on all three configs at
   epochs=3, seed=0.
4. Per-epoch timings show a clear warm-up-vs-steady-state split
   (~1.6-1.9s first epoch vs ~1.3-1.4s steady for xor/grammar; ~0.64s vs
   ~0.20s for legacy) — `wall_s_per_epoch` in the table above already
   excludes epoch 1 per the harness's own steady-state averaging.

### Task 3 remote baselines (ArborStudio via tunnel)

Run 2026-07-03 with `make bench_remote ARBOR_TUNNEL=1 MODEL=<cfg> EPOCHS=3`
(ArborStudio = Mac Studio M4 Max, macOS 26.5.1, reached off-LAN through the
wikiOracle.org reverse tunnel; cpu-pinned per the target, native no-space
path so `compile_mode` is the default `auto` — inductor compiles cleanly,
no CppCompileError; seed 0).

| config | epochs x batches | wall s/epoch (steady) | output_loss | recon_loss | exact_match | decoded rows | notes |
|---|---|---|---|---|---|---|---|
| `data/MM_20M_xor.xml` | 3 x uncapped (1 batch/epoch) | 0.797 | 0.178018 | 0.0 | 0.0 | 4 | `recon_loss` exactly 0.0 — same dead recon channel as local |
| `data/MM_20M_grammar.xml` (pre-meronomy-switch) | 3 x uncapped (1 batch/epoch) | 1.124 | 0.0 | 0.268371 | 0.0 | 4 | `output_loss` exactly 0.0 — same dead output channel as local |
| `data/MM_20M_legacy.xml` | 3 x uncapped (1 batch/epoch) | 0.157 | 0.125736 | 0.004399 | 0.0 | 4 | both channels live; fastest config, mirroring local |

JSON files (repo `output/`, pulled via `make bench_pull ARBOR_TUNNEL=1`):
- `output/recon_MM_20M_xor_20260703-092703_50666.json`
- `output/recon_MM_20M_grammar_20260703-092725_50760.json`
- `output/recon_MM_20M_legacy_20260703-092745_50846.json`

Losses match the local baselines to ~6 decimals on every config (seed-0
cross-machine reproducibility holds), and s/epoch is uniformly faster on
ArborStudio — 1.8x on xor (0.797 vs 1.449), 1.2x on grammar (1.124 vs
1.363), 1.25x on legacy (0.157 vs 0.197) — with `auto` compile engaged
remotely vs pinned eager locally. Compile experiment (Task-8 early signal,
`MM_xor_fixture` EPOCHS=1): eager 0.553 s/epoch vs auto 0.393 vs forced
inductor 0.315, identical losses — inductor works and already wins on the
tiny fixture.

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

### Task 4 probe verdicts + checkpoint decisions (Alec, 2026-07-03)

TWO distinct dead-channel bugs: (a) xor recon dead -- `create_ir_mask` assumes
Embedding `.what`, early-returns on meronomy's Codebook (bin/Models.py:2774)
$\to$ explicit-zeros branch; (b) grammar output dead -- equal-dim/unequal-shape
silent zero (bin/Models.py:4111) atop byte-cursor stub labels (real labels never
reach the head; legacy's "live" output regresses stub zeros too). `.where` is
NEVER WRITTEN on the radix/meronomy path (encoder stamps no whereEncoding;
reverse yields ~1e-3 noise past a 1e-8 floor $\to$ garbage offsets). Granularity:
active-slot nearest-row is 2/4 exact (editdist 6 vs 92); pad slots decode as
content. Whole-sentence percepts (basicLevelMaxSize=24 promotion) mean no word
tiling exists. Grammar double-counts one reconstruction (D3 == reverse value).

DECISIONS: branches 5b+5c+5d approved; 5a DEFERRED until channels live (zeros
can't be rescaled; legacy ~2.5-decade imbalance revisited post-5b). Round-trip
bar = WORDS MUST TILE -- top-down word-level analysis (whitespace/punct cut)
must bound percept growth; investigate wiring vs tuning of the basic-level stop
(new 5e). Grammar label supervision DEFERRED to Task 6 (byte-cursor path
retires with the meronomy switch); only the silent shape-gate gets fixed now.
Dedupe reconstruction/reconstruction_reverse NOW -- unify into one
reverse-implemented objective (masking as a mode), verify reading in code.
Fail-loud/warn-once at BOTH silent-zero sites regardless of branch.

### Task 5b execution (2026-07-03, local cpu/eager, seed 0)

**Dedupe reading (verified in code + probe).** The two terms are ONE
objective only on the serial/D3 path: `_d3_reconstruction_loss`
(reverse(S) vs `_ar_embedded`) and the `lossRev` block (reverse(STM
snapshot slot 0) vs forwardInput) return bit-identical values on grammar
(0.2683708071708679 in both slots, all epochs) -- the same
reverse-implemented comparison entered the training total twice, at
$rr \cdot \mathrm{lossIn} + rr \cdot \mathrm{lossRev}$. On the PARALLEL
path they are genuinely independent objectives (legacy probe: masked-LM
0.0044 vs reverse round-trip 0.1195). Dedupe therefore fires only where
they coincide: TRAIN batches with `_d3_active` skip the second reverse
pass and the `reconstruction_reverse` accumulation entirely; EVAL keeps
the reverse pass (it stages `inputPred` and the decode-path state).

**Fixes (bin/Models.py only):**

1. `create_ir_mask` now supports Codebook `.what` (radix/meronomy PS).
   Gate changed from "has `null_percept_idx`" to "has a 2-D `getW()` row
   table"; the injected NULL for a Codebook is the documented all-zeros
   MASK vector (that path has no NULL row -- Spaces.py MPHF-table note);
   pad exclusion falls back to content-mass when `_index` is absent.
   Embedding paths are byte-identical (legacy losses reproduce to 6
   decimals). Route chosen over promoting the reverse round-trip into the
   `reconstruction` slot because (a) it engages the EXISTING masked
   objective with the minimal diff, (b) xor's reverse round-trip is
   $\sim 10^{-4}$ (weak) vs $\sim 2\times 10^{-3}$ masked, and (c) the
   reverse-into-lossIn route would have recreated the double-count on the
   parallel path.
2. Output shape gate: new `_align_output_pred` mean-reduces
   label-singleton axes after the trailing-dim loop (head [B,4,4] vs
   labels [B,1,1] reconciles explicitly); irreconcilable shapes emit a
   warn-once RuntimeWarning and zero the term.
3. Warn-once `_warn_zeroed_channel(site, detail)` (per site+config,
   mirrors `_csw_overflow_seen`): fires at the zeroed-reconstruction
   else-branch in `runBatch` (eager, outside the compiled forward -- the
   compiled `create_ir_mask` stays warning-free), at the shape gate, and
   at the output-loss except-branch.
4. `_infer_ir`: Codebook `.what` returns `[]` before the Embedding-only
   `wv.most_similar` readout -- the newly-staging mask had un-gated that
   path (`generate_sentence` smoke on radix configs raised
   AttributeError); pre-fix behaviour restored.

**Post-fix table (harness `bin/recon_bench.py`, epochs 3, seed 0):**

| config | output_loss | recon_loss | vs baseline |
|---|---|---|---|
| `MM_20M_xor` | 0.186752 | 0.001754 | recon was 0.0 DEAD $\to$ LIVE (compute_masked branch all batches; grads reach 16 params incl. `perceptualSpace.subspace.what.W`); output trajectory shifted (0.178 $\to$ 0.187) by the now-live recon gradient |
| `MM_20M_grammar` | 0.168615 | 0.268405 | output was 0.0 DEAD $\to$ LIVE-but-stub-supervised (labels are the Task-6 byte-cursor stubs; decreasing 0.175 $\to$ 0.169 over 3 epochs); recon now a SINGLE term (was double-counted) |
| `MM_20M_legacy` | 0.125736 | 0.004399 | byte-identical to baseline (per-epoch losses reproduce exactly) |

xor recon is not strictly monotonic across 3 epochs (0.00209 / 0.00166 /
0.00175 -- the bernoulli mask is stochastic); epoch 2 < epoch 0.

**Tests:** `test/test_reconstruction_roundtrip.py` (6 tests: live recon,
grads-flow, output-not-silent-zero, shape-gate reconcile, warn-once at
both former silent-zero sites, no-double-count). Regression green:
`test_recon_bench.py` + `test_basicmodel.py` (210 passed),
`test_mereology_word_binding.py` / `test_conceptualize.py` /
`test_ir_mode.py` / `test_ir_only_refactor.py` /
`test_radix_recon_render.py` / `test_tied_vectors_compile.py` /
`test_mlx_export.py` (41 passed, matches HEAD), plus
`test_conceptual_recurrence.py` / `test_dimensional_governance.py` /
`test_ltm_consolidation.py` / `test_brick_no_sync.py` /
`test_cuda_graph_capture.py` / `test_compiled_step_invoked.py` /
`test_mm_xor.py` / `test_mm_boolean.py` / `test_stream_smoke.py`
(59 passed).

**Deviations / findings (numbered):**

1. Running `pytest test/test_recon_bench.py test/test_basicmodel.py` as a
   bare pair aborts on MPS (M2 GPU OOM + radix KeyError) on HEAD too --
   PRE-EXISTING: `test_basicmodel.py` hard-sets `BASICMODEL_DEVICE=gpu`
   at module (collection) time, and `recon_bench` imports `util` lazily,
   so the pair runs GPU. The full suite is protected by accident (an
   earlier file freezes cpu first). Workaround used for the gate:
   `python -c "import util; pytest.main([...])"` (freezes cpu
   pre-collection, the full-suite regime) $\to$ 210 passed.
2. `test_compiled_step_invoked.py::test_compiled_step_is_invoked` XPASSES
   on HEAD as well (stale xfail, likely fixed by torch 2.12) -- not a 5b
   effect.
3. The dedupe is train-only by design: eval keeps the second reverse pass
   because `inputPred` staging and the decode path depend on its side
   effects; the double-count only ever entered the TRAINING total.

### Device policy (Alec, 2026-07-03)

GPU-FIRST everywhere by default: long-run training on a borrowed CUDA server
is imminent (parent Makefile Lambda/EC2 H100 configs). CPU pins are the
exception and need a stated reason. Standing exceptions: seeded fidelity
comparisons on local MPS (documented nondeterministic) stay cpu-pinned until
CUDA; the local M2 MPS OOM on heavy pairings. Consequences for this plan:
Task 8's ladder promotes GPU rungs to primary (MPS-eager on the Macs now,
CUDA+compile on the borrowed server as the real target; ArborStudio's
native-path inductor finding de-risks the compile half); bench targets grow a
device knob (BASICMODEL_DEVICE passthrough already exists on bench_remote);
epochs/hour ON GPU is the headline throughput number.

### Task 5e execution (word-level tiling; 2026-07-03, local cpu/eager, seed 0)

**Root cause: WIRING, not tuning.** The word-isolation cut existed but no
promotion/spell-out consumer was wired to it. The chain: IS hands PS ONE
whole-line token per row (bin/Spaces.py `host_tokens = [[s] for s in
surfaces]`, the "IS always emits RAW" block in ``forward``); PS
``_embed_radix`` Pass 1 called ``ps.observe_chunk`` per LEXER TOKEN, so the
whole line accrued sightings; ``RadixLayer.observe_chunk`` (bin/Layers.py)
promotes purely on hit count $\ge$ threshold (xor: 2) and length $\ge$
min_length (2) -- no upper bound, no boundary check -- so `'hello world'`
promoted to ONE percept and ``spell_out``'s longest-match then returned it
whole. The Pass-2 ``Meronomy.word_spans`` cut (under ``_meronomy_words``)
only TAGGED emitted pids with word-group indices; it never bounded anything.
``<basicLevelMinSize>``/``<basicLevelMaxSize>`` are defined in
data/model.xsd and set in the configs but read NOWHERE in bin/ (dead knobs;
the size bound also could not express "don't cross whitespace" -- the
doc/Mereology.md 2026-06-29 note already called the word cut the operative
brake). The WS$\to$PS ``_passback_scope_where`` handoff
(``WholeSpace.passback_action`` / ``BasicModel._passback_scope_ps``) only
scopes $t>0$ subsymbolic passes by zeroing percept-event slots -- it never
reaches the store, and on xor it is dark anyway (no readingAttention, no
words-category intent). The whole-line 1-chunk regime dates to the
2026-06-09 "No whitespace pre-parse" commit (970a578, an SBOW anti-collapse
justification); the later word-isolation cut landed as tagging only.

**Fix (structural, per the checkpoint decision):**
1. ``Meronomy.word_tiling`` (new, bin/Meronomy.py): the COMPLETE
   word/space/punct tiling -- ``word_spans`` word spans interleaved with
   separator runs (maximal same-class runs), covering every byte once
   (`'hello world'` $\to$ `['hello', ' ', 'world']`).
2. ``PartSpace._embed_radix`` (bin/Spaces.py), gated ``_meronomy_words``:
   the cut is computed ONCE per row AHEAD of Pass 1; Pass 1 observes per
   TILING SEGMENT (parts top out at word size); Pass 2 cuts each token at
   tile boundaries and spells out per piece, so ``longest_match`` cannot
   cross a boundary even if a crossing percept exists in the trie; the
   per-pid ``.where`` offset is the piece (word-tile) start (was the
   whole-line token start, i.e. all zeros here). Off-flag byte-identical.
3. ``RadixLayer`` (bin/Layers.py): new ``word_bounded`` ctor flag (PS passes
   ``_meronomy_words``; the bpe/mphf mirror store stays unbounded) and a
   ``_promotable`` gate -- a promotable chunk is ONE word-tiling tile --
   enforced at BOTH promotion sites (``observe_chunk`` and
   ``lookup_with_id`` step 4), so promotion cannot merge across a word
   boundary even when size allows.

**Evidence (xor, seed 0, 3 epochs + eval):** promoted store entries are
exactly ``[hello, loving, there, world]`` (no whole-sentence percepts);
every row's active-percept sequence EQUALS its word tiling
(``[hello, ' ', world]`` etc.); word groups tag word/separator/word as
0/-1/1. Granularity probe (10 epochs): pre-fix per-slot words were
``['hello world', 'e', ...]`` (one sentence-percept per row; B word-active
3/4 "exact" only because the single percept WAS the sentence); post-fix
slots hold word percepts (slot 0 correct 4/4) with the residual mismatch
now slot alignment ($5c$ ``.where``) and pad-slot decode pollution ($5d$),
not tiling.

**Harness (epochs 3, seed 0) vs the 5b table:** xor 0.186752/0.001754
$\to$ 0.186193/0.001716 (output/recon; expected shift -- the representation
changed from 1 sentence-percept to a 3-percept word tiling);
grammar 0.168615/0.268405 and legacy 0.125736/0.004399 BYTE-IDENTICAL
(non-meronomy paths untouched).

**Tests:** two new pins in test/test_reconstruction_roundtrip.py
(``test_xor_percepts_tile_words``, ``test_word_tiling_survives_promotion``);
all six 5b pins pass unchanged (they were $>0$ liveness pins -- no meaning
change). Regression sweep green: test_recon_bench,
test_mereology_word_binding, test_conceptualize, test_meronomy,
test_meronomy_laws (55), test_mm_xor, test_radix_recon_render,
test_ir_mode (27), test_mereology_raise, test_where_run_structure (34).

**Deviations / findings (numbered):**
1. test_where_attention_handoff.py::test_mereology_raise_stamps_stage0_ws_
   and_is_byte_identical fails PRE-EXISTING: a file-shadowing bisect shows
   pristine-HEAD determinism holds, and restoring only the 5b bin/Models.py
   reproduces the failure (stateful, not RNG -- reseeding does not restore
   equality). 5b regression on the radix eval forward; flagged as a
   separate task, NOT a 5e effect (5e files shadowed to pre-state still
   fail; my diff is inert on synthesis=radix configs).
2. ``basicLevelMinSize``/``basicLevelMaxSize`` remain unimplemented (dead
   config knobs) -- the structural word bound supersedes them for TEXT; the
   principled ratio-driven basic-level convergence stays the
   doc/Mereology.md REVISIT item.
3. The per-pid ``.where`` offsets under meronomy are now word-tile starts
   (previously all 0 -- the whole-line token start). This feeds the $5c$
   ``.where`` work correct data; the whereEncoding stamp/decode itself is
   untouched (5c's scope).

### Design note: types from properties; word-sized .where contract (Alec, 2026-07-03)

`basicLevelMin/MaxSize` are NOT needed parameters — strip them. The analysis
methods we enable are a set of PROPERTIES identifying the TYPES we attend to,
by dividing the input (left of space, right of space, punctuation, numbers).
That division yields word-sized objects; each object's `.where` then either
(a) ASSOCIATES with a maximal part from PS (a promoted percept covering
exactly that span), or (b) is SENT BACK to PS to produce a parts-based
definition covering exactly that `.where` (spell-out bounded by the span).
5e implements (b)'s bound structurally; 5c's `.where` channel carries these
word-sized spans and reconstruction recovers placement as their running sum.

### Task 5c/5d/5.5 execution (2026-07-03, local cpu/eager, seed 0)

**THE BAR IS GREEN**: `test_mm20m_xor_exact_roundtrip` passes with
`exact_match_rate == 1.0` AND `where_recovery == 1.0` at
`EPOCHS_PINNED = 25` (RUN_SLOW-gated, $\sim$70 s), plus a fast sibling at
the harness default budget (epochs 3) pinning the same two 1.0s.

**Design as built (adapts the 5c/5d branches to two measured facts):**

1. **Encode stamp** (bin/Spaces.py `_embed_radix`, gated
   `_meronomy_words`): each content slot's word-tile START is stamped as a
   `.where` INSTANT ($|.| = 1$) into the muxed where-band (the band was
   zero-padded, never written); pads ($-1$) carry no claim. The stamp only
   fires when the band lies past the percept content columns
   (`W.shape[-1] <= idx[0]`), so a content-width codebook can never be
   overwritten. Input-determined — the determinism bar
   (test_where_attention_handoff.py, 7 passed) stays green.
2. **Noise floor** (`_decode_where_offset`): the $10^{-8}$ guard became a
   magnitude floor of $0.1$ ($\sin^2 + \cos^2 < 0.01 \to$ no offset
   claim). Measured: a real stamp is $1.0$; an unwritten band reconstructs
   $\sim 10^{-2}$ noise (pre-fix that noise decoded to garbage offsets
   $\sim$16k-35k and stamped them into the render).
3. **Measured fact A — the band's ANGLE does not survive the reverse**:
   reconstructed claim positions err by $\sim$900-3600 bytes (band angle
   noise $\sim$0.04 rad at period `maxVal` 131337; scaling the period
   cannot reach byte precision, and bracket EXTENTS die at this period
   anyway — the `decode_span` instant-snap floor is $\sim$30 bytes). Only
   the claim MAGNITUDE (written vs unwritten) is a robust carrier. Hence
   the contract's reading: the absolute `.where` value CONFIRMS
   (`meta["where_abs"]`), never dictates.
4. **Measured fact B — unrestricted content decode diverges with
   training**: the reverse inflates slot norms $2$-$3.4\times$ and the
   L2 nearest-row drifts row-constant (epochs 40: every row decodes
   `'hellotherev'`-style; epochs 80: all rows identical). But the SAME
   content vectors ARE row-discriminative under the contract's arm (a):
   restricted to percepts covering EXACTLY the tile's span (size), cosine
   association is 12/12 at epochs 20/40/80 (seed 0).
5. **Decode** (`_decode_radix_meta` + new `RadixLayer.associate_span`,
   bin/Layers.py): the forward parks the 5e tiling record per slot
   (`_forward_input['tile_spans']` — the "WS supplies TYPE tiling"
   scaffold, same-batch, input-determined). The render set = the slots the
   forward actually populated (5d; pads never render; `.where` claims are
   the self-contained fallback gate when no scaffold rode the batch —
   hand-staged thunks keep the legacy decode-all path). Per rendered slot:
   arm (a) — `associate_span(content, size=span_size)`, cosine over the
   size bucket (scale-invariant against the norm inflation; the `_d3`
   nearest-row machinery promoted into the decode per the 5d branch, made
   scale-robust); arm (b) — empty bucket falls back to the unrestricted
   associate (parts). PLACEMENT IS THE RUNNING SUM of emitted part sizes
   (Architecture.md sec B); the decoded absolute claims land in
   `meta["where_abs"]` as the confirmation channel.
6. **recon_bench**: `where_recovery` is now REAL — the fraction of decoded
   percepts whose recovered span (start, len) exactly matches the true
   `Meronomy.word_tiling` span (ordinal-aligned; extra/missing count
   against); `-1.0` only when no decode is available (note says so). The
   `-1.0` placeholder path and its note are gone.

**Probe tables (probe_where_recovery / probe_decode_granularity, seed 0):**

| epochs 10 | pre-5c/5d | post |
|---|---|---|
| decoded rows' slot count | 8 (pads decode as content at noise offsets) | 3 (= forward-active) |
| rendered offsets | $\sim$16k-35k garbage | running sums = true tile starts (0/5/6, 0/6/7) |
| A slab exact / median editdist | 0/4 / 13 | 0/4 / 1 (epochs 10 = the dip; see trajectory) |
| B word median editdist | 23 | 2 |

At the pinned budget (epochs 25): A slab, B word, B word-active ALL
`exact=True` 4/4, per-slot words `['hello', ' ', 'world']` at offsets
`[0, 5, 6]` — the A/B rows collapsed to exact once pads were gated and the
span association landed. (C byte stays a raw ungated diagnostic.)

**Convergence trajectory (full pipeline `run_config` semantics, seed 0):**

| E | 1 | 2 | 3 | 10 | 15 | 20 | 25 | 30 | 40 | 80 |
|---|---|---|---|---|---|---|---|---|---|---|
| exact_match | 1.0 | 0.5 | 1.0 | 0.0 | 0.5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| where_recovery | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

Smallest STABLE convergence = 20 (the 1.0s at E=1/3 are early-geometry
blips — E=2/10/15 dip); $20 \times 1.25 = 25 =$ `EPOCHS_PINNED`, inside
the verified [20, 80] plateau. Seeds: seed 2 reaches 1.0 at E=40; seed 1
tops out at 0.5 by E=40 (content association `'e'`/`'world'` confusions)
— the bar pins seed 0 per the plan; the seed spread is a reported concern,
not a weakened assertion. `where_recovery` = 1.0 throughout (the tiling
channel is budget-independent once the store promotes, E $\ge$ 1).

**Harness (bin/recon_bench.py, epochs 3, seed 0) vs the 5e table:**

| config | output_loss | recon_loss | exact | where_recovery |
|---|---|---|---|---|
| `MM_20M_xor` | 0.186193 $\to$ 0.187600 | 0.001716 $\to$ 0.001812 | 0.0 $\to$ **1.0** | placeholder $\to$ **1.0** |
| `MM_20M_grammar` | 0.168615 (byte-identical) | 0.268405 (byte-identical) | 0.0 | 0.0 (real metric, honest) |
| `MM_20M_legacy` | 0.125736 (byte-identical) | 0.004399 (byte-identical) | 0.0 | 0.0 (real metric, honest) |

xor's loss shift is the expected stamp effect (the forward event now
carries the where instants; the recon objectives learn them too).

**Tests:** four new pins in test/test_reconstruction_roundtrip.py
(`test_mm20m_xor_exact_roundtrip` [RUN_SLOW], the epochs-3 fast sibling,
`test_associate_span_two_arms`, `test_radix_decode_gates_pad_slots`); all
ten prior pins unchanged $\to$ 14 passed (RUN_SLOW=1, 91 s). Sweep green:
test_where_attention_handoff (7), test_radix_recon_render +
test_recon_bench (7), test_mereology_word_binding + test_mm_xor +
test_mm_boolean (17 passed, 2 skipped), test_sparse_concept_e2e (16),
test_meronomy + test_meronomy_laws + test_conceptualize + test_ir_mode
(52), test_mereology_raise + test_where_run_structure (34),
test_xor_spaces + test_explicit_dimensions (23 passed, 2 skipped,
1 xfailed).

**Deviations / findings (numbered):**

1. The plan's fast-sibling shape ("exact_match strictly increasing over 3
   epochs") is UNSATISFIABLE: the trajectory STARTS at 1.0 (E=1) and is
   non-monotone (see table). Replaced with the epochs-3 harness-default
   pin (`exact == 1.0` and `where_recovery == 1.0`), which doubles as the
   task's where_recovery-on-trained-xor assertion.
2. Spans are stamped as INSTANTS at tile starts, not (start, end)
   brackets: at period 131337 the bracket extent is $\sim$4 decades below
   the `decode_span` instant-snap resolution floor, so an encoded extent
   is unrecoverable even before reverse noise. Tile SIZES therefore ride
   the same-batch analysis scaffold (`tile_spans`), which the 5d
   forward-populated-mask sanction already required for gating.
3. Non-monotone content association is a TRAINING-side residual, not a
   decode residual: the reverse round-trip term (`reconstruction_reverse`,
   weight `reconstructionScale` 0.5 on a $\sim 10^{-4}$ loss) barely
   constrains per-slot content against the output objective, so slot
   norms inflate and unrestricted identity drifts (measured fact B). The
   size-bucketed association absorbs it on xor; the principled fix is the
   DEFERRED 5a rebalance (its precondition — live channels — is now met).
4. Floor calibration: post-stamp, the reverse reconstructs active-slot
   claims at $|.| \sim 0.2$ (5$\times$ attenuated) and leaks $\sim$0.05
   into pads — the 0.1 floor sits between with $\sim 2\times$ margin
   each way. The scaffold gate (primary) removes the fragility from the
   render path; the floor only gates scaffold-less paths.
5. seed 2 / epochs 25 logs `recon_loss == 0.0` for its final epoch — the
   stochastic bernoulli mask drew an empty mask that epoch (the 5b
   warn-once fires); pre-existing stochasticity, not a 5c/5d effect.

### Decision: reconstruction must re-derive tiling BLIND (Alec, 2026-07-04)

The Gate-2 exact-round-trip that landed (5c/5d) is SCAFFOLD-FED: the decode
consumes the forward's own input tiling (`_forward_input['tile_spans']`) as
the render set, so `exact_match==1.0` measures learned CONTENT identity over
analysis-given spans, with the `.where` band CONFIRMING (not dictating)
placement. Alec 2026-07-04: **this is a stepping stone, not the bar** —
reconstruction must re-derive the tiling BLIND (from the representation
alone, no scaffold). Consequences:

- The scaffold render-set gating (5d) becomes a fallback/debug path; the
  real decode must recover the segmentation from the reverse.
- Prerequisite: the `.where` precision problem must be solved first. Today
  the where-band recovers to $\Delta\phi \approx 0.04$ rad at period
  $P=131{,}337$ $\to$ $\Delta x \approx 840$ bytes, useless for blind span
  recovery on ~11-byte rows. Governing relation
  $\Delta x = (\Delta\phi/2\pi)\,P$. SUB-FORK (open, needs a choice +
  a per-frequency phase-noise probe): (i) PERIOD-MATCH $P$ to the data
  range (impactful, caps unambiguous range at $P$); (ii) MULTI-SCALE /
  residue LADDER (coarse period for range + high-frequency rungs for
  resolution; principled; highest useful rung $P_{hi}\approx 2\pi/\Delta\phi
  \approx 157$ for sub-byte; only works if reverse phase-noise is roughly
  frequency-independent -- MEASURE first); (iii) LOSSLESS INTEGER SIDE-BAND
  (`subspace.where.setW(word_offset_grid)` already holds exact offsets; the
  reverse just never reads it -- thread it through, lossless by
  construction, sidesteps phase recovery entirely).
- New Gate-2b: exact round-trip with the scaffold REMOVED (tiling
  re-derived), on xor then grammar.

5a (rebalance) is orthogonal and proceeds now; the blind-recovery thread
(precision sub-fork + scaffold removal) is the next design decision.

### Task 5a execution (reconstruction-channel rebalance; 2026-07-04, local cpu/eager)

**Measured imbalance (probe_rebalance.py, xor seed 0, init).** The training
total assembles as $(1-rr)\,\mathrm{lossOut} + rr\,\mathrm{lossIn} +
rr\,\mathrm{lossRev}$ (ModelLoss.forward, Layers.py $\approx$13780, plus
the lossRev add at Models.py $\approx$4427), $rr =
\mathrm{reconstructionScale}$. At $rr = 0.5$: raw out $= 0.174992$, raw
lossIn (masked-LM) $= 2.364\times 10^{-3}$, raw lossRev (reverse
round-trip) $= 7.717\times 10^{-4}$ $\to$ scaled out $= 8.75\times
10^{-2}$, scaled recon $= 1.57\times 10^{-3}$, ratio $= 0.0179$ (a
$\sim$56x channel imbalance; the reverse term ALONE is $\sim$227x under
scaled output). Per-epoch ratio holds at 0.010--0.018 over 6 epochs --
mis-scaled, not dead.

**Scope decision: (A) config-local.** reconstructionScale is NOT a shared
default -- every config carries its own value (xor 0.5, legacy 0.1,
grammar 0.1; read per-config at Models.py:587). data/MM_20M_xor.xml alone
changed ($0.5 \to 0.85$); legacy/grammar byte-identical BY CONSTRUCTION
and verified by harness re-run (legacy 0.125736/0.004399, grammar
0.168615/0.268405, exact). No code default changed, no re-baseline.

**Constraint found: $rr$ is unitInterval-bounded and CONVEX.** The init
ratio is $rr \cdot 3.135\times 10^{-3} / ((1-rr) \cdot 0.175)$: ratio 0.1
needs $rr = 0.85$; parity needs $rr = 0.98$ (output supervision $\to$ 2%).
whatScale cannot shift the ratio -- lossOut ALSO flows through
ModelLoss.compute (Models.py:4159), so the what/where/when scales multiply
BOTH channels. Within-a-decade is therefore reachable only at the decade's
bottom edge without starving the head; deeper rebalance is structural (a
separate reverse-term weight = option B, deferred).

**Candidates measured** (in-process $rr$ override, full run_config
semantics; exact_match at E $\in \{3,10,20,25,40\}$, seeds 0/1/2):

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| BEFORE $rr=0.5$ | 1.0, 0.0, 1.0, 1.0, 1.0 | 1.0, 0.0, 0.0, 0.5, 0.25 | 1.0, 0.25, 0.5, 0.5, 1.0 |
| $rr=0.85$ (ratio 0.1015) | 1.0, 0.5, 1.0, 1.0, 1.0 | 1.0, 0.0, 0.0, 0.25, 0.25 | 1.0, 0.0, 0.0, 0.0, 1.0 |
| $rr=0.90$ (ratio 0.1613) | 1.0, 0.5, 1.0, 0.75, 1.0 | 1.0, 0.0, 0.0, 0.5, 0.0 | 1.0, 0.0, 0.0, 0.0, 1.0 |

**Choice: $rr = 0.85$.** It reaches the intended decade at init (0.1015),
halves the seed-0 E=10 dip ($0.0 \to 0.5$) with every other seed-0 point
at 1.0, and keeps the pinned bar green. $rr = 0.90$ is DISQUALIFIED: a new
seed-0 dip at the pinned budget (E=25 $= 0.75$ breaks
test_mm20m_xor_exact_roundtrip) and seed-1 E=40 collapses to 0.0.

**Harness (epochs 3, seed 0):** xor 0.187600/0.001812 $\to$
0.183390/0.001658 (output/recon; the expected shift from the reweighted
gradient mix); exact 1.0 and where_recovery 1.0 unchanged. EPOCHS_PINNED
stays 25 (conservative -- the E=10 dip persists reduced, so no basis to
drop the pin).

**Tests:** new pin ``test_recon_channel_within_decade_of_output``
(captures the raw output/reconstruction/reconstruction_reverse init-batch
terms via the TheError.add hook, applies the $rr$ blend, asserts $0.1 \le$
ratio $\le 10$; RED pre-edit at ratio 0.01792, GREEN post-edit at 0.10153,
deterministic cpu/eager seed 0).

**Deviations / findings (numbered):**

1. HONEST RESIDUAL: seed 1 still does NOT converge by E=40 (0.25 before,
   0.25 at $rr=0.85$), and seed 2's mid-trajectory dips DEEPEN
   (0.25/0.5/0.5 $\to$ 0.0/0.0/0.0 at E=10/20/25) though its E=40 endpoint
   stays 1.0. The rebalance, bounded by the convex blend, does not fix the
   content-association confusions (5c/5d measured fact B) -- that residual
   belongs to the blind-recovery/precision thread, not to weighting.
2. The init ratio at $rr=0.85$ sits at the decade's BOTTOM edge with
   $\sim$1.5% margin (deterministic on cpu/eager seed 0). Pushing deeper
   costs output supervision superlinearly; parity would leave the head at
   2%. If parity is wanted, the structural route is a separate
   reverse-term weight (option B, re-baselines legacy/grammar) --
   deferred.

### Falsy-zero scale fix (Alec addendum; 2026-07-04, local cpu/eager)

**Bug.** ModelLoss.__init__ (bin/Layers.py $\approx$13600) used
``float(x or default)`` for all five scales; a configured 0.0 is falsy, so
$0.0 \to$ the default silently. data/MM_xor_loopback.xml and
data/idempotent.xml set ``<reconstructionScale>0.0</reconstructionScale>``
(XSD: "0 = output-only training") yet trained at $rr = 0.5$ -- CONFIRMED
at build: pre-fix loopback reports reconstruction_scale $= 0.5$.

**Fix.** None-vs-0.0 distinction on all five scales
(reconstruction/what/where/when/embedding), each preserving its former
None-default (0.5/0.7/0.2/0.1/0.1). Nonzero configured values were never
coerced, so the 5a xor value (0.85) and legacy/grammar (0.1) are untouched
by this fix -- the interaction surface is exactly 0.0.

**Behavior delta (isolated: falsy fix toggled at a fixed pre-rebalance
tree; epochs 3, seed 0, cpu/eager):**

- MM_xor_loopback.xml: stored scale $0.5 \to 0.0$ (correct); the 3-epoch
  loss trajectory is FLOAT-IDENTICAL pre/post (out 0.175134/0.175144/
  0.175166, recon 0.145339/0.121672/0.074280). The $rr$ weight is
  observably INERT on this config's dynamics: the reconstruction term
  carries grad_fn and reaches 4 optimizer params, but all are transient
  carriers (perceptualSpace.sigma.raw_bfly_{L,d,U},
  conceptualSpaces.0.subspace.event.W -- re-set/re-derived each forward),
  so nothing persistent trains through it.
- data/idempotent.xml: FAILS PRE-EXISTING (autograd inplace RuntimeError,
  ``[104, 10] at version 4``) on the pre-fix tree and identically on the
  post-fix tree -- the failure predates and survives the fix; no
  end-to-end delta measurable. Build-time scale is now honestly 0.0.

**Tests.** New pin ``test_model_loss_honors_explicit_zero_scales`` (0.0
KEPT on reconstruction_scale and what_scale; ``total()`` $==$ lossOut
exactly at $rr=0$; None-unset keeps each documented default). Consumer
sweep over every test file loading either 0.0-config (20 files beyond the
roundtrip file): green -- none pinned the coerced-0.5 behavior.

**5a + falsy verification (final counts):** RUN_SLOW=1
test_reconstruction_roundtrip.py 16 passed (14 prior + ratio pin + falsy
pin; includes THE bar at E=25 and the E=3 fast sibling at $rr=0.85$);
test_where_attention_handoff.py + test_recon_bench.py +
test_sparse_concept_e2e.py 27 passed (7+4+16; determinism bar green);
consumer sweep part 2 (10 files) 71 passed, 1 xfailed; part 1 (10 files)
86 passed, 3 skipped, 1 xfailed, 5 xpassed -- the 5 xpasses (4 in
test_c_prior_slotwise.py, 1 in test_cs_stm_bookkeeping.py, all
CS.forward codebook-snap xfail reasons) are PRE-EXISTING stale xfails:
toggling the falsy fix off reproduces the identical 5 xpasses. One
CONTENTION FLAKE observed and cleared: test_explicit_dimensions.py::
TestXorExactCliReconstruction::test_output_mse_is_crisp failed once
under parallel background load (CLI subprocess 60s timeout vs $\sim$66s
class wall under contention; uses data/XOR_exact.xml -- orthogonal to
both changes) and passes standalone and in the serial sweep.
Harness: xor $0.187600/0.001812 \to 0.183390/0.001658$
(exact 1.0, where_recovery 1.0); legacy $0.125736/0.004399$ and grammar
$0.168615/0.268405$ EXACT (byte-identity held through both changes).

### Decision: precision sub-fork = PERIOD-MATCH + overflow warning (Alec, 2026-07-04)

Lower the `.where` sinusoid period to a low value sized to the data; WARN
(with a raise-it message) if an input ever exceeds it. Range assumption:
8192 = max characters in a SENTENCE (the largest input size in use,
MM_20M_legacy); prior sentences are referenced by `.where` + `.when`
JOINTLY, so the period need not span history. Validation probe FIRST:
measure recovered-position error vs period through the REAL encode+reverse
(the single-sinusoid bound predicts ~$\pm 52$ bytes at $P=8192$ under
$\Delta\phi \approx 0.04$ rad; if WhereEncoding is multi-frequency the
joint decode may beat that bound substantially — measure, then size $P$).
Gate 2b (blind tiling recovery) builds on the outcome.

### idempotent.xml pre-existing inplace crash FIXED: clone the factored-crossing read (2026-07-04)

**Root cause (anomaly-mode + version-watch probe evidence).** The
"FAILS PRE-EXISTING" record above (autograd inplace RuntimeError,
``[104, 10] at version 4``) is the meronomy factored crossing reading a
LIVE codebook view: ``ConceptualSpace._factor_crossing`` (bin/Spaces.py)
passes ``rows = sub.prototype()`` -- after ``.detach().to(...)``, still an
ALIAS of the muxed CS event-codebook Parameter
(``conceptualSpaces.0.subspace.event.W``, $[V{=}10, D{=}104]$) -- into
``factor_percept``, whose ``q @ rows.transpose(-1, -2)`` saves the
transposed view $[104, 10]$ for backward (MmBackward0 needs it for
grad wrt the percept leg). Four later in-graph writers then bump the
Parameter's version $0 \to 4$ before ``totalLoss.backward()``
(Models.py:4654), all the SAME mechanism -- ``set_event`` $\to$
``set_muxed`` $\to$ ``Codebook.quantize`` $\to$ VQ-EMA
``self.codebook.copy_`` (Layers.py:14941): (1) Models.py:4297
``cs.subspace.set_event(terminal_idea)``; (2) Models.py:9083
``_reverse_body`` ``sub.set_event(recovered)``; (3) Spaces.py:7922
``reverseEnd`` via Models.py:9092; (4) Spaces.py:11818 ``_reverse_text``
$\to$ ``normalize`` $\to$ ``put`` $\to$ ``set_event``.

**Why only idempotent.xml.** Meronomy is ON in data/model.xml defaults,
so every config runs ``bind_streams`` $\to$ ``_factor_crossing`` -- but
the working analogues (MM_20M_legacy/grammar) have a PURE-EVENT CS
subspace (``codebook_slot=None``; legacy's ``<codebook>quantize</codebook>``
is commented out), so the crossing returns at the ``rows is None`` gate
and nothing is saved. idempotent.xml uniquely arms the path: CS
``<codebook>quantize</codebook>`` makes the event slot a Codebook
(muxed), and the uniform 104-wide slab passes BOTH width gates (rows
width $==$ bind $D$; event width $==$ codebook ``nDim``, so ``set_muxed``
actually snaps/EMA-writes).

**Fix (source, one site).** The SS-leg clone lesson
(2026-07-02 EXECUTION NOTES item 2; same pattern already applied at the
sibling read ``cs_snap_order0``): CLONE the read.
``_factor_crossing`` now snapshots ``rows = rows.clone()`` after the
width gate (bin/Spaces.py:15954-15958) so the matmul saves the snapshot,
not the live Parameter. Numerically identity: backward now uses the
values as-of-read (the correct math); configs whose crossing skips never
reach the clone.

**Tests + verification.** Failing pin FIRST:
``test_reconstruction_roundtrip.py::test_idempotent_config_trains_one_epoch_clean``
(1 epoch, finite losses; failed with the exact RuntimeError pre-fix,
$\sim$4s, no RUN_SLOW gate needed). Post-fix: the repro
(``recon_bench.py data/idempotent.xml --epochs 3 --seed 0`` cpu/eager)
exits 0 with output_loss 0.17494817078113556 / recon_loss 0.0;
RUN_SLOW=1 test_reconstruction_roundtrip.py 17 passed (16 prior + pin);
test_where_attention_handoff.py 7 passed; nearest-consumer sweep
(test_interface_factoring.py + test_conceptual_recurrence.py +
test_search_then_mint.py) 32 passed, 1 skipped; byte-identity held EXACT:
legacy $0.125736/0.004399$, grammar $0.168615/0.268405$.

### nWhere=0 lossRev wiring fix (Alec-approved re-baseline; 2026-07-04, local cpu/eager, seed 0)

**Root cause (the third silent-channel instance, after 5b's two).**
``ModelLoss.__init__`` sources its what/where/when split from
``canonical_shape("OutputSpace")`` $= (0, 0)$ (bin/Layers.py
$\approx$13613; construction site Models.py $\approx$5610) -- CORRECT for
the lossOut call (Models.py $\approx$4159; output events carry no band by
architecture), but the SAME instance's ``compute`` served the lossRev
path (Models.py $\approx$4315) comparing INPUT events whose muxed layout
is ``[what(1020) | where(2) | when(2)]``
(``canonical_shape("InputSpace")`` $= (2, 2)$; WhereEncoding slots
$[-4, -3]$, WhenRangeEncoding $[-2, -1]$). With ``nWhere == 0`` the whole
1024-dim event took ``what_scale`` and the 2-dim where band's error was
diluted by $2/1024$ inside the what-mean -- ``where_scale`` /
``when_scale`` NEVER applied on lossRev.

**Fix (two sites).** (1) ``ModelLoss.compute`` gains per-call
``nWhere``/``nWhen`` overrides (default ``None`` $\to$ the constructor
band; every existing call is byte-compatible). (2) New seam
``BasicModel._reverse_event_loss(rev_ev, fwd_ev)`` (next to the 5b
``_align_output_pred`` gate): clips to the shared $[K, D]$ window, reads
the live band from ``inputSpace.subspace.nWhere/nWhen``, and passes it
through; runBatch's lossRev now routes through the seam. Fail-loud per
the 5b pattern: an unavailable band (``lossrev_band_unknown``) or a
width clip that would misalign the trailing band
(``lossrev_band_clipped``) warns once and falls back to whole-event
what_scale -- never silently.

**TDD.** Failing pin FIRST:
``test_where_scale_applies_to_lossrev`` (seam-level: fwd/rev differing
ONLY in the where band must yield ``where_scale * MSE(band)``; plus a
real-train-batch spy asserting the 3-dim event compare carries the input
band). Pre-fix: loss $= 3.418\times 10^{-4}$ ($=$ what\_scale
$\cdot$ MSE(band) $\cdot 2/1024$) vs expected $0.05$ ($=$ where\_scale
$\cdot$ MSE(band)) -- the measured $\sim$146x dilution
(``where_scale / (what_scale/512)``). Post-fix: passes.

**Re-baseline (bin/recon_bench.py, epochs 3, seed 0, cpu/eager; pre-fix
values reproduced EXACTLY before the fix, same harness):**

| config | output_loss old $\to$ new | recon_loss old $\to$ new | exact | where_rec |
|---|---|---|---|---|
| `MM_20M_xor` | 0.183390 $\to$ 0.186961 | 0.001658 $\to$ 0.001271 | 1.0 $\to$ **0.5** | 1.0 |
| `MM_20M_legacy` | 0.125736 $\to$ 0.125735 | 0.0043986 $\to$ 0.0043987 | 0.0 | 0.0 |
| `MM_20M_grammar` | 0.168615 (byte-identical) | 0.268405 (byte-identical) | 0.0 | 0.0 |

Full precision: xor $0.18339034914970398/0.0016575129702687263 \to
0.18696050345897675/0.0012714503100141883$; legacy
$0.12573622167110443/0.004398638848215342 \to
0.12573517858982086/0.00439868588000536$ (lossRev is LIVE on the
parallel path, small real shift); grammar EXACT byte-identity -- the D3
dedupe means TRAIN never computes lossRev there (see finding 2).

**Ratio pin re-derived.** Raw init lossRev $7.717\times 10^{-4} \to
5.552\times 10^{-2}$ ($\sim$72x -- the where band's reverse error now
enters at weight 0.2 instead of $0.7/512$); measured init ratio
$0.10153 \to 1.874$. ``test_recon_channel_within_decade_of_output``
keeps its intended $[0.1, 10]$ decade bound UNCHANGED (now mid-decade,
was bottom-edge); the 5a $rr = 0.85$ still lands in the decade -- no
retune needed (docstring updated with the new measurement).

**THE BAR: RED at $E = 25$ -- convergence shifted; proposal pending.**
New seed-0 trajectory (full ``run_config`` semantics):

| E | 3 | 10 | 20 | 25 | 30 | 35 | 38 | 40 | 50 | 80 |
|---|---|---|---|---|---|---|---|---|---|---|
| exact_match | 0.5 | 0.25 | 0.5 | 0.5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| where_recovery | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

1.0 IS still reached -- the plateau moved from $[20, 80]$ to $[30, 80]$
(verified at 30/35/38/40/50/80; the old E=3 early-geometry blip at 1.0
is gone). PROPOSED (not applied -- Alec's knob): ``EPOCHS_PINNED``
$25 \to 38$ ($= 30 \times 1.25$ per the 5c/5d formula; measured 1.0/1.0
at exactly 38). Until decided, ``test_mm20m_xor_exact_roundtrip`` is the
single RED in the file. ``where_recovery`` stays 1.0 at every budget --
the tiling channel is unaffected; the residual is the known
content-association drift (5c/5d measured fact B).

**Pins updated (each flagged in its docstring):**
1. ``test_mm20m_xor_roundtrip_at_harness_budget``: exact\_match at the
   epochs-3 budget re-pinned $1.0 \to 0.5$ (the measured deterministic
   post-fix trajectory point; where\_recovery $== 1.0$ kept).
2. ``test_recon_channel_within_decade_of_output``: assertion interval
   UNCHANGED; docstring records $0.10153 \to 1.874$.
3. ``EPOCHS_PINNED`` comment records the proposal; the VALUE stays 25.
No test pinned the old loss numbers numerically (grepped).

**Verification sweeps:** RUN_SLOW=1 test_reconstruction_roundtrip.py
17 passed + 1 failed (the failure IS the E=25 bar, above);
test_where_attention_handoff.py 7 passed (determinism bar green);
test_recon_bench.py 4 passed; test_sparse_concept_e2e.py 16 passed;
test_ir_only_refactor.py (the ``reconstruction_reverse`` liveness
consumer) 10 passed, 2 skipped. The idempotent pin is green inside the
roundtrip run.

**Deviations / findings (numbered):**
1. ``test_sparse_concept_e2e.py`` fails COLLECTION standalone
   (``ModuleNotFoundError: test_basicmodel``) -- PRE-EXISTING local
   quirk: a gitignored ``test/__init__.py`` (May 9 artifact,
   .gitignore:2) flips pytest to package-mode imports, so the sibling
   ``from test_basicmodel import ...`` needs test/ on sys.path.
   ``PYTHONPATH=test`` is the workaround used; unrelated to this fix.
2. Grammar is byte-identical BY MECHANISM, not accident: the D3 dedupe
   (5b) skips lossRev on train batches, and the D3 objective computes
   through ``self.loss.compute`` at Models.py $\approx$9373 WITHOUT the
   band -- the serial-path twin of this bug (the reverse objective's
   where band is still what-scaled there). Same pattern at the
   ``compute_masked`` lossIn site (Models.py $\approx$4227, PartSpace
   events). Both are FURTHER instances of the silent-channel pattern
   left UNTOUCHED here (each re-baselines additional configs) -- they
   need their own Alec decision.
3. The harness's reported recon_loss is the ``reconstruction`` slot
   ONLY (``BatchResult.lossIn`` $=$ the masked-LM / D3 term; lossRev
   never enters the report, only the training total) -- so xor's
   $0.001658 \to 0.001271$ is an INDIRECT trajectory shift from the
   new where-band gradient in totalLoss, not a re-scaled lossRev
   reading. The raw lossRev itself is visible only via the ratio-pin
   hook ($7.7\times 10^{-4} \to 5.55\times 10^{-2}$ at init).

### Silent-band wiring fix, remaining two sites: D3 twin + compute_masked lossIn (Alec-approved re-baselines; 2026-07-04, local cpu/eager, seed 0)

**Scope.** The two FURTHER instances of the nWhere=0 silent-band pattern
left untouched by the lossRev fix (its finding 2): (1) the D3 twin --
``_d3_reconstruction_loss`` called ``loss.compute`` band-less at
Models.py $\approx$9402; on the serial path the D3 term IS the
reconstruction objective (the dedupe made it so); (2) the whole-slab
masked-LM lossIn -- ``runBatch`` called ``loss.compute_masked`` at
Models.py $\approx$4259, and ``compute_masked`` read the constructor
band ($(0, 0)$, canonical_shape("OutputSpace")) with no per-call
override at all.

**Layout findings (runtime-probed, both bands LIVE -- wiring, not
explicit-zero).** Site 1: grammar's D3 compare is reverse(S)
$[4, 128, 12]$ vs ``inputSpace._ar_embedded`` $[4, 1024, 12]$ -- equal
widths (K-clip only, band never misaligned), INPUT layout
``[what(8)|where(2)|when(2)]``; first-batch band MSE where/when
$0.5556/0.5448$ vs what $0.3022$. Site 2: the masked-LM compares
PERCEPTUAL events -- canonical_shape("PartSpace") $= (2, 2)$, xor
percept ``muxedSize`` 1024, and the band is live and mispredicted
(first-batch masked where-band SE $1.387$ vs what $12.43$), so the
correct fix is band wiring, not an explicit ``nWhere=0``.

**Dilution measured.** Site 1 ($D = 12$): the where band entered the
what-mean at what\_scale $\cdot 2/12 \approx 0.117$ instead of
where\_scale $0.2$ ($\sim$1.71x under); the when band was slightly OVER-
weighted pre-fix ($0.117$ vs when\_scale $0.1$). First-batch compare
$0.2693996$ unbanded $\to 0.3771294$ banded. Site 2 ($D = 1024$): the
same $\sim$146x where dilution as lossRev
(where\_scale/(what\_scale $\cdot 2/1024$)); first-batch lossIn
$0.0023637 \to 0.0370181$.

**Fix (extends the lossRev machinery; no new pattern).** Site 1: the D3
loss now routes through the SAME ``_reverse_event_loss`` seam (identical
clip-to-shared-$[K, D]$ + detach semantics; the seam docstring records
it serves the D3 twin). Site 2: ``ModelLoss.compute_masked`` gains the
same per-call ``nWhere``/``nWhen`` overrides as ``compute`` (default
``None`` $\to$ constructor band; every existing call byte-compatible),
and a new sibling seam ``BasicModel._masked_event_loss(pred, target,
mask)`` owns the K-clip, reads the live band from
``perceptualSpace.subspace.nWhere/nWhen``, and fails loud per the 5b
pattern (``lossin_band_unknown`` when the band is unavailable;
``lossin_band_clipped`` when unequal pred/target widths would misalign
the trailing band under compute\_masked's internal D-clip -- warn-once,
fall back to whole-event what\_scale, never silently).

**TDD (failing pins FIRST, both red with the measured dilution).**
``test_where_scale_applies_to_d3_reconstruction``: only-the-where-band
differs through ``_d3_reconstruction_loss`` itself (staged
``_stm_single_S``/``_ar_embedded``, patched ``_reverse_from_S``) --
pre-fix $0.0291667$ ($=$ what\_scale $\cdot$ MSE(band) $\cdot 2/12$) vs
expected $0.05$ ($=$ where\_scale $\cdot$ MSE(band)); plus a real
grammar-train-batch spy asserting the width-12 event compare carries the
input band. ``test_where_scale_applies_to_masked_lossin``: same
structure at the masked site -- pre-fix $3.418\times 10^{-4}$ vs
expected $0.05$ (the $\sim$146x dilution); plus a real xor-train-batch
spy asserting the masked compare carries the percept band. Both green
post-fix.

**Re-baseline (bin/recon_bench.py, epochs 3, seed 0, cpu/eager; pre-fix
values reproduced EXACTLY before the fix, same harness):**

| config | output_loss old $\to$ new | recon_loss old $\to$ new | exact | where_rec |
|---|---|---|---|---|
| `MM_20M_grammar` | 0.168615 (byte-identical) | 0.268405 $\to$ **0.375690** | 0.0 | 0.0 |
| `MM_20M_xor` | 0.186961 $\to$ 0.186838 | 0.001271 $\to$ **0.015737** | 0.5 | 1.0 |
| `MM_20M_legacy` | 0.125735 $\to$ 0.124488 | 0.004399 $\to$ **0.073159** | 0.0 | 0.0 |

Full precision: grammar $0.16861549019813538$ (byte-identical) /
$0.26840516924858093 \to 0.3756902515888214$; xor
$0.18696050345897675/0.0012714503100141883 \to
0.1868378072977066/0.01573723368346691$; legacy
$0.12573517858982086/0.00439868588000536 \to
0.12448841333389282/0.07315883040428162$. GRAMMAR MOVED -- site 1 is
its live reconstruction term and the intended honest re-baseline.

**Ratio pin re-derived.** Raw init lossIn $0.0023637 \to 0.0370179$
(the banded value predicted analytically from the probe, matched
exactly); measured init ratio $1.874 \to 2.997$ -- still mid-decade,
``test_recon_channel_within_decade_of_output`` keeps its $[0.1, 10]$
bound UNCHANGED (docstring records the new measurement).

**THE BAR: GREEN at $E = 38$, no proposal needed.** Site 2 is xor's
live recon term, so convergence was re-measured: exact\_match/
where\_recovery $= 1.0/1.0$ at $E \in \{30, 38, 50\}$ (full
``run_config`` semantics, seed 0) -- the $[30, 80]$ plateau start
holds. The $E = 3$ harness-budget point also HELD ($0.5/1.0$), so no
assertion value changed anywhere; ``EPOCHS_PINNED`` stays 38.

**Pins updated (docstring measurements only; no assertion changed):**
1. ``test_recon_channel_within_decade_of_output``: docstring records
   raw init lossIn $0.0023637 \to 0.0370179$, ratio $1.874 \to 2.997$.
2. ``test_mm20m_xor_roundtrip_at_harness_budget``: docstring records
   the re-measured (unchanged) $E{=}3$ point and the $30/38/50$ plateau
   verification.

**Verification sweeps:** RUN_SLOW=1 test_reconstruction_roundtrip.py
**20 passed** (18 prior + the 2 new pins; THE BAR and the idempotent
pin green inside); test_where_attention_handoff.py 7 passed +
test_recon_bench.py 4 passed (one run, 11 passed);
test_sparse_concept_e2e.py 16 passed (PYTHONPATH=test, the known
package-mode quirk); test_ir_only_refactor.py 10 passed, 2 skipped.

**Deviations / findings (numbered):**
1. Grammar's output\_loss byte-identity is BY MECHANISM: the D3 term is
   grad-LIVE (probe: grad\_fn present, nonzero grads to 6 params --
   ``inputSpace.outputSpace._vocabulary`` vectors,
   ``perceptualSpace.sigma`` butterfly L/d/U, STM reducer anchors), but
   all 6 sit on the reverse/reduce chain, disjoint from the forward
   head -- so the reported output trajectory cannot move while the
   recon trajectory does (and did: $0.375690 \ne$ first-batch
   $0.377129$, a real gradient-path shift).
2. Site 1 needed NO ``compute_masked``-style signature work: the D3
   compare has exactly the lossRev seam's shape contract, so the fix is
   one line through the existing seam (extend-don't-duplicate held).
3. At $D = 12$ the pre-fix bug was two-sided: where under-weighted
   ($\sim$1.71x) AND when over-weighted ($\sim$1.17x) -- band dilution
   is width-dependent, so the grammar/xor magnitudes differ by orders
   while being the same wiring defect.
4. Legacy's recon term moved $16.6$x ($0.004399 \to 0.073159$) --
   compute\_masked is live on the parallel path there too; its
   exact/where metrics stay $0.0$ (unaffected channel, same as the
   lossRev fix).

### Task 6 execution: MM_20M_grammar $\to$ meronomy/meronomy (2026-07-04, local cpu/eager, seed 0)

**6.1 Reshape (landed).** data/MM_20M_grammar.xml is now
synthesis=meronomy + analysis=meronomy on the XOR-proven dims adapted to
serial mode -- the SAME adaptation the committed serial+radix sibling
data/MM_mereology_serial.xml already carries verbatim (IS 8192/256/1024;
PS 8192/1024/65536/1024/8/1024 + chunkPromotionThreshold 2; CS
8/1024/8/8/1024/8/1024; WS meronomy 8/1024/65536/1024/1024/8 with
butterfly=false KEPT per plan; OS 8/1024/1/1/1/1, nDim $4 \to 1$ = the
0/1 target width, and nInputDim 1024 to satisfy the WS$\to$OS 8192 slab
flatten that the old 8x8=64 OS no longer would). Serial derivation,
symbolicOrder=1, the complete.grammar SymbolSpace, and the training block
(lr 0.0005, batch 64, reconstructionScale 0.1) are UNCHANGED. Every
dimension decision is documented in the config comments.

**6.2 Build + smoke: GREEN.** One epoch (harness):
output\_loss 0.17499743402004242, recon\_loss 0.0026048007421195507,
exact 0.0, where\_recovery 0.25, 4 rows decoded
(output/recon_MM_20M_grammar_20260704-120404_49747.json).

**Label story: RESOLVED as designed (probe-verified).** analysis=meronomy
resolves the lexer to ``raw`` (Spaces.py resolve\_lexer), so
``use_byte_cursor = text_input and byte_lexer`` = False; the trial cursor
delivers the real supervised out\_items. The head-loss target on a real
train batch IS ``[0.0, 1.0, 1.0, 0.0]`` (spy on \_align\_output\_pred:
pred [4,4,1] vs target [4,1,1]). The output term carries grad\_fn and
reaches 6 params (OS \_readout\_bias + layers.0.W, PS what.W, PS sigma
butterfly L/d/U) -- genuinely supervised AND trainable. HONEST CAVEAT:
at grammar's configured lr 0.0005 the head barely moves (pred 0.5002
after 10 epochs; reported output\_loss frozen at 0.175000 $= 0.7 \times
0.25$ through E=80) -- under-trained, not dead.

**Old $\to$ new (harness, epochs 3, seed 0):** output
$0.168615 \to 0.175000$; recon $0.375690 \to 0.002443$ (the D3 term now
compares 1024-wide meronomy events, and only ONE recovered slot after the
K-clip -- see the residual below); exact 0.0 $\to$ 0.0; where\_recovery
0.0 $\to$ 0.25.

**6.3 Round-trip: STOPPED AT THE MEASUREMENT (plan's not-reachable arm) --
no pin added, bar NOT weakened.** Convergence grid (full run\_config
semantics, seed 0):

| E | 3 | 10 | 20 | 30 | 40 | 60 | 80 |
|---|---|---|---|---|---|---|---|
| exact_match | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| where_recovery | 0.25 | 0.25 | 0.25 | 1/3 | 1/3 | 1/3 | 1/3 |
| recon_loss | 0.002443 | 0.002012 | 0.001542 | 0.001206 | 0.000970 | 0.000688 | 0.000512 |

**The precise residual (mechanism nailed, every step measured):**

1. WHICH ROWS: all four (each is 3 tiles, word/space/word). The ANALYSIS
   side is PERFECT -- store promotes exactly [hello, loving, there,
   world], tile\_spans match the true tilings, per-row active pids ARE
   the word tiling ([hello, ' ', world] etc.).
2. WHAT DIFFERS: the decode renders exactly ONE token per row (E=10:
   'world' / 'loving' / 'loving' / 'loving' -- rows 1-3 render content
   from the WRONG row) at where\_abs noise $\sim$1234.
3. WHERE THE DECODE DIVERGES: serial reduces the sentence to ONE deep
   idea before any reverse -- measured STM occupancy after the walk is
   slot 0 only (norms [1.854, 0, ..., 0]; \_stm\_reduce\_to\_single\_S).
   \_reconstruction\_seed's serial arm (Models.py $\approx$3536) seeds
   ``snap[:, :1, :]`` [4,1,1024]; with the meronomy pair's EQUAL PS/CS
   widths the reverseBegin wide$\leftrightarrow$deep regroup is an
   identity, so the recovered event is [4,1,1024] -- ONE percept slot,
   structurally, at every budget. \_decode\_radix\_meta's 5d scaffold
   gate requires recovered $N ==$ pid\_grid $N$ (8), so the 5e tile
   scaffold is bypassed and the noise-claims fallback renders the single
   slot. The D3 objective K-clips reverse(S) [4,1,1024] against
   \_ar\_embedded [4,8,1024] $\to$ slot 0 only, so the decreasing
   recon\_loss supervises one slot's content.
4. THE OLD SHAPE CANNOT HOST THE PAIR: the pre-switch deep-CS dims
   (PS 12-wide, CS 1028) gave the regroup its capacity (1 idea $\to$ 128
   wide slots -- the old D3 compare [4,128,12]), but meronomy on those
   dims CRASHES in reverseEnd (``reshape [4,1,12] invalid for 6144``,
   probed on a scratch fixture) -- the meronomy reverse bookkeeping
   assumes the equal-width deep shape. So no CONFIG choice reaches the
   bar: equal widths cap the serial reverse at one slot; unequal widths
   crash the meronomy reverse.

NEXT (Alec's design decision, same family as the blind-tiling Gate-2b
fork): either the serial decode seeds the PRE-reduce per-word STM ideas
(parallel-style multi-slot reverse), or the single-S contract gets a
capacity-matched regroup under meronomy, or serial reconstruction is
defined at a coarser granularity. Until then grammar has NO roundtrip pin
(the plan's explicit not-reachable arm); the liveness/label pins cover it.

**Regressions (all green, measured counts):** RUN_SLOW=1
test_reconstruction_roundtrip.py + test_recon_bench.py 24 passed in one
run (20 + 4; THE xor bar at E=38 green inside); harness byte-identity:
xor $0.1868378072977066/0.01573723368346691$ (exact 0.5, where 1.0) and
legacy $0.12448841333389282/0.07315883040428162$ reproduce EXACTLY
(untouched paths); test_sparse_concept_e2e.py +
test_where_attention_handoff.py + test_dimensional_governance.py (fast
tier) 32 passed 1 skipped (16 + 7 + 9; the skip is the RUN_SLOW deep-CS
test, run separately: 1 passed, now 7.9s on the inline fixture);
test_conceptual_recurrence.py 13 passed 1 skipped (pre-existing skip;
incl. test_mm5m_grammar_builds_and_forwards on the reshaped config).

**Deviations / findings (numbered):**

1. The two test_dimensional_governance.py tests premised on grammar's OLD
   deep-CS shape (test_serial_relaxes_symbol_dim_passthrough,
   test_deep_cs_reverse_round_trips_to_ps_width) now build that shape from
   an inline fixture (_DEEP_CS_SERIAL_XML, the pre-switch grammar space
   blocks verbatim) via the file's existing _build_from_text helper -- the
   deep-CS regroup coverage is PRESERVED, not deleted; assertions
   unchanged. Repointing to MM_5M_IR was tried and rejected (its forward
   dies on the xor loader, ``reshape [1,-1,1024] invalid for 12``).
2. Two docstring-only updates in test_reconstruction_roundtrip.py, no
   assertion changes: test_grammar_output_loss_not_silent_zero records the
   label-story resolution; test_where_scale_applies_to_d3_reconstruction
   notes its historical widths are the pre-switch shape (it reads D from
   the built model, so the pin is shape-agnostic).
3. where\_recovery 0.25 $\to$ 1/3 across the grid is the single rendered
   slot's span occasionally matching tile 0 -- NOT a tiling-channel
   regression (the forward tiling is exact; the metric counts the missing
   2 tiles against).
4. WS nVectors 1000 $\to$ 65536: the old value was the byte-path
   grammar-op budget; the meronomy WS mirrors the proven pair's store
   budget. The complete.grammar machinery builds and the D3/serial path
   runs (\_d3\_active True on train batches) -- no observed conflict.

### Task 7 execution: canonical config matrix (2026-07-04, local cpu/eager, seed 0)

**Files.** New `test/test_config_matrix.py` (parametrized SMOKE over 8
configs) + four `data/matrix/` variants, each a COPY of its parent with ONE
architecture flag flipped (header comment names the flip, no `--`; the
`<architecture>` complexType is `xs:all` so the flag can sit anywhere and
still validate against model.xsd). `run_config` already had the
`max_batches` passthrough (Task 1), so no harness change was needed.

**The 8-row matrix (SMOKE: build + one capped epoch + decode executes +
finite losses; `max_batches=SMOKE_BATCHES=1`, which is a WHOLE xor epoch
here -- the 4-row dataset is 1 batch/epoch at every config's batchSize).**
Per-config first-build smoke (`recon_bench.py --epochs 1 --max-batches 1`,
1-batch wall INCLUDING warm-up):

| # | config | flip off parent | 1-batch wall s | output_loss | recon_loss | exact | where_rec | decode |
|---|---|---|---|---|---|---|---|---|
| 1 | `data/MM_20M_grammar.xml` | (base: meronomy pair) | 4.44 | 0.174997 | 0.002605 | 0.0 | 0.25 | clean |
| 2 | `data/MM_20M_xor.xml` | (base: parallel sO=0) | 3.24 | 0.174992 | 0.037018 | 0.75 | 1.0 | clean |
| 3 | `data/MM_20M_legacy.xml` | (base: bpe/byte) | 1.05 | 0.174975 | 0.074886 | 0.0 | 0.0 | clean |
| 4 | `data/matrix/MM_20M_xor_noraise.xml` | mereologyRaise true$\to$false | 3.18 | 0.174992 | 0.037018 | 0.75 | 1.0 | clean(*) |
| 5 | `data/matrix/MM_20M_grammar_stack.xml` | subsymbolicStack on | 4.51 | 0.174997 | 0.002605 | 0.0 | 0.25 | clean |
| 6 | `data/matrix/MM_20M_grammar_reading.xml` | readingAttention on | 4.40 | 0.175004 | 0.002599 | 0.0 | 0.25 | clean |
| 7 | `data/matrix/MM_20M_grammar_twopass.xml` | learning + exploreTemperature | 7.69 | 0.174997 | 0.002605 | 0.0 | 0.25 | clean |
| 8 | `data/MM_sparse_concept.xml` | (base: sO=3 wave, smoke ONLY) | 1.23 | 0.174994 | 0.000457 | 0.0 | 0.0 | clean(*) |

All 8 BUILD and RUN with finite losses and NO `decode_note` (the decode
path executed on every row). (*) rows 4 and 8 emit the by-design
`reconstruction_zeroed` warn-once (5b): with mereologyRaise off (row 4) /
the dark wave (row 8) a batch stages no masked-LM inputs and no D3, so the
recon term is zeroed WITH a warning -- but the REPORTED recon_loss is still
finite/nonzero (the reported value is the reconstruction slot over the epoch,
not that single zeroed batch) and no `decode_note` is added, so the SMOKE
assertions hold. Row 8 also emits the documented `CS snap block exhausted`
overflow warnings (4-row snap block, `_csw_overflow`). Neither warning is a
failure.

**Variant build verdicts (Task 7.1: does the current code accept the flipped
flag on that base?).** All FOUR variants build and run; none dropped, none
needed a flag fix. Evidence the flip is LIVE (not silently ignored):
- row 4 xor_noraise: the `reconstruction_zeroed` warn-once fires (raising
  off changes the recon-channel wiring on the parallel path) -- flag active.
- row 6 grammar_reading: output_loss shifts 0.174997 $\to$ 0.175004 (the
  readingAttention module is allocated and its next-word CE term contributes)
  -- flag active.
- row 7 grammar_twopass: the train batch runs TWICE (two `batch = 0` deltas
  logged: $\Delta$3.400s then $\Delta$3.597s) -- the `<learning>` two-pass
  soft-superposition is active; the double forward is why it is the slowest
  row (~7.7s). exploreTemperature 0.7 rides as its companion.
- row 5 grammar_stack: losses byte-identical to the base grammar on the
  1-batch smoke (subsymbolicStack construction is RNG-neutral per the XSD;
  the distinct per-pass PS sigmas / WS pis are built but the single-batch
  numerics coincide) -- the flag is parsed and the stack allocated; a longer
  run would diverge. Verdict: accepted, smoke-clean.

**Timing + the RUN_SLOW gate (Task 7.1 sizing, measured under load avg ~4).**
The full 8-config matrix in pytest is BUILD-BOUND (8 fresh model builds;
`max_batches=1` makes the epoch itself trivial) at **89-97 s** wall
(`make test`-style single-process, three runs: 89.02 / 93.53 / 97.10 s).
That brushes/exceeds the ~90 s target, so the two HEAVIEST grammar variants
(row 6 readingAttention, row 7 two-pass) gate behind RUN_SLOW: the FAST tier
(6 rows) is **69-72 s** (measured 69.37 / 71.40 s), solidly under budget,
and `make test_all` (RUN_SLOW=1) runs the full eight (93.53 s). Rationale:
the gated pair is the same serial-grammar BASE already smoked fast by
grammar_stack (row 5), so `make test` still exercises a serial-grammar
variant build; the fast tier keeps all THREE base paths + both cheap
variants (xor_noraise on the parallel path, grammar_stack on the serial
path). All 8 param IDs stay visible in `--collect-only` (the two are skipped,
not removed). `SMOKE_BATCHES = 1`.

**RUN_SLOW round-trip tier (Task 7.2/7.3).** NOT duplicated: the file's
footer comment points at the existing xor bar
`test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip`
(EPOCHS_PINNED=38, RUN_SLOW-gated there). NO grammar round-trip is added --
the grammatical-derivation round-trip is the DEFERRED design fork (Task 6
notes: the serial single-S reduce caps the reverse at one slot). Grammar
rides the matrix at the SMOKE tier only.

**Verification (Task 7.4).**
- `test/test_config_matrix.py` FAST tier: 6 passed, 2 skipped, 69.37 s.
- RUN_SLOW=1: 8 passed, 93.53 s. All 8 param IDs visible in
  `--collect-only`. (`make test_all` sets RUN_SLOW=1; `make test` does not
  -- Makefile:104.)
- Regression -- RUN_SLOW=1 `test_reconstruction_roundtrip.py` +
  `test_recon_bench.py`: **24 passed** (20 roundtrip pins incl. THE xor bar
  at E=38 + 4 recon_bench), 226 s -- matches the Task-6 baseline count, zero
  regression from the new files.
- Harness byte-identity spot check (epochs 3, seed 0): xor
  0.1868378072977066 / 0.01573723368346691 (exact 0.5, where 1.0) and legacy
  0.12448841333389282 / 0.07315883040428162 reproduce EXACTLY -- the four new
  `data/matrix/` files + the new test touch no `bin/` code and no existing
  config, so the shared paths are byte-identical by construction and verified.

**Deviations / findings (numbered):**
1. `data/matrix/` did not exist; created it. The variant files are REAL
   files authored at design time (not generated at runtime), so `make test`
   collects them.
2. The grammar_twopass "one flag" is two coupled XML elements (`<learning>`
   + its companion `<exploreTemperature>`) -- the plan names them together
   ("learning+exploreTemperature"); `<learning>` is the knob, exploreTemperature
   (0.7) is its pass-B temperature. Counted as the single logical flip.
3. subsymbolicStack (row 5) is byte-identical to base grammar on the 1-batch
   smoke -- expected (RNG-neutral construction); the divergence is a
   longer-run property, out of scope for a build/finite-loss SMOKE.

### Task 8 execution: speed pass on ArborStudio (2026-07-04, M4 Max 36GB, macOS 26.5.1, via wikiOracle.org tunnel)

**Harness finding (load-bearing): `MODEL_COMPILE` was INERT under `recon_bench` until this task.**
`enable_compiled_step` (the `torch.compile` enlistment) is called only by `ModelFactory.run` and
`test/brick_recon.py`; `recon_bench` builds via `BaseModel.from_config` + `runEpoch`, so `_compiled_step`
stayed `None` and every "compiled" bench run was eager. PROOF: the Task-8.1 remote grammar profiles for
`MODEL_COMPILE=eager` vs `=auto` are byte-identical op tables (163313 `aten::copy_`, 30771
`CopySlices`, no dynamo frames); the Task-2/Task-3 `compile_mode` field was recorded but had no effect on
computation. Fix: a `--compiled-step` opt-in flag on `recon_bench` (default OFF, so the deterministic
cpu/eager test path is unchanged and byte-identical). The flag reorders construction RNG (an
`_stm_reducer` pre-warm) so its fixture output shifts $0.17478 \to 0.17502$; WITHIN the flag,
cross-backend runs are clean (`none` == dynamo-`eager` bit-identical, $0.17502212524414062$). The
FIDELITY GATE (`test_reconstruction_roundtrip.py` + `test_recon_bench.py`, cpu/eager, RUN_SLOW=1)
stays **24 passed** with the flag added.

**8.1 Profiles (remote, cpu, EPOCHS=3, `--profile`; top-3 self-CPU per config).**
grammar (`MM_20M_grammar`, meronomy pair, serial):
1. `aten::copy_` 53% (1.563 s, 163 313 calls) under `torch::autograd::CopySlices` (55% total, 30 771
   nodes) — sliced in-place writes on the serial per-word path (STM shift / event staging); graph
   bookkeeping, not math, dominates.
2. `aten::mm` only 6.3% (480 calls) — compute is a small fraction; the config is OVERHEAD-BOUND at the
   4-row dev scale.
3. small-op storm: `add_` 34 974 / `fill_` 7 441 / `select` 136 638 / `as_strided` 178 543 calls.
xor (`MM_20M_xor`, parallel, sO=0):
1. elementwise storm `add` 11.7% + `mul` 11.5% + `sub` 9.7% + `fill_` 9.0% (fusion territory — why
   inductor helps xor, below).
2. `aten::copy_` 9.9%.
3. one-off big ops: `cumsum` 6×5.8 ms (span-carrier prefix sum, `Spaces.py:19876`), `remainder` 3×8.8 ms
   (`Lexicon.wrap` legacy torus wrap, `Layers.py:432`) — once/epoch, minor.
The grammar-`auto` profile == grammar-`eager` profile (the inertness proof above).

**8.2 The ladder (remote, cpu, EPOCHS=5, `--compiled-step`, seed 0; steady = mean of the settled last-2
epochs; epochs/hour = 3600/steady).**

| config | backend / mode | e1 (warm-up) s | steady s/epoch | epochs/hour | fidelity vs eager | verdict |
|---|---|---|---|---|---|---|
| grammar | none | 1.88 | 1.498 | 2403 | — (ref) | baseline |
| grammar | eager (dynamo) | 109.4 | 1.482 | 2430 | bit-identical | ~parity, huge warm-up |
| grammar | inductor auto (max-autotune) | 522.2 | 1.735 | 2075 | bit-identical | 15% SLOWER steady |
| grammar | inductor reduce-overhead | 458.5 | 1.721 | 2092 | bit-identical | 15% SLOWER steady |
| xor | none | 1.16 | 0.796 | 4524 | — (ref) | baseline |
| xor | eager (dynamo) | 6.38 | 0.782 | 4601 | bit-identical | parity |
| xor | inductor auto (max-autotune) | 38.0 | 0.711 | 5062 | DRIFTS | 11% faster steady |
| xor | inductor reduce-overhead | 31.1 | 0.708 | 5085 | DRIFTS | 11% faster steady, cheaper warm-up |

FIDELITY: grammar is bit-identical across ALL four backends ($0.17500004172325134$ /
$0.10008491575717926$). xor `none`==`eager` bit-identical ($0.211443/0.005525$) but the inductor modes
SHIFT numerics ($0.215902/0.005104$; `exact` $0.25\to0.0$, `where` $1.0\to1/3$ at E=5) — float
re-association in fused kernels. THIS is why the round-trip gate pins cpu/eager.

**Compile-cost census (Alec's rung; `TORCH_LOGS=recompiles,recompiles_verbose`, `MODEL_COMPILE=auto`,
EPOCHS=2). VERDICT: PYTHON-VALUE (length) guard churn — hypothesis (1).**
- Hyp (3) autotune REJECTED by the ro-differential (grammar ro 458 s ~= auto 522 s warm-up: skipping
  autotune benchmarking barely moved it). Hyp (2) one-giant-graph REJECTED: `unique_graphs=31` (not 1),
  and `compile_times()` shows ~40+ small compiles (0.03–15 s each), not one 500-s frame.
- Hyp (1) CONFIRMED. grammar: `unique_graphs=31`, `calls_captured=44904`, and dynamo HIT
  `cache_size_limit=8` — `convert_frame.py [8/8] "torch._dynamo hit config.recompile_limit (8)"` — after
  which that frame runs EAGER for the rest of the run. xor CONTROL: `unique_graphs=3`,
  `calls_captured=6710`, ZERO graph breaks, ZERO recompiles, limit never hit (parallel path is clean, as
  expected).
- THE churning frame, verbatim: `_per_word_body_step` (`bin/Models.py:7656`), recompiled [8/1]…[8/7]
  then limit-hit [8/8], guard `len(out_slot) == 1` … `== 7`
  `# if out_slot is not None and isinstance(out_slot, list):  # bin/Models.py:7828`. `out_slot`
  (`self._per_word_contributions`, Models.py:7975) is the per-word STM accumulator LIST; it grows +1 per
  word via `out_slot.append(contribution)` (Models.py:7829), so each distinct sentence word-count is a new
  python-constant guard → new compile. The 4 dev rows span word-counts 1..7 → 7 recompiles → blows the
  limit of 8. This is exactly Alec's "compiling multiple bodies parameterized by a vector changing length".
- Secondary python-value guards (small recompiles each): `gaussian_window_word` (`Models.py:3020`, guard
  `p == 0`, word-position int), `word_at` (`Spaces.py:8980/8992`, guard `p == 0` / `p >= slab.shape[1]`,
  tick-position int), `_sentence_prelude` (`Models.py:7648`, `not hasattr(self,'_prelude_pumps')`, a
  once-only attr flip).
- One TENSOR-size guard, but STRUCTURAL/one-shot, not per-batch churn: [8/1] `tensor
  wholeSpace…what.W size mismatch at index 0. expected 2, actual 64` — the WS percept CODEBOOK growing
  2→64 rows as promotion fills it; self-settles once the fixed rows stop minting percepts (why rung-2
  epochs ≥4 are quiet). Torch's own hint: `force_parameter_static_shapes=False`.
- The 12 grammar graph breaks are the tolerated `Tensor.item()` `@torch.compiler.disable` router islands
  (gb0124, `capture_scalar_outputs=False`) — expected under the relaxed-fullgraph host-island gate,
  benign.

FIX CLASSIFICATION (RECOMMENDATIONS ONLY — no code changed this task; Alec's call):
- Primary (`out_slot` length churn) — PYTHON-LENGTH, so per the scheme:
  (a) PAD-TO-FIXED-K (the wave's fixed-K discipline, and the natural fit for the "word tensors are all the
      same size" invariant): accumulate into a preallocated `[B, K_max, D]` buffer indexed by word position
      instead of `list.append`; `K_max` = max sentence words → ONE static shape → one compile. Preferred.
  (b) EAGER-ISLAND: `@torch.compiler.disable` the per-word accumulation so the length-varying python stays
      out of the graph (joins the ~8 existing islands). Smaller diff; keeps per-word compute eager.
- Secondary (position ints in `gaussian_window_word` / `word_at`): canonicalize — hoist the `p == 0`
  special-case out of the compiled region or mark the loop index a non-guarded value.
- Structural (codebook `W` growth): pre-size WS/PS `what.W` to the final row budget at build, or set
  `force_parameter_static_shapes=False`. Low priority (self-settles).

**Rung: prefetch (`<numWorkers>`).** GREP: NO `MM_20M_*` config sets `<numWorkers>` — the default is 0
(`data/model.xml:189`, synchronous in-process). Measured on a scratchpad COPY
(`MM_20M_grammar_nw2.xml`, `numWorkers=2`; `data/` untouched): steady **1.527 s/epoch** vs none 1.498 —
a 2% REGRESSION (thread + queue overhead with nothing to hide), losses bit-identical
($0.175000/0.100085$). NOTE: the `_TickPrefetcher` (Models.py:5029) is a SINGLE background thread hiding
`next_tick` CPU cost behind compute; `numWorkers` sets QUEUE DEPTH, not thread count. On the 4-row dev
dataset one tick = one whole epoch, so there is nothing to prefetch ahead — the measured no-op/regression
is exactly expected; the knob only earns its keep on a multi-batch real dataset. RECOMMENDATION: leave the
shipped default 0; introduce `numWorkers>0` only on a real dataset (and re-measure there, ideally on CUDA
where GPU compute releases the GIL for the worker to overlap).

**Rung: batch sweep.** The trial cursor sets `num_streams = batchSize` (Models.py runEpoch) and the xor
dataset has 4 rows, so `batchSize` clamps to ≤4 rows/tick regardless of the XML value. Measured on
scratchpad copies (grammar `batchSize` ∈ {2, 128, 256}; default 64), steady s/epoch:
bs128 **1.496** ≈ bs256 **1.503** ≈ none 1.498 (all clamp to the 4 rows in ONE tick — a true no-op, losses
bit-identical); bs2 **2.363** = 57% SLOWER (splitting 4 rows into 2 ticks DOUBLES the per-tick serial
overhead, `out` drifts to $0.175031$ from the different batching). So on this data the only movement is
DOWNWARD from under-filling. RECOMMENDATION: `batchSize` is purely a real-dataset knob; no dev-scale
signal to raise it, and lowering it below the row count only hurts. Re-sweep on the real dataset (larger
batches amortize the fixed per-tick serial cost that dominates the profile).

**GPU rung (BASICMODEL_DEVICE=gpu = MPS on the M4 Max, eager — GPU-first headline attempt). BOTH CONFIGS
BLOCKED ON MPS (reported, not fought, per the plan).**
- grammar-mps: CONSTRUCTION crash — `RuntimeError: Expected a 'mps' device type for generator but found
  'cpu'` at `bin/Language.py:2365` `_make_lex_gate` (`torch.randn(lex.weight.shape, generator=gen)*0.05`
  builds a CPU `torch.Generator` under an MPS default device). Chain: `from_config` →
  `_create_per_stage` (Models.py:6131) → `SyntacticLayer` → `_attach_per_space_syntactic_layer`
  (Language.py:11616) → `_make_lex_gate`. The grammar `complete.grammar` layer never builds on MPS — a
  real device-portability gap, orthogonal to speed.
- xor-mps: runs partway, then MPS OOM (`kIOGPUCommandBufferCallbackErrorOutOfMemory` — the 65536-row
  codebook forward on 36 GB) followed by the house fail-loud guard doing its job:
  `WholeSpace.insert_meta: fused_vec contains NaN/Inf on fresh META insert` (MPS emitted NaN; the guard
  raised rather than seeding the codebook).
- GPU HEADLINE TODAY = NONE (MPS blocked on both). The GPU number is deferred to the CUDA server; the M4
  MPS is not a usable GPU proxy for these two configs today.

**Epochs/hour before → after (ArborStudio, the headline).**
- MM_20M_grammar: the Task-3 remote baseline (1.124 s/epoch, cpu/`auto`-but-inert) was the PRE-meronomy
  grammar config; today's config is the heavier post-Task-6 meronomy pair. Best LANDED grammar today =
  cpu/eager (or none), steady **1.482 s/epoch = 2430 epochs/hour** (compile provably does not help grammar
  at steady state; it is 15% SLOWER under inductor and only reaches parity after a ~110-s dynamo warm-up).
  The 1.124→1.482 change is the architecture switch (meronomy pair), not a tuning regression.
- MM_20M_xor: baseline cpu/none **0.796 s/epoch = 4524 epochs/hour** → best cpu/inductor-reduce-overhead
  **0.708 s/epoch = 5085 epochs/hour** (+12%) — BUT inductor drifts xor numerics off the eager pins, so
  this is a THROUGHPUT-only win to be taken only when the fidelity pins are not the gate.

**Debug vs production compile defaults (Alec policy "skip autotune for debug"; amortization break-even =
warm-up / per-epoch-gain).**
- grammar: DEBUG `MODEL_COMPILE=none` (compile never pays back — steady is slower under inductor at every
  horizon until the `out_slot` churn is fixed). PRODUCTION `none` as well, UNTIL the pad-to-fixed-K fix
  lands; then re-measure (a single static per-word graph should let inductor amortize).
- xor: DEBUG `none`. PRODUCTION `reduce-overhead` for runs ≳ 350 epochs (ro warm-up ~30 s over none, gain
  ~0.088 s/epoch → break-even ~340 epochs; ro ties auto's steady while skipping autotune per policy) —
  with the standing fidelity caveat that inductor shifts xor numerics vs the eager pins, so seeded
  fidelity comparisons stay cpu/eager.
- CAVEAT on all break-evens: measured on the 4-row dev dataset; real-data epochs are much longer, so the
  per-epoch gain scales up while compile cost stays ~fixed → production break-evens arrive EARLIER than
  the dev-scale numbers.

**CUDA-server transfer outlook (which rungs carry to the borrowed CUDA box).**
- COMPILE: YES and MORE SO. Inductor+CUDA has real fused-kernel + CUDAGraph headroom the CPU lacks (the
  `reduce-overhead`/`max-autotune` CUDAGraph modes are built for CUDA); the ArborStudio native-path
  inductor result de-risks the compile half (it builds and is bit-identical on grammar). BLOCKER that
  transfers WITH it: the grammar `out_slot`-length recompile churn hits `cache_size_limit=8` on ANY
  backend (it is a dynamo-level guard, device-independent) — so the pad-to-fixed-K (or eager-island) fix
  is the prerequisite for grammar to benefit from CUDA compile. xor already compiles clean (3 graphs, no
  churn) and should win immediately on CUDA.
- MPS: NO — does not transfer (MPS-specific). Both of today's MPS failures are Apple-Metal artifacts (the
  CPU-generator device mismatch is an MPS default-device quirk; the OOM/NaN is Metal). CUDA has none of
  these; the grammar generator bug should be fixed regardless (it is a latent non-CPU portability defect).
- WORKERS / BATCH: YES — both are real-dataset knobs that were structural no-ops only because the dev
  dataset is 4 rows / 1 batch per epoch. On CUDA with a real multi-batch dataset, `numWorkers>0` (queue-depth
  prefetch behind GPU compute that releases the GIL) and larger `batchSize` are the standard throughput
  levers and should be re-swept there.

**Files changed:** `bin/recon_bench.py` (added the `--compiled-step` flag + `compiled_step`/
`torch_compile_mode` record fields; default-off, gate stays 24-green), this plan's Task-8 checkboxes +
these EXECUTION NOTES. No shipped config edited (numWorkers/batch/compile-mode are RECOMMENDATIONS; the
scratchpad config copies live outside `data/`). Run records + profile tables in `output/` (gitignored).

**Deviations / findings (numbered):**
1. The whole speed pass rests on the harness fix above: every pre-Task-8 `compile_mode` value in the run
   records (Tasks 2/3) is INERT (eager regardless). The Task-3 "auto vs eager 1.8× on xor" speedup was
   HARDWARE (M4 vs M2), not compilation. Records stand as timing data; their `compile_mode` field is
   descriptive-only for those runs.
2. Grammar is bit-identical across backends BY MECHANISM: with the `out_slot` frame limit-hit reverting to
   eager and the serial reduce/reverse chain disjoint from the compiled head, the forward numerics cannot
   move — consistent with the Task-5/6 byte-identity findings.
3. The MPS grammar generator bug (Language.py:2365) is a genuine latent defect (CPU `Generator` under a
   non-CPU default device), surfaced only because this rung is the first GPU build of the meronomy-pair
   grammar. Flagged as a follow-up (not fixed here — out of the speed pass's scope, and no numerics gate
   covers it yet).
