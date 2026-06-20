# model.xsd refresh + schema validation gate

Date: 2026-06-05

Goal (user request): bring `data/model.xsd` up to date with recent config
refactoring, have every model XML reference it, and run XSD validation as a
precursor to model creation. Also: drop the obsolete `butterflyN` knob.

## Schema changes (`data/model.xsd`)

* **`butterfly` added to `conceptualSpaceType` and `symbolicSpaceType`.**
  Only `perceptualSpaceType` had it, but XOR_exact / MM_xor set
  `<butterfly>` on CS and SS too (the SS cross-slot cascade is what lets
  XOR combine its two word slots).
* **`butterflyN` removed** from `perceptualSpaceType`. The cascade length
  $N$ is auto-computed from the space shape ($\textrm{nInput} \times
  \textrm{nInputDim}$) and padded to the next power of two; there is no
  explicit length knob (Spaces.py builds the PiLayer with the computed
  $N$). No data XML used it; `self.butterflyN` is an internal attribute.
* **`codebookModeEnum` now also accepts the legacy booleans `true` /
  `false`** (aliases `true` $\to$ `quantize`, `false` $\to$ `none`) to
  match `Space.normalize_codebook_mode`'s backward-compat parsing. Several
  configs / test fixtures still use the boolean form.
* **`useStackRouter`** added to `symbolicSpaceType` (live: SymbolicSpace
  reads it to gate its stack-rewrite LanguageLayer dispatch).
* **`reverseScale`** added to `trainingType` (deprecated alias for
  `reconstructionScale`; `util.TheXMLConfig` remaps it with a warning).
* **`trainEmbeddings` / `trainEmbedding`** added to `architectureType`
  (Models.py reads them from the architecture block ahead of
  `<training><trainEmbedding>`; the plural boolean form maps
  `true` $\to$ BOTH, `false` $\to$ NONE).

All 39 `data/*.xml` validate against the updated schema
(`xmllint --noout --schema data/model.xsd data/<f>.xml`).

## Schema reference in every model XML

Each `data/*.xml` `<model>` root now carries
`xmlns:xsi=".../XMLSchema-instance"` +
`xsi:noNamespaceSchemaLocation="model.xsd"` (editor / tooling support).
The Python config parser ignores these root attributes (it iterates
section children), so loading is unaffected.

## Validation gate (`bin/util.py`)

`XMLConfig._validate_against_schema` was a soft, lxml-only, no-op-when-
lxml-absent warning. It is now the **hard precursor gate** for model
creation (`create_from_config` $\to$ `init_config` $\to$ `load`/`overlay`
$\to$ here):

* Resolves the schema as the XML's sibling `model.xsd`, else the canonical
  `<project>/data/model.xsd` (so `/tmp` variant configs written by tests
  still validate against the real schema).
* Validates via `lxml` if importable, else the `xmllint` CLI; if neither
  backend exists it warns once and skips (does not block on missing
  tooling).
* **Raises `ValueError` on a schema violation** so an invalid config never
  silently builds a model (fail-loud).
* Caches successful validations by `(abspath, mtime)` so the defaults file
  and repeated loads don't re-spawn the validator.

## Test cleanup (strict-everywhere consequence)

Making the gate hard-fail everywhere surfaced pre-existing cruft in
test-generated configs. Cleaned:

* **Retired/dead elements removed** from 13 test files: `maskedPrediction`
  (retired 2026-05-14), legacy `type` (superseded by `modelType`),
  `useGrammar` (derived, not read), `useVQVAE` (retired 2026-05-29), and
  the old per-tier naming in `test_basicmodel.py` (`nPercepts`,
  `inputDim`, `perceptDim`, ...). Empty sections left behind (e.g. an
  emptied `<WordSpace>`) were dropped too (an empty element parses to a
  string and breaks `_deep_merge`).
* **Misplacement bugs fixed**: `_build_*` helpers in
  `test_lexicon_ownership.py` and `test_basicmodel.py` (x6) appended
  `<autoload>` directly under `<architecture>`, but the code reads
  `training.autoload` (`_t("autoload")`); the value was silently ignored
  (default `autoload=true`, a no-op in tests with no checkpoint). Now
  nested under `<architecture><training>`. A `test_basicmodel` factory
  config also had `<nInput>/<nOutput>` directly under `<architecture>`
  (moved into `<InputSpace>`).
* **False positives left alone**: `<n>` (`op_I<n>` docstring), `<config>`
  (`data/<config>.xml` docstring), `<pre>` (HTML in the report writer),
  `<conversation>` (model output tags), the `<chartTau>`/`<parserBackend>`
  prose in `test_stage_3_parser_cleanup` -- none reach loaded config XML.

## Additional schema gaps found by the suite-wide sweep

Running the full suite under the gate surfaced three more genuine schema
gaps (the schema was stricter than the configs the code accepts):

* **`architecture` made optional** (`minOccurs="0"`) on the root `<model>`
  — it was required, but minimal fixtures (grammar-only configs, e.g.
  `test_grammar_start_rule`) omit it and inherit the whole block from the
  defaults overlay.
* **`languageType` allows repeated `<start>`** — multiple `<start>` tags
  define accepted unreduced forms / start patterns; switched `xs:all`
  $\to$ `xs:sequence` (`start*`, then `grammar?`, then `interpretation?`)
  since `xs:all` caps each child at one.
* **`symbolLearning` accepts an `enabled` attribute** — the code reads
  `architecture.symbolLearning.enabled` as an attribute
  (`<symbolLearning enabled="true"/>`); the schema only had a child
  `<enabled>` element. Both forms now validate.

Plus the `test_basicmodel.py` `_build_*` autoload-under-architecture
misplacement (x6, fixed to nest under `<training>`) and one factory
config's space dims moved out of `<architecture>` into `<InputSpace>`.

## Status / out of scope

* All `data/*.xml` validate; the gate is wired and fail-loud; build-all
  (`test_modality_configs`) green; the cleaned test files pass with zero
  schema-validation failures.
* `test_router_fires_per_word` (2 tests) fails with `n_calls=0` (per-word
  router did not fire) -- a router/grammar **logic** failure, NOT a
  schema-validation failure (0 schema failures for it). Unrelated to this
  schema work; triage with the prior grammar-path (`SS.sigma`) session.
* `MM_grammar.xml` convergence remains the known WIP from the prior
  session (not radix; not touched here).
