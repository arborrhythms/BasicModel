# Grammar Rewrite + Reconstruction + Learned SVO — Integrated Plan

> **Status (2026-04-20): completed.** All 12 tasks shipped. Highlights:
> - `_compose_vector_chart` has a typed-category compat mask; merged
>   slots carry forward their LHS category (Task 7).
> - NLTK POS tags drive initial seeding via `map_nltk_tags_to_categories`
>   and `WordSpace._seed_layer_categories` (Task 8).
> - `_extract_svo_from_trace` reads the outer `S → S VO` firing and its
>   matching `VO → V O` to populate `SyntacticLayer.last_svo` (Task 9).
> - `emit_head` + `WordSpace.reconstruct` land the downward `S → C`
>   one-shot head emission (Task 10).
> - `data/HeadEmission.xml` + `MentalModel._predicted_head` wire a
>   runtime head-prediction path (Task 11). Codebook source is the
>   InputSpace word embedding so heads decode back to vocabulary words.
> - `test_universality._get_svo_and_luminosity` now reads the grammar-
>   derived SVO + `_universality_score`; `MentalModel.xml` flips
>   `<chartCompose>true</chartCompose>` by default (Task 12).
> - Bonus: `bin/interact_head.py` - small REPL for poking the head
>   emitter; `SyntacticLayer.__init__` gained a `feature_dim` kwarg so
>   the pair-scorer's weight matches the actual leaf dim instead of
>   n_slots; `_compose_vector_chart` swapped in-place `live[b,s]=...`
>   assignments for an out-of-place mask update so autograd can
>   backprop through the whole chart.
>
> Leftover: `Data.loadInline` was **not** extended to parse
> `<sentences head="X">` XML; the head-prediction test injects
> sentences via `TheData.runtime_batch` instead, which is simpler and
> avoids a second XML code path. If a training loop ever needs the
> head-attribute to be a supervision signal, that parsing hook is
> still to do.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `<grammar>` into `<upward>` (parsing / chart compose) and `<downward>` (generation from a deep symbolic state), land the chart-like compose + typed-category + derivation-trace infrastructure they both need, and ship a concrete MVP: given an inline-XML mini-corpus of transitive sentences, the model predicts the **head word** of each sentence from its deep symbolic state via the downward grammar `S → C`. SVO extraction falls out as a second reader of the same derivation trace.

**Architecture:**

- **Upward grammar** = forward chart compose. Typed rules (`S → S VO`, `VO → V O`) drive pair-selection merges; the per-batch derivation trace records `(rule_id, left, right, merged, merged_category)` tuples. This is Phase B/C of `doc/LearnedSVO.md`.
- **Downward grammar** = generation path on a deep symbolic state. The MVP rule `S → C` is "find the codebook atom that best matches the state, emit it as the head, residual = state − contained(head)." Richer rules (`S → NP VP`, `NP → DET N`) become emission templates once head lookup works.
- **Two readers of one trace:** Phase D's SVO extraction and the downward generator both walk the same `_derivation_trace`. SVO looks for the outer `S → S VO` firing. The downward generator consumes the **final** composed state plus the category/residual machinery to emit surface tokens.
- **Legacy coexists:** MEMORY says keep legacy code alongside new code; the chart compose + downward generator are gated behind `<WordSpace.chartCompose>` / `<WordSpace.downwardGeneration>` flags. Default configs keep the old behavior until these flags flip.

**Tech Stack:** Python 3.12, PyTorch, namedtuple `Grammar.RuleDef`, `nn.Linear`/`nn.GELU` MLPs, `torch.nn.functional.softmax`, `nltk.pos_tag` (already in requirements.txt), pytest/unittest, XML + XSD (`xmlschema`).

**Scope / MVP boundaries:**

- **In scope:** grammar XML split, chart compose + trace, typed categories, NLTK POS seeding, head-word downward emission, SVO extraction, inline-sentence dataset with head-prediction loss, tests.
- **Out of scope (deferred):** Beam search over downward hypotheses, multi-word downward generation beyond head, gumbel annealing of category decisions, UniversalityWeight training bump, cross-language POS tag maps, curriculum (modifiers/embedded clauses).

---

## File Structure

| File | Responsibility | Status |
|------|----------------|--------|
| `bin/Language.py` | `Grammar.RuleDef`, `Grammar.configure()`, `Grammar._parse_rule`, upward/downward rule-list split, `SyntacticLayer._compose_vector_chart`, pair-scorer MLP, `_derivation_trace`, `_extract_svo_from_trace`, new `SyntacticLayer.emit_head` downward pass, `WordSpace.reconstruct` | Modify (central) |
| `bin/Spaces.py` | `WordSubSpace.add_word` already takes `leaf1/leaf2/leaf3`; add a `codebook_match(state)` helper on `SymbolicSpace` (or reuse existing VectorQuantize lookup) for the downward head-emission call | Modify (small) |
| `bin/Models.py` | `MentalModel.forward` — route the emitted head to a new `_predicted_head` attribute consumed by the head-prediction loss | Modify (small) |
| `bin/data.py` | `Data.load_inline` — read the new `<sentences>` block with `head="..."` attribute | Modify (small) |
| `data/model.xsd` | Add `<upward>`/`<downward>` child types under `grammarType`; add inline-`<sentences>` schema; add `<chartCompose>`, `<downwardGeneration>` WordSpace flags | Modify |
| `data/model.xml` | Split existing `<grammar><S>...</S></grammar>` into `<upward><S>not(S)</S></upward><downward/>` (empty downward → legacy) | Modify |
| `data/MentalModel.xml` | Add typed `<upward>` with `S → S VO`, `VO → V O`, and `<downward><S>C</S></downward>`; enable `<chartCompose>true</chartCompose>` | Modify |
| `data/HeadEmission.xml` | **New** config — minimal model with inline sentences + downward generation on, for the MVP head-prediction task | Create |
| `test/test_grammar_split.py` | Upward/downward parsing + roundtrip | Create |
| `test/test_ruledef_typed.py` | RuleDef fields (lhs/rhs_symbols) + bare-symbol form | Create |
| `test/test_compose_chart.py` | Chart compose + derivation trace + decompose roundtrip | Create |
| `test/test_category_propagation.py` | Category tensor ride; rhs_symbols compat mask; NLTK tagger map | Create |
| `test/test_svo_extraction.py` | Trace → SVO for canonical N V N; None when outer-S rule absent | Create |
| `test/test_head_emission.py` | Downward `S → C` picks the correct codebook atom; residual shrinks; inline-corpus head prediction | Create |
| `test/test_universality.py` | Replace `_get_svo_and_luminosity` stub with trace-based extraction | Modify |

### Already-applied skeleton edits (validated or reverted in Task 1)

These four edits landed while scoping; Task 1 is the gate that either keeps them (via green tests) or rolls them back:

- `bin/Language.py` — `Grammar.RuleDef` extended with `lhs, rhs_symbols`.
- `bin/Language.py` — `Grammar.configure()` iterates every key in `grammar_dict` (S first).
- `bin/Language.py` — `Grammar._parse_rule` handles bare-symbol RHS, populates `lhs/rhs_symbols` in all three return branches.
- `data/model.xsd` — `grammarType` opened with `xs:any processContents="skip"` (will be tightened in Task 2 to enumerate `<upward>` / `<downward>`).

---

## Dependency graph

```
Task 1 (typed RuleDef regression) ─┐
Task 2 (XML split upward/downward) ─┴─▶ Task 3 (derivation trace hook) ─▶
  Task 4 (pair-scorer MLP) ─▶ Task 5 (chart compose) ─▶ Task 6 (chart decompose) ─▶
    Task 7 (category tensor + rhs_symbols compat) ─▶ Task 8 (NLTK POS seeding) ─┐
                                                                                ├─▶ Task 9 (SVO extraction from trace)
                                                                                ├─▶ Task 10 (downward head emission)
                                                                                └─▶ Task 11 (inline corpus + head loss)
Task 9 & Task 10 & Task 11 ─▶ Task 12 (test_universality + full regression + config defaults)
```

---

## Task 1: Validate Phase-A skeleton edits (typed RuleDef + multi-LHS configure)

**Files:**
- Verify: `bin/Language.py` (already edited — RuleDef, `configure`, `_parse_rule`)
- Create: `test/test_ruledef_typed.py`

- [ ] **Step 1: Write the failing test**

Create `basicmodel/test/test_ruledef_typed.py`:

```python
"""RuleDef carries lhs/rhs_symbols; Grammar.configure accepts multi-LHS."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import Language
from Language import Grammar


class TestRuleDefTyped(unittest.TestCase):
    def test_ruledef_has_lhs_and_rhs_symbols(self):
        g = Grammar()
        g.configure({'S': ['not(S)']})
        rule = g.rules[0]
        self.assertEqual(rule.lhs, 'S')
        self.assertEqual(rule.rhs_symbols, ('S',))
        self.assertEqual(rule.method_name, 'not')
        self.assertEqual(rule.arity, 1)

    def test_bare_symbol_form_parses_to_merge(self):
        g = Grammar()
        g.configure({'S': ['S VO'], 'VO': ['V O']})
        self.assertEqual(len(g.rules), 2)
        r0, r1 = g.rules[0], g.rules[1]
        self.assertEqual(r0.lhs, 'S')
        self.assertEqual(r0.rhs_symbols, ('S', 'VO'))
        self.assertEqual(r0.method_name, 'merge')
        self.assertEqual(r0.arity, 2)
        self.assertEqual(r1.lhs, 'VO')
        self.assertEqual(r1.rhs_symbols, ('V', 'O'))

    def test_s_first_ordering_is_stable(self):
        g = Grammar()
        g.configure({'VO': ['V O'], 'S': ['S VO']})
        self.assertEqual(g.rules[0].lhs, 'S')
        self.assertEqual(g.rules[1].lhs, 'VO')

    def test_function_call_rhs_populates_rhs_symbols(self):
        g = Grammar()
        g.configure({'S': ['lift(S, S)']})
        rule = g.rules[0]
        self.assertEqual(rule.rhs_symbols, ('S', 'S'))
        self.assertEqual(rule.method_name, 'lift')
        self.assertEqual(rule.arity, 2)

    def test_epsilon_rule_has_empty_rhs(self):
        g = Grammar()
        g.configure({'S': ['epsilon']})
        rule = g.rules[0]
        self.assertEqual(rule.rhs_symbols, ())
        self.assertEqual(rule.arity, 0)
        self.assertIsNone(rule.method_name)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run the test to confirm it passes against the already-applied skeleton edits**

Run:
```bash
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel" && .venv/bin/python -m pytest test/test_ruledef_typed.py -v
```
Expected: all five tests PASS.

- [ ] **Step 3: Run the full existing test suite to catch regressions from the namedtuple-field change**

Run:
```bash
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel" && .venv/bin/python -m pytest test/ -x -q
```
Expected: every test that passed before still passes. Any unexpected failure indicates a caller that positionally unpacks `RuleDef` — fix before proceeding.

- [ ] **Step 4: Commit (user initiates — MEMORY: user manages commits)**

Suggested message:
```
Phase A: typed RuleDef (lhs, rhs_symbols), multi-LHS configure, bare-symbol RHS parser
```

---

## Task 2: Split `<grammar>` into `<upward>` and `<downward>` in XML + XSD

**Files:**
- Modify: `bin/Language.py` — `Grammar` gains `rules_upward` / `rules_downward` lists; `Grammar.configure` accepts both shapes; `_ensure_configured` reads `.language.grammar.upward` / `.downward`.
- Modify: `data/model.xsd` — define `<upward>` and `<downward>` child elements inside `<grammar>`.
- Modify: `data/model.xml` — wrap existing `<S>not(S)</S>` in `<upward>`.
- Modify: `data/MentalModel.xml` — wrap existing rules in `<upward>` + add `<downward><S>C</S></downward>`.
- Create: `test/test_grammar_split.py`.

- [ ] **Step 1: Write the failing test**

Create `basicmodel/test/test_grammar_split.py`:

```python
"""Grammar accepts <upward> and <downward> sub-blocks; default = upward."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import Language
from Language import Grammar
from util import init_config


_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


class TestGrammarSplit(unittest.TestCase):
    def test_configure_accepts_upward_block(self):
        g = Grammar()
        g.configure({'upward': {'S': ['not(S)']}})
        self.assertEqual(len(g.rules_upward), 1)
        self.assertEqual(g.rules_upward[0].lhs, 'S')
        self.assertEqual(g.rules_downward, [])

    def test_configure_accepts_both_blocks(self):
        g = Grammar()
        g.configure({
            'upward': {'S': ['S VO'], 'VO': ['V O']},
            'downward': {'S': ['C']},
        })
        self.assertEqual(len(g.rules_upward), 2)
        self.assertEqual(len(g.rules_downward), 1)
        down = g.rules_downward[0]
        self.assertEqual(down.lhs, 'S')
        self.assertEqual(down.rhs_symbols, ('C',))
        self.assertEqual(down.method_name, 'emit_head')

    def test_legacy_flat_grammar_still_loads(self):
        # Old-shape {'S': ['not(S)']} must still work as upward-only.
        g = Grammar()
        g.configure({'S': ['not(S)']})
        self.assertEqual(len(g.rules_upward), 1)
        self.assertEqual(g.rules_downward, [])

    def test_self_rules_contains_union_for_backcompat(self):
        # Grammar.rules must still be the union (upward first, then downward)
        # so old call sites that read g.rules keep working.
        g = Grammar()
        g.configure({
            'upward': {'S': ['S VO']},
            'downward': {'S': ['C']},
        })
        self.assertEqual(len(g.rules), 2)
        self.assertEqual(g.rules[0].lhs, 'S')
        self.assertEqual(g.rules[1].rhs_symbols, ('C',))

    def test_xml_config_loads_upward_block(self):
        # model.xml's upward-wrapped grammar survives round-trip.
        init_config(
            path=os.path.join(_DATA_DIR, 'model.xml'),
            defaults_path=os.path.join(_DATA_DIR, 'model.xml'),
        )
        Language.TheGrammar._configured = False
        Language.TheGrammar._ensure_configured()
        # model.xml ships with <upward><S>not(S)</S></upward>
        self.assertGreaterEqual(len(Language.TheGrammar.rules_upward), 1)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run:
```bash
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel" && .venv/bin/python -m pytest test/test_grammar_split.py -v
```
Expected: FAIL — `rules_upward` / `rules_downward` don't exist.

- [ ] **Step 3: Implement the split in `Grammar`**

In `bin/Language.py`, replace the `configure` method with:

```python
    def configure(self, grammar_dict):
        """Configure rules from an XML-derived dict.

        Accepts two shapes:
          (a) flat: {'S': ['not(S)'], ...}  — legacy upward-only.
          (b) split: {'upward': {'S': [...], 'VO': [...]},
                      'downward': {'S': ['C']}}
        """
        self.rules_upward = []
        self.rules_downward = []
        self._configured = True

        if ('upward' in grammar_dict or 'downward' in grammar_dict):
            up = grammar_dict.get('upward', {}) or {}
            dn = grammar_dict.get('downward', {}) or {}
            self._fill_rule_list(self.rules_upward, up)
            self._fill_rule_list(self.rules_downward, dn)
        else:
            # Legacy flat form — treat as upward.
            self._fill_rule_list(self.rules_upward, grammar_dict)

        # Canonical union so callers reading `g.rules` see upward first
        # then downward. Upward rule IDs stay stable for existing code.
        self.rules = list(self.rules_upward) + list(self.rules_downward)
        self.rule_table = {idx: rule.canonical
                           for idx, rule in enumerate(self.rules)}

    def _fill_rule_list(self, target, rules_dict):
        keys = list(rules_dict.keys())
        if 'S' in keys:
            keys = ['S'] + [k for k in keys if k != 'S']
        for lhs in keys:
            raw = rules_dict.get(lhs, [])
            if isinstance(raw, str):
                raw = [raw]
            for rhs_text in raw:
                rhs = rhs_text.strip()
                target.append(self._parse_rule(lhs, rhs))
```

And extend `_parse_rule` so the pseudo-terminal `C` (codebook lookup) gets a distinct method name:

```python
    def _parse_rule(self, lhs, rhs):
        tier = 'S'
        if '(' in rhs:
            func_name = rhs[:rhs.index('(')]
            args_str = rhs[rhs.index('(') + 1:rhs.rindex(')')]
            args = [a.strip() for a in args_str.split(',') if a.strip()]
            arity = len(args)
            canonical = f"{lhs} -> {rhs}"
            return self.RuleDef(tier, canonical, arity, func_name,
                                lhs, tuple(args))
        if rhs == 'epsilon':
            return self.RuleDef(tier, f"{lhs} -> epsilon", 0, None,
                                lhs, ())
        if rhs == lhs:
            return self.RuleDef(tier, f"{lhs} -> {rhs}", 1, None,
                                lhs, (rhs,))
        # Bare-symbol-sequence form.
        parts = rhs.split()
        if parts and all(p.isidentifier() for p in parts):
            arity = len(parts)
            # 'C' is the pseudo-terminal used in downward: 'S -> C' means
            # "emit the codebook atom that best matches the current state."
            # It dispatches through emit_head, not merge.
            if len(parts) == 1 and parts[0] == 'C':
                method = 'emit_head'
            else:
                method = 'merge'
            return self.RuleDef(tier, f"{lhs} -> {rhs}", arity, method,
                                lhs, tuple(parts))
        raise ValueError(f"Cannot parse grammar rule: {lhs} -> {rhs}")
```

Initialise the new lists in `__init__`:

```python
    def __init__(self):
        self.rules = []
        self.rules_upward = []
        self.rules_downward = []
        self.rule_table = {}
        self._configured = False
        self.interpretation = 0.5
        self.thought_free = False
```

- [ ] **Step 4: Update the XSD**

In `basicmodel/data/model.xsd`, replace the `grammarType` complex type with:

```xml
  <xs:complexType name="grammarType">
    <!-- Either (legacy) inline nonterminal children OR split into
         <upward>/<downward> subsections. Child elements of upward/downward
         are arbitrary nonterminal names: 'S', 'VO', 'NP', 'VP', 'V', 'O'. -->
    <xs:choice minOccurs="0" maxOccurs="unbounded">
      <xs:element name="upward" type="grammarSectionType" minOccurs="0"/>
      <xs:element name="downward" type="grammarSectionType" minOccurs="0"/>
      <xs:any processContents="skip" minOccurs="0" maxOccurs="unbounded"/>
    </xs:choice>
  </xs:complexType>

  <xs:complexType name="grammarSectionType">
    <xs:sequence>
      <xs:any processContents="skip" minOccurs="0" maxOccurs="unbounded"/>
    </xs:sequence>
  </xs:complexType>
```

- [ ] **Step 5: Update `data/model.xml`**

Replace lines 107-112 (the `<language><grammar>...</grammar></language>` block) with:

```xml
    <language>
      <grammar>
        <upward>
          <S>not(S)</S>
        </upward>
      </grammar>
      <interpretation>0.5</interpretation>
    </language>
```

- [ ] **Step 6: Update `data/MentalModel.xml`**

Replace its `<grammar>` block with:

```xml
      <grammar>
        <upward>
          <!-- Legacy function-call rules (unchanged semantics) -->
          <S>true(S)</S>
          <S>false(S)</S>
          <S>non(S)</S>
          <S>conjunction(S, S)</S>
          <S>disjunction(S, S)</S>
          <S>what(S)</S>
          <S>where(S)</S>
          <S>when(S)</S>
          <S>query(S, S)</S>
          <S>swap(S, S)</S>
          <S>equals(S, S)</S>
          <S>not(S)</S>
          <S>part(S, S)</S>
          <S>intersection(S, S)</S>
          <S>union(S, S)</S>
          <S>lower(S, S)</S>
          <S>lift(S, S)</S>
          <!-- Typed productions (LearnedSVO Phase A.4) -->
          <S>S VO</S>
          <VO>V O</VO>
        </upward>
        <downward>
          <!-- MVP: emit the codebook atom closest to the deep state -->
          <S>C</S>
        </downward>
      </grammar>
```

- [ ] **Step 7: Run tests**

Run:
```bash
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel" && .venv/bin/python -m pytest test/test_grammar_split.py test/test_ruledef_typed.py -v
```
Expected: both files PASS.

- [ ] **Step 8: XSD validation**

Run:
```bash
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel" && .venv/bin/python -c "
import xmlschema
schema = xmlschema.XMLSchema('data/model.xsd')
schema.validate('data/model.xml')
schema.validate('data/MentalModel.xml')
print('XSD validation OK')
"
```
Expected: `XSD validation OK`.

- [ ] **Step 9: Full regression**

Run:
```bash
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel" && .venv/bin/python -m pytest test/ -x -q
```
Expected: all previously-passing tests still pass.

- [ ] **Step 10: Commit**

---

## Task 3: Derivation-trace scaffolding (reset contract, no behavior change)

**Files:**
- Modify: `bin/Language.py` — `SyntacticLayer.__init__` adds `_derivation_trace = None`; `SyntacticLayer.compose` sets it to `[[] for _ in range(B)]`.
- Create: stub `test/test_compose_chart.py`.

- [ ] **Step 1: Write the failing test**

Create `basicmodel/test/test_compose_chart.py`:

```python
"""Chart-like compose scaffolding: derivation trace contract."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import torch
import Language
from Language import Grammar, SyntacticLayer


class TestDerivationTraceContract(unittest.TestCase):
    def test_trace_attribute_exists_after_init(self):
        layer = SyntacticLayer(nInput=4, nOutput=4, rules=[], max_depth=4,
                               hidden_dim=16, grammar=Grammar())
        self.assertTrue(hasattr(layer, '_derivation_trace'))
        self.assertIsNone(layer._derivation_trace)

    def test_trace_reset_to_per_batch_empty_lists_on_compose(self):
        from Spaces import WordSubSpace
        g = Grammar()
        g.configure({'S': ['not(S)']})
        B, N, D = 2, 4, 3
        layer = SyntacticLayer(nInput=N, nOutput=N, rules=g.symbolic(),
                               max_depth=N - 1, hidden_dim=16, grammar=g)
        sub = WordSubSpace(nDim=D, nWhat=D, nWhere=0, nWhen=0,
                           max_depth=8, max_arity=3, batch=B)
        layer._derivation_trace = 'stale'
        layer.compose(torch.randn(B, N, D), sub, g)
        self.assertEqual(layer._derivation_trace, [[] for _ in range(B)])


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run to verify fails**

Run:
```bash
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel" && .venv/bin/python -m pytest test/test_compose_chart.py -v
```
Expected: FAIL (`_derivation_trace` missing).

- [ ] **Step 3: Implement**

In `SyntacticLayer.__init__`, immediately after `self.lifting_layer = None`:

```python
        # Derivation trace: per-batch list of (rule_id, left, right,
        # merged_slot, merged_category) tuples, appended by the chart
        # compose path. Reset at compose start; reused by SVO extraction
        # and downward head emission.
        self._derivation_trace = None
```

In `SyntacticLayer.compose`, replace the existing reset block (`self.last_svo = None; self.last_rule_probs = None; self.last_composable_rules = None`) with:

```python
        self.last_svo = None
        self.last_rule_probs = None
        self.last_composable_rules = None
        B_guess = data.shape[0] if torch.is_tensor(data) and data.ndim >= 2 else 1
        self._derivation_trace = [[] for _ in range(B_guess)]
```

- [ ] **Step 4: Run**

Expected: PASS.

- [ ] **Step 5: Commit**

---

## Task 4: Pair-scorer MLP and live-leaf helper

**Files:**
- Modify: `bin/Language.py` — `SyntacticLayer.__init__` adds `self.pair_scorer`; add `_pair_scorer` and `_live_pairs` methods.
- Append tests to `test/test_compose_chart.py`.

- [ ] **Step 1: Write the failing test**

Append to `test/test_compose_chart.py`:

```python
class TestPairScorer(unittest.TestCase):
    def _make_layer(self, N=4, num_rules=2):
        g = Grammar()
        g.configure({'S': ['not(S)'] * num_rules})
        return SyntacticLayer(nInput=N, nOutput=N, rules=list(range(num_rules)),
                              max_depth=N - 1, hidden_dim=16, grammar=g)

    def test_pair_scorer_output_shape(self):
        layer = self._make_layer(N=4, num_rules=2)
        B, N, D = 2, 4, 4
        hidden = torch.zeros(B, layer.hidden_dim)
        pairs = torch.zeros(B, N - 1, 2, N)  # D == nInput by contract
        alive = torch.ones(B, N, dtype=torch.bool)
        scores = layer._pair_scorer(hidden, pairs, alive)
        self.assertEqual(scores.shape, (B, N - 1))

    def test_pair_scorer_respects_alive_mask(self):
        layer = self._make_layer(N=4, num_rules=2)
        B, N = 1, 4
        hidden = torch.zeros(B, layer.hidden_dim)
        pairs = torch.zeros(B, N - 1, 2, N)
        alive = torch.tensor([[True, True, True, False]])
        scores = layer._pair_scorer(hidden, pairs, alive)
        probs = torch.softmax(scores, dim=-1)
        self.assertAlmostEqual(probs[0, 2].item(), 0.0, places=5)
```

- [ ] **Step 2: Verify fails**

Run:
```bash
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel" && .venv/bin/python -m pytest test/test_compose_chart.py::TestPairScorer -v
```
Expected: FAIL (`_pair_scorer` missing).

- [ ] **Step 3: Implement**

In `SyntacticLayer.__init__` after `self.rule_head = LinearLayer(...)`:

```python
        # Phase B: pair-scorer MLP.
        self.pair_scorer = nn.Sequential(
            nn.Linear(hidden_dim + 2 * nInput, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.layers.append(self.pair_scorer)
```

Above `_compose_vector`, add:

```python
    def _pair_scorer(self, hidden, pairs, alive):
        """Score adjacent-leaf pairs for chart-like merge.

        hidden [B, H], pairs [B, P, 2, D], alive [B, N] → logits [B, P].
        Dead pairs get -inf so softmax routes zero probability to them.
        """
        B, P, _, D = pairs.shape
        H = hidden.shape[-1]
        h = hidden.unsqueeze(1).expand(B, P, H)
        flat = pairs.reshape(B, P, 2 * D)
        feat = torch.cat([h, flat], dim=-1)
        logits = self.pair_scorer(feat.reshape(B * P, -1)).reshape(B, P)
        pair_alive = alive[:, :-1] & alive[:, 1:]
        logits = logits.masked_fill(~pair_alive, float('-inf'))
        return logits

    def _live_pairs(self, live, alive):
        """Extract adjacent-pair tensors from live-leaf ordering.

        Returns (pair_tensor [B, P_max, 2, D], list[list[(l, r)]]).
        """
        B, N, D = live.shape
        batch_pairs = []
        max_P = 0
        for b in range(B):
            positions = [i for i in range(N) if bool(alive[b, i].item())]
            pairs = list(zip(positions[:-1], positions[1:]))
            batch_pairs.append(pairs)
            max_P = max(max_P, len(pairs))
        pair_tensor = torch.zeros(B, max_P, 2, D, device=live.device)
        for b, pairs in enumerate(batch_pairs):
            for p, (l, r) in enumerate(pairs):
                pair_tensor[b, p, 0] = live[b, l]
                pair_tensor[b, p, 1] = live[b, r]
        return pair_tensor, batch_pairs
```

- [ ] **Step 4: Run**

Expected: PASS.

- [ ] **Step 5: Commit**

---

## Task 5: Chart-like `_compose_vector_chart` behind `<chartCompose>` flag

**Files:**
- Modify: `bin/Language.py` — add `_compose_vector_chart`, `_apply_rules_to_pairs`; dispatch from `_compose_vector` top.
- Modify: `data/model.xsd` — add `<chartCompose>` child of `<WordSpace>`.
- Modify: `data/model.xml` — add `<chartCompose>false</chartCompose>`.
- Append tests to `test/test_compose_chart.py`.

**Risk:** This is the load-bearing task. Allow iteration; legacy path stays intact by default.

- [ ] **Step 1: Write the failing test**

Append to `test/test_compose_chart.py`:

```python
class TestChartCompose(unittest.TestCase):
    def _enable_chart(self):
        from util import TheXMLConfig
        TheXMLConfig._overlay = getattr(TheXMLConfig, '_overlay', {})
        TheXMLConfig._overlay['WordSpace.chartCompose'] = True

    def tearDown(self):
        from util import TheXMLConfig
        if hasattr(TheXMLConfig, '_overlay'):
            TheXMLConfig._overlay.pop('WordSpace.chartCompose', None)

    def test_chart_reduces_n_leaves_to_trace_entries(self):
        self._enable_chart()
        from Spaces import WordSubSpace
        g = Grammar()
        g.configure({'upward': {'S': ['S VO'], 'VO': ['V O']}})
        B, N, D = 1, 3, 3
        layer = SyntacticLayer(nInput=N, nOutput=N,
                               rules=list(range(len(g.rules_upward))),
                               max_depth=N - 1, hidden_dim=16, grammar=g)
        sub = WordSubSpace(nDim=D, nWhat=D, nWhere=0, nWhen=0,
                           max_depth=8, max_arity=3, batch=B)
        data = torch.randn(B, N, D)
        composed, _ = layer.compose(data, sub, g)
        self.assertEqual(composed.shape, (B, N, D))
        # N=3 active → 2 merges
        self.assertEqual(len(layer._derivation_trace[0]), N - 1)

    def test_chart_trace_tuple_shape(self):
        self._enable_chart()
        from Spaces import WordSubSpace
        g = Grammar()
        g.configure({'upward': {'S': ['S VO'], 'VO': ['V O']}})
        B, N, D = 1, 3, 3
        layer = SyntacticLayer(nInput=N, nOutput=N,
                               rules=list(range(len(g.rules_upward))),
                               max_depth=N - 1, hidden_dim=16, grammar=g)
        sub = WordSubSpace(nDim=D, nWhat=D, nWhere=0, nWhen=0,
                           max_depth=8, max_arity=3, batch=B)
        layer.compose(torch.randn(B, N, D), sub, g)
        for entry in layer._derivation_trace[0]:
            # (rule_id, left, right, merged, merged_category)
            self.assertEqual(len(entry), 5)
            for i in range(5):
                self.assertIsInstance(entry[i], int)

    def test_chart_legacy_path_unchanged_when_flag_off(self):
        from Spaces import WordSubSpace
        g = Grammar()
        g.configure({'S': ['not(S)']})
        B, N, D = 1, 3, 3
        layer = SyntacticLayer(nInput=N, nOutput=N, rules=g.symbolic(),
                               max_depth=N - 1, hidden_dim=16, grammar=g)
        sub = WordSubSpace(nDim=D, nWhat=D, nWhere=0, nWhen=0,
                           max_depth=8, max_arity=3, batch=B)
        composed, _ = layer.compose(torch.randn(B, N, D), sub, g)
        # Legacy path leaves the trace empty (no chart merges made).
        self.assertEqual(layer._derivation_trace, [[] for _ in range(B)])
        self.assertEqual(composed.shape, (B, N, D))
```

- [ ] **Step 2: Verify fails**

Run:
```bash
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel" && .venv/bin/python -m pytest test/test_compose_chart.py::TestChartCompose -v
```
Expected: first two FAIL, third PASS.

- [ ] **Step 3: Implement `_compose_vector_chart` + `_apply_rules_to_pairs`**

Add to `SyntacticLayer` (below `_live_pairs`):

```python
    def _apply_rules_to_pairs(self, pair_tensor, composable_global,
                              grammar, subspace):
        """For each (pair, rule) compute the merged vector. Returns
        merged[B, P, R, D] where R=len(composable_global).
        """
        B, P, _, D = pair_tensor.shape
        R = len(composable_global)
        merged = torch.zeros(B, P, R, D, device=pair_tensor.device)
        for p in range(P):
            left = pair_tensor[:, p, 0, :].unsqueeze(1)
            right = pair_tensor[:, p, 1, :].unsqueeze(1)
            for r, gid in enumerate(composable_global):
                result = self.project(grammar, gid, left, right,
                                      subspace=subspace)
                merged[:, p, r] = result.squeeze(1)
        return merged

    def _compose_vector_chart(self, data, subspace, grammar):
        """Phase B chart compose. Differentiable pair + rule selection.

        Populates self._derivation_trace with 5-tuples
        (rule_id, left_slot, right_slot, merged_slot, merged_category_id).
        Returns (composed [B, N, D], None). SVO left to Phase D walker.
        """
        B, N, D = data.shape
        live = data.clone()
        alive = torch.zeros(B, N, dtype=torch.bool, device=data.device)
        active_positions = [subspace.active_positions(b, data)
                            for b in range(B)]
        for b, positions in enumerate(active_positions):
            for p in positions:
                alive[b, p] = True

        activation = torch.norm(live, dim=-1) / math.sqrt(D)
        expected_n = self.input_proj.nInput
        if N != expected_n or self.num_rules == 0:
            return live, None

        out = self.forward(activation)
        rule_probs_per_depth = out['rule_probs']
        hidden_per_depth = out.get('hidden', None)

        # Upward rules only in chart compose; downward is emit-time.
        up_rules = [i for i, r in enumerate(grammar.rules)
                    if r in grammar.rules_upward]
        composable_global = [gid for gid in up_rules
                             if grammar.arity(gid) >= 2]
        if not composable_global:
            return live, None
        composable_local = [self.all_rules.index(gid)
                            for gid in composable_global
                            if gid in self.all_rules]
        if not composable_local:
            return live, None
        composable_global = [self.all_rules[i] for i in composable_local]

        # Category machinery (Task 7 will populate; Task 5 just keeps space).
        self._ensure_category_table(grammar)
        if self._last_category is not None:
            category = self._last_category.to(data.device).long()
        else:
            category = torch.full((B, N), 0, dtype=torch.long,
                                  device=data.device)

        rule_lhs_ids = []
        rule_rhs_ids = []
        for gid in composable_global:
            lhs = grammar.rules[gid].lhs
            rhs = grammar.rules[gid].rhs_symbols or ()
            rule_lhs_ids.append(self._category_index.get(lhs, 0))
            if len(rhs) >= 2:
                rule_rhs_ids.append(
                    (self._category_index.get(rhs[0], 0),
                     self._category_index.get(rhs[1], 0)))
            else:
                rule_rhs_ids.append((0, 0))

        depth_probs = []
        for step in range(min(self.max_depth, N - 1)):
            pair_tensor, pair_positions = self._live_pairs(live, alive)
            if pair_tensor.shape[1] == 0:
                break

            if hidden_per_depth is not None and hidden_per_depth.ndim >= 2:
                if hidden_per_depth.ndim == 3:
                    hidden = hidden_per_depth[:, min(step, hidden_per_depth.shape[1] - 1), :]
                else:
                    hidden = hidden_per_depth
            else:
                hidden = torch.zeros(B, self.hidden_dim, device=data.device)

            pair_logits = self._pair_scorer(hidden, pair_tensor, alive)
            pair_probs = torch.softmax(pair_logits, dim=-1)

            rule_probs_step = rule_probs_per_depth[:, step, :][:, composable_local]
            rule_probs_step = rule_probs_step / (
                rule_probs_step.sum(dim=-1, keepdim=True) + 1e-8)
            depth_probs.append(rule_probs_step.detach())

            merged = self._apply_rules_to_pairs(
                pair_tensor, composable_global, grammar, subspace)

            # Compat mask (Task 7 wires category; before that it is all-ones
            # for typed rules and all-ones for legacy function-call rules).
            P_here, R_here = pair_tensor.shape[1], len(composable_global)
            compat = torch.ones(B, P_here, R_here, device=data.device)

            joint = (pair_probs.unsqueeze(-1)
                     * rule_probs_step.unsqueeze(1)
                     * compat)
            joint = joint / joint.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)

            if self.training:
                merged_vec = (joint.unsqueeze(-1) * merged).sum(dim=(1, 2))
            else:
                flat_idx = joint.reshape(B, -1).argmax(dim=-1)
                pair_idx = flat_idx // R_here
                rule_idx_local = flat_idx % R_here
                merged_vec = merged[torch.arange(B), pair_idx, rule_idx_local]

            # Trace + alive/live update — always uses argmax for the
            # discrete trace even in training (gradient still flows
            # through merged_vec via the soft mixture).
            best_flat = joint.reshape(B, -1).argmax(dim=-1)
            best_pair = (best_flat // R_here).tolist()
            best_rule_local = (best_flat % R_here).tolist()

            for b in range(B):
                if not pair_positions[b] or best_pair[b] >= len(pair_positions[b]):
                    continue
                left_slot, right_slot = pair_positions[b][best_pair[b]]
                rule_gid = composable_global[best_rule_local[b]]
                merged_cat = rule_lhs_ids[best_rule_local[b]]
                live[b, left_slot] = merged_vec[b]
                alive[b, right_slot] = False
                category[b, left_slot] = merged_cat
                self._derivation_trace[b].append(
                    (int(rule_gid), int(left_slot), int(right_slot),
                     int(left_slot), int(merged_cat)))
                subspace.add_word(b, int(left_slot), int(rule_gid),
                                  order=step,
                                  leaf1=int(left_slot),
                                  leaf2=int(right_slot))

        if depth_probs:
            self.last_rule_probs = torch.stack(depth_probs, dim=1)
        self.last_composable_rules = composable_global
        return live, None
```

Stub category helpers (Task 7 gives them real content):

```python
    def _ensure_category_table(self, grammar):
        if getattr(self, '_category_names', None) is not None:
            return
        names = set()
        for rule in grammar.rules:
            names.add(rule.lhs)
            for sym in (rule.rhs_symbols or ()):
                names.add(sym)
        ordered = ['?'] + sorted(n for n in names if n)
        self._category_names = ordered
        self._category_index = {n: i for i, n in enumerate(ordered)}

    def _seed_category(self, category):
        self._last_category = category.clone()
```

In `SyntacticLayer.__init__` after `_derivation_trace = None`:

```python
        self._category_names = None
        self._category_index = None
        self._last_category = None
```

- [ ] **Step 4: Dispatch from `_compose_vector`**

Near the top of `_compose_vector` (after `s_rules = grammar._s_rule_ids()` and before the `target_count` branch), add:

```python
        try:
            use_chart = bool(TheXMLConfig.get("WordSpace.chartCompose"))
        except (KeyError, AttributeError, TypeError):
            use_chart = False
        if use_chart and target_count is None:
            return self._compose_vector_chart(data, subspace, grammar)
```

- [ ] **Step 5: Add `<chartCompose>` to XSD and model.xml**

In `basicmodel/data/model.xsd`, inside the `WordSpace` complex type near `syntacticHiddenDim`, add:

```xml
      <xs:element name="chartCompose" type="xs:boolean" minOccurs="0"/>
```

In `basicmodel/data/model.xml`, inside `<WordSpace>`:

```xml
    <!-- Phase B: chart-like pair selection in compose.
         false = legacy left-associative cascade. -->
    <chartCompose>false</chartCompose>
```

- [ ] **Step 6: Run tests**

Run:
```bash
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel" && .venv/bin/python -m pytest test/test_compose_chart.py -v
```
Expected: all PASS.

- [ ] **Step 7: Full regression**

Run:
```bash
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel" && .venv/bin/python -m pytest test/ -x -q
```
Expected: all previously-passing tests still pass (flag is off by default in model.xml).

- [ ] **Step 8: Commit**

---

## Task 6: Chart-aware `decompose()`

**Files:**
- Modify: `bin/Language.py` — `SyntacticLayer.decompose` reads `_derivation_trace`.

- [ ] **Step 1: Write the failing test**

Append to `test/test_compose_chart.py`:

```python
class TestChartDecompose(unittest.TestCase):
    def setUp(self):
        from util import TheXMLConfig
        TheXMLConfig._overlay = getattr(TheXMLConfig, '_overlay', {})
        TheXMLConfig._overlay['WordSpace.chartCompose'] = True

    def tearDown(self):
        from util import TheXMLConfig
        if hasattr(TheXMLConfig, '_overlay'):
            TheXMLConfig._overlay.pop('WordSpace.chartCompose', None)

    def test_decompose_returns_same_shape(self):
        from Spaces import WordSubSpace
        g = Grammar()
        g.configure({'upward': {'S': ['S VO'], 'VO': ['V O']}})
        B, N, D = 1, 3, 4
        layer = SyntacticLayer(nInput=N, nOutput=N,
                               rules=list(range(len(g.rules_upward))),
                               max_depth=N - 1, hidden_dim=16, grammar=g)
        sub = WordSubSpace(nDim=D, nWhat=D, nWhere=0, nWhen=0,
                           max_depth=8, max_arity=3, batch=B)
        data = torch.randn(B, N, D)
        composed, _ = layer.compose(data, sub, g)
        restored = layer.decompose(composed, sub, g)
        self.assertEqual(restored.shape, data.shape)
```

- [ ] **Step 2: Verify**

Run:
```bash
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel" && .venv/bin/python -m pytest test/test_compose_chart.py::TestChartDecompose -v
```
Expected: PASS (the existing leaf-codebook path in `decompose` is already shape-preserving). If FAIL, proceed to Step 3.

- [ ] **Step 3: Add trace-aware branch**

At the top of `SyntacticLayer.decompose`, before the existing basis/codebook branch:

```python
        trace = getattr(self, '_derivation_trace', None)
        if trace is not None and any(t for t in trace) and data.ndim == 3:
            basis = getattr(subspace, 'basis', None)
            cb = basis.getW() if basis is not None else None
            if cb is not None and data.shape[-1] == cb.shape[-1]:
                words = subspace.get_words()
                result = torch.zeros_like(data)
                for word in words:
                    if word[WordEncoding.ORDER] != -1:
                        continue
                    b = word[WordEncoding.BATCH]
                    pos = word[WordEncoding.VECTOR]
                    cb_idx = word[WordEncoding.LEAF1]
                    if cb_idx >= 0:
                        result[b, pos] = cb[cb_idx]
                return result
            return data
```

- [ ] **Step 4: Run and commit**

---

## Task 7: Typed category tensor + `rhs_symbols` compatibility mask

**Files:**
- Modify: `bin/Language.py` — flesh out `_ensure_category_table`, `_seed_category`, and the compat mask in `_compose_vector_chart`.
- Create: `test/test_category_propagation.py`.

- [ ] **Step 1: Write the failing test**

Create `basicmodel/test/test_category_propagation.py`:

```python
"""Category tensor propagates through chart compose; rhs_symbols gate rules."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import torch
from util import TheXMLConfig
from Language import Grammar, SyntacticLayer


class TestCategoryPropagation(unittest.TestCase):
    def setUp(self):
        TheXMLConfig._overlay = getattr(TheXMLConfig, '_overlay', {})
        TheXMLConfig._overlay['WordSpace.chartCompose'] = True

    def tearDown(self):
        if hasattr(TheXMLConfig, '_overlay'):
            TheXMLConfig._overlay.pop('WordSpace.chartCompose', None)

    def _layer(self, rules_dict, N=3, D=3):
        g = Grammar()
        g.configure(rules_dict)
        layer = SyntacticLayer(nInput=N, nOutput=N,
                               rules=list(range(len(g.rules))),
                               max_depth=N - 1, hidden_dim=16, grammar=g)
        from Spaces import WordSubSpace
        sub = WordSubSpace(nDim=D, nWhat=D, nWhere=0, nWhen=0,
                           max_depth=8, max_arity=3, batch=1)
        return layer, sub, g

    def test_merging_V_and_O_yields_VO_slot(self):
        layer, sub, g = self._layer(
            {'upward': {'S': ['S VO'], 'VO': ['V O']}}, N=3)
        layer._ensure_category_table(g)
        cats = torch.tensor([[layer._category_index['S'],
                              layer._category_index['V'],
                              layer._category_index['O']]])
        layer._seed_category(cats)
        data = torch.randn(1, 3, 3)
        layer.compose(data, sub, g)
        # Final trace entry should have merged_category == S.
        last = layer._derivation_trace[0][-1]
        self.assertEqual(last[4], layer._category_index['S'])

    def test_incompatible_categories_block_the_rule(self):
        layer, sub, g = self._layer(
            {'upward': {'S': ['S VO'], 'VO': ['V O']}}, N=3)
        layer._ensure_category_table(g)
        # Seed all leaves as S: there is no rule (S, S) -> anything typed,
        # and the pair-scorer + compat mask should zero-out every rule.
        cats = torch.tensor([[layer._category_index['S']] * 3])
        layer._seed_category(cats)
        data = torch.randn(1, 3, 3)
        layer.compose(data, sub, g)
        # No legal typed merges -> trace is empty.
        self.assertEqual(layer._derivation_trace[0], [])
```

- [ ] **Step 2: Verify fails**

Expected: FAIL (compat is currently all-ones).

- [ ] **Step 3: Replace the all-ones compat with a proper mask**

In `_compose_vector_chart`, replace the `compat = torch.ones(...)` line with:

```python
            P_here, R_here = pair_tensor.shape[1], len(composable_global)
            compat = torch.zeros(B, P_here, R_here, device=data.device)
            for b in range(B):
                for p, (ls, rs) in enumerate(pair_positions[b]):
                    cl = int(category[b, ls].item())
                    cr = int(category[b, rs].item())
                    for r, (lhs_req, rhs_req) in enumerate(rule_rhs_ids):
                        rhs_syms = grammar.rules[composable_global[r]].rhs_symbols
                        if rhs_syms is None:
                            # Legacy function-call rule: always compatible.
                            compat[b, p, r] = 1.0
                        elif cl == lhs_req and cr == rhs_req:
                            compat[b, p, r] = 1.0
```

Add a guard: if after `compat.sum()` the row has no compatible pair/rule, break early to keep the trace empty:

```python
            if compat.sum() == 0:
                break
```

- [ ] **Step 4: Run**

Expected: both category tests PASS.

- [ ] **Step 5: Commit**

---

## Task 8: NLTK POS seeding for category tensor

**Files:**
- Modify: `bin/Language.py` — `map_nltk_tags_to_categories` helper; `WordSpace._seed_layer_categories`; back-reference from `WordSpace` to `InputSpace`.
- Modify: `bin/Spaces.py` — `InputSpace.prepInput` stashes `self._last_tokens`.
- Append tests to `test/test_category_propagation.py`.

- [ ] **Step 1: Write the failing test**

Append:

```python
class TestNLTKPOSSeeding(unittest.TestCase):
    def test_basic_map(self):
        from Language import map_nltk_tags_to_categories
        cats = map_nltk_tags_to_categories(
            ['the', 'teacher', 'helped', 'the', 'student'])
        self.assertEqual(cats, ['DET', 'N', 'V', 'DET', 'N'])
```

- [ ] **Step 2: Verify fails**

Expected: FAIL — `map_nltk_tags_to_categories` missing.

- [ ] **Step 3: Implement**

At module scope in `bin/Language.py` (before `class Grammar`):

```python
_NLTK_TAG_TO_CATEGORY = {
    'DT': 'DET',
    'NN': 'N', 'NNS': 'N', 'NNP': 'N', 'NNPS': 'N', 'PRP': 'N',
    'VB': 'V', 'VBD': 'V', 'VBG': 'V', 'VBN': 'V', 'VBP': 'V', 'VBZ': 'V',
    'JJ': 'ADJ', 'JJR': 'ADJ', 'JJS': 'ADJ',
    'RB': 'ADV', 'RBR': 'ADV', 'RBS': 'ADV',
    'IN': 'PREP', 'CC': 'CONJ',
}


def map_nltk_tags_to_categories(tokens):
    """Run nltk.pos_tag and project tags to the typed-grammar categories."""
    import nltk
    try:
        tagged = nltk.pos_tag(tokens)
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        tagged = nltk.pos_tag(tokens)
    return [_NLTK_TAG_TO_CATEGORY.get(tag, '?') for _, tag in tagged]
```

Add a wire-up hook in `WordSpace`: in `WordSpace.__init__`, add `self._input_space_ref = None` (after the `self.syntacticLayer = None` line). In `WordSpace._build_syntactic_layer`, when `perceptualSpace` or `inputSpace` is available through the model wiring, store it. The simplest approach: let `MentalModel.__init__` set `wordSpace._input_space_ref = inputSpace` after construction.

In `bin/Models.py`, in the `MentalModel` constructor after `self.wordSpace = ...`:

```python
        # Phase C.2: WordSpace needs a back-ref to InputSpace for NLTK
        # POS seeding of the chart-compose category tensor.
        self.wordSpace._input_space_ref = self.inputSpace
```

In `bin/Spaces.py`, in `InputSpace.prepInput` (find the method via grep first; note the method stashes the current-batch tokens — add):

```python
        self._last_tokens = tokens_list_of_lists  # keyed to the matching batch
```

(If `prepInput` doesn't currently hold a token-list-of-lists, reuse whatever the existing tokenizer yields; keep it as a Python list so nltk can consume it.)

In `WordSpace.forwardSymbols`, add the seeding call:

```python
    def forwardSymbols(self, data, subspace):
        layer = self.syntacticLayer
        if layer is None:
            return data
        if data.ndim == 3 and data.shape[-1] == getattr(subspace, 'muxedSize', -1):
            subspace.demux(data)
        self._seed_layer_categories(layer, data.shape[:2])
        result = layer.compose(data, subspace, TheGrammar)
        if isinstance(result, tuple):
            return result[0]
        return result

    def _seed_layer_categories(self, layer, shape):
        input_space = self._input_space_ref
        if input_space is None:
            return
        tokens = getattr(input_space, '_last_tokens', None)
        if not tokens:
            return
        B, N = shape
        layer._ensure_category_table(TheGrammar)
        cat = torch.full((B, N), 0, dtype=torch.long)
        for b, row in enumerate(tokens):
            if not row:
                continue
            cats = map_nltk_tags_to_categories(row[:N])
            for p, name in enumerate(cats):
                cat[b, p] = layer._category_index.get(name, 0)
        layer._seed_category(cat)
```

- [ ] **Step 4: Run**

Expected: PASS.

- [ ] **Step 5: Full regression**

Run:
```bash
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel" && .venv/bin/python -m pytest test/ -x -q
```
Expected: all previously-passing tests still pass.

- [ ] **Step 6: Commit**

---

## Task 9: SVO extraction from derivation trace

**Files:**
- Modify: `bin/Language.py` — `SyntacticLayer._extract_svo_from_trace`; remove the positional SVO tap from the legacy cascade.
- Create: `test/test_svo_extraction.py`.

- [ ] **Step 1: Write the failing test**

Create `basicmodel/test/test_svo_extraction.py`:

```python
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import torch
from util import TheXMLConfig
from Language import Grammar, SyntacticLayer


class TestSVOFromTrace(unittest.TestCase):
    def setUp(self):
        TheXMLConfig._overlay = getattr(TheXMLConfig, '_overlay', {})
        TheXMLConfig._overlay['WordSpace.chartCompose'] = True

    def tearDown(self):
        if hasattr(TheXMLConfig, '_overlay'):
            TheXMLConfig._overlay.pop('WordSpace.chartCompose', None)

    def test_svo_for_n_v_n(self):
        from Spaces import WordSubSpace
        g = Grammar()
        g.configure({'upward': {'S': ['S VO'], 'VO': ['V O']}})
        B, N, D = 1, 3, 4
        layer = SyntacticLayer(nInput=N, nOutput=N,
                               rules=list(range(len(g.rules_upward))),
                               max_depth=N - 1, hidden_dim=16, grammar=g)
        sub = WordSubSpace(nDim=D, nWhat=D, nWhere=0, nWhen=0,
                           max_depth=8, max_arity=3, batch=B)
        layer._ensure_category_table(g)
        cats = torch.tensor([[layer._category_index['S'],
                              layer._category_index['V'],
                              layer._category_index['O']]])
        layer._seed_category(cats)
        data = torch.randn(B, N, D)
        layer.compose(data, sub, g)
        svo = layer.last_svo
        self.assertIsNotNone(svo)
        s, v, o = svo
        self.assertTrue(torch.allclose(s[0, 0], data[0, 0]))
        self.assertTrue(torch.allclose(v[0, 0], data[0, 1]))
        self.assertTrue(torch.allclose(o[0, 0], data[0, 2]))

    def test_svo_none_without_outer_s_rule(self):
        from Spaces import WordSubSpace
        g = Grammar()
        g.configure({'S': ['not(S)']})  # no S -> S VO
        B, N, D = 1, 3, 4
        layer = SyntacticLayer(nInput=N, nOutput=N, rules=g.symbolic(),
                               max_depth=N - 1, hidden_dim=16, grammar=g)
        sub = WordSubSpace(nDim=D, nWhat=D, nWhere=0, nWhen=0,
                           max_depth=8, max_arity=3, batch=B)
        layer.compose(torch.randn(B, N, D), sub, g)
        self.assertIsNone(layer.last_svo)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Verify fails**

Expected: first test FAIL.

- [ ] **Step 3: Implement `_extract_svo_from_trace`**

Add to `SyntacticLayer` (below `_apply_rules_to_pairs`):

```python
    def _extract_svo_from_trace(self, grammar, original_data):
        """Walk self._derivation_trace → last_svo (subject, verb, object).

        Looks for the outermost `S -> S VO` firing and its matching
        inner `VO -> V O` firing; subject = arg-0 of outer, verb/object =
        arg-0/arg-1 of inner. Leaves last_svo None if no batch row has
        the canonical shape.
        """
        if self._derivation_trace is None:
            return
        B, N, D = original_data.shape
        s_list, v_list, o_list = [], [], []
        any_valid = False
        zero = torch.zeros(1, D, device=original_data.device)
        for b in range(B):
            trace = self._derivation_trace[b]
            outer = None
            for entry in reversed(trace):
                rule_id = entry[0]
                rule = grammar.rules[rule_id]
                if rule.lhs == 'S' and rule.rhs_symbols == ('S', 'VO'):
                    outer = entry
                    break
            if outer is None:
                s_list.append(zero); v_list.append(zero); o_list.append(zero)
                continue
            left_slot, right_slot = outer[1], outer[2]
            vo_entry = None
            for entry in trace:
                rule_id = entry[0]
                rule = grammar.rules[rule_id]
                if (rule.lhs == 'VO' and rule.rhs_symbols == ('V', 'O')
                        and entry[3] == right_slot):
                    vo_entry = entry
                    break
            if vo_entry is None:
                s_list.append(zero); v_list.append(zero); o_list.append(zero)
                continue
            s_list.append(original_data[b, left_slot:left_slot + 1])
            v_list.append(original_data[b, vo_entry[1]:vo_entry[1] + 1])
            o_list.append(original_data[b, vo_entry[2]:vo_entry[2] + 1])
            any_valid = True
        if any_valid:
            self.last_svo = (
                torch.stack(s_list, dim=0),
                torch.stack(v_list, dim=0),
                torch.stack(o_list, dim=0),
            )
```

Call at the end of `_compose_vector_chart`, replacing the final `return live, None`:

```python
        self._extract_svo_from_trace(grammar, data)
        return live, self.last_svo
```

- [ ] **Step 4: Remove the positional SVO tap from the legacy cascade**

In `_compose_vector` (legacy path), delete the block that reads `if min_leaves >= 3 and max_leaves >= 3: self.last_svo = (...)` and replace with the comment:

```python
        # Positional SVO tap removed 2026-04-20: SVO is now derived from
        # the chart-compose derivation trace by _extract_svo_from_trace.
        # Legacy cascade leaves self.last_svo None (set by compose()).
```

- [ ] **Step 5: Run**

Expected: both tests PASS; full regression still green.

- [ ] **Step 6: Commit**

---

## Task 10: Downward head emission (`S → C`)

**Files:**
- Modify: `bin/Language.py` — new `SyntacticLayer.emit_head(state, symbolicSpace)` method; new `WordSpace.reconstruct(state)` orchestrator.
- Modify: `bin/Spaces.py` — `SymbolicSpace.codebook_match(state)` helper returning `(best_idx, contained_contribution, residual)`.
- Create: `test/test_head_emission.py`.

**Design:** The MVP downward rule `S → C` is the simplest possible generator — it takes the deep state at position 0 (the fold of chart compose), finds the codebook atom that best matches via scalar parthood, and returns the atom + residual. This is the direct answer to the original request: *"examine a complete symbolic state, compare with the codebook in SymbolicSpace; the word/symbol that most nearly matches will be the one that best represents that meaning."*

- [ ] **Step 1: Write the failing test**

Create `basicmodel/test/test_head_emission.py`:

```python
"""Downward S -> C emits the codebook atom that best matches the deep state."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import torch
from util import TheXMLConfig, init_config
from Language import Grammar, SyntacticLayer


class TestHeadEmission(unittest.TestCase):
    def setUp(self):
        TheXMLConfig._overlay = getattr(TheXMLConfig, '_overlay', {})
        TheXMLConfig._overlay['WordSpace.chartCompose'] = True

    def tearDown(self):
        if hasattr(TheXMLConfig, '_overlay'):
            TheXMLConfig._overlay.pop('WordSpace.chartCompose', None)

    def test_emit_head_returns_best_codebook_idx(self):
        # Craft a codebook of 3 known atoms; the deep state exactly matches
        # atom 1 plus some noise. emit_head should return idx=1.
        from Spaces import Codebook
        cb = Codebook()
        cb.create(nInput=0, nVectors=3, nDim=4, customVQ=True,
                  monotonic=True, passThrough=False)
        # Replace the random codebook with three known atoms.
        with torch.no_grad():
            cb.getW().copy_(torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]))
        g = Grammar()
        g.configure({'upward': {'S': ['not(S)']},
                     'downward': {'S': ['C']}})
        layer = SyntacticLayer(nInput=4, nOutput=4, rules=g.symbolic(),
                               max_depth=2, hidden_dim=16, grammar=g)
        # The deep state is close to atom 1 + a small residual.
        state = torch.tensor([[0.05, 0.9, 0.02, 0.01]])  # [B=1, D=4]
        best_idx, contained, residual = layer.emit_head(state, cb)
        self.assertEqual(best_idx.item(), 1)
        # Contained contribution should be a scalar multiple of atom 1.
        self.assertTrue(torch.allclose(contained[0, 1:2], torch.tensor([[0.9]]),
                                       atol=1e-4))
        # Residual must be smaller in norm than the original state.
        self.assertLess(residual.norm().item(), state.norm().item())

    def test_reconstruct_one_word_sentence(self):
        # End-to-end: a one-leaf state → emit_head gives the matching atom.
        from Spaces import Codebook
        cb = Codebook()
        cb.create(nInput=0, nVectors=2, nDim=3, customVQ=True,
                  monotonic=True, passThrough=False)
        with torch.no_grad():
            cb.getW().copy_(torch.tensor([[1.0, 0.0, 0.0],
                                          [0.0, 1.0, 0.0]]))
        g = Grammar()
        g.configure({'upward': {'S': ['not(S)']},
                     'downward': {'S': ['C']}})
        layer = SyntacticLayer(nInput=3, nOutput=3, rules=g.symbolic(),
                               max_depth=2, hidden_dim=16, grammar=g)
        state = torch.tensor([[0.2, 0.8, 0.0]])
        idx, _, _ = layer.emit_head(state, cb)
        self.assertEqual(idx.item(), 1)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Verify fails**

Expected: FAIL — `emit_head` missing.

- [ ] **Step 3: Implement `emit_head` on `SyntacticLayer`**

Add below `_extract_svo_from_trace`:

```python
    def emit_head(self, state, codebook):
        """Downward `S -> C`: emit the codebook atom that best matches `state`.

        Uses scalar parthood (clipped cosine against codebook rows) to
        measure how much of each atom lives in `state`. Returns:

          best_idx      [B]  — codebook row index of the best match.
          contained     [B, D] — part_score * atom (the slice of the atom
                                 that is actually in `state`).
          residual      [B, D] — state - contained.

        This is the atomic operation of the downward generator: one step
        of "look at the remaining meaning, emit the atom it is most
        richly part of, subtract its contribution."
        """
        # state: [B, D]; codebook.getW(): [V, D]
        W = codebook.getW()
        # Cosine-style part score per atom.
        state_norm = state / state.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        W_norm = W / W.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        scores = state_norm @ W_norm.T          # [B, V]
        scores = scores.clamp(0.0, 1.0)          # non-negative part
        best_idx = scores.argmax(dim=-1)         # [B]
        best_atom = W[best_idx]                  # [B, D]
        # Scalar contribution = score * atom — how much of atom is in state.
        scalar = scores.gather(1, best_idx.unsqueeze(-1))  # [B, 1]
        contained = scalar * best_atom
        residual = state - contained
        return best_idx, contained, residual
```

Add `WordSpace.reconstruct` (below `reverseSymbols`):

```python
    def reconstruct(self, state, symbolicSpace, max_tokens=1):
        """Run the downward grammar on a deep state.

        MVP: emit_head on `S -> C` exactly once, returning the head atom
        index and the residual. `max_tokens > 1` is reserved for later
        expansion (e.g. NP VP templates that consume the residual).

        Returns a dict:
          'heads':      list[int] of length min(max_tokens, 1)
          'residual':   [B, D] tensor — leftover meaning after emission
          'state':      [B, D] tensor — original input state (echo)
        """
        layer = self.syntacticLayer
        if layer is None or state is None:
            return {'heads': [], 'residual': state, 'state': state}
        cb = symbolicSpace.subspace.basis
        idx, contained, residual = layer.emit_head(state, cb)
        return {
            'heads': idx.tolist(),
            'contained': contained,
            'residual': residual,
            'state': state,
        }
```

- [ ] **Step 4: Run**

Expected: both emission tests PASS.

- [ ] **Step 5: Commit**

---

## Task 11: Inline `<sentences>` dataset + head-prediction training hook

**Files:**
- Modify: `bin/data.py` — `Data.load_inline` reads `<sentences><sentence head="...">...</sentence></sentences>`.
- Modify: `data/model.xsd` — define `sentencesType`.
- Create: `data/HeadEmission.xml` — minimal model config for the MVP task.
- Modify: `bin/Models.py` — `MentalModel.forward` exposes `self._predicted_head`; `head_prediction_loss` computes cross-entropy against the expected head id.
- Create: `test/test_head_emission.py::TestHeadPredictionCorpus` — run the MVP forward/loss loop over 5 sentences.

- [ ] **Step 1: Write the failing test**

Append to `test/test_head_emission.py`:

```python
class TestHeadPredictionCorpus(unittest.TestCase):
    """MVP: feed inline sentences; model predicts each sentence's head word.

    Untrained model fails — this test is xfail until training lands. What
    the test fixes is the PLUMBING: forward runs, emit_head fires on the
    deep state, and `MentalModel._predicted_head` exposes a per-batch
    prediction the loss fn can reach.
    """

    def test_head_prediction_path_is_wired(self):
        import warnings
        import Models
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        init_config(path=os.path.join(data_dir, 'HeadEmission.xml'),
                    defaults_path=os.path.join(data_dir, 'model.xml'))
        import Language
        Language.TheGrammar._configured = False
        model, _ = Models.MentalModel.from_config(
            os.path.join(data_dir, 'HeadEmission.xml'))
        sentences = ['the teacher helped the student']
        outputs = [torch.tensor([0.0])] * len(sentences)
        with Models.TheData.runtime_batch(sentences, outputs), \
             warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            train_input, _ = model.inputSpace.getTrainData()
            x = model.inputSpace.prepInput(train_input[:1])
            model.eval()
            with torch.no_grad():
                model.forward(x)
        head = getattr(model, '_predicted_head', None)
        self.assertIsNotNone(head)
        self.assertIsInstance(head, list)  # list of codebook indices
        self.assertEqual(len(head), 1)
```

- [ ] **Step 2: Verify fails**

Expected: FAIL — `HeadEmission.xml` missing / `_predicted_head` missing.

- [ ] **Step 3: Create `data/HeadEmission.xml`**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!-- MVP head-prediction task: inline sentences, downward 'S -> C'. -->
<model xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:noNamespaceSchemaLocation="model.xsd">
  <architecture>
    <nEpochs>1</nEpochs>
    <batchSize>1</batchSize>
  </architecture>
  <WordSpace>
    <chartCompose>true</chartCompose>
    <downwardGeneration>true</downwardGeneration>
    <language>
      <grammar>
        <upward>
          <S>S VO</S>
          <VO>V O</VO>
        </upward>
        <downward>
          <S>C</S>
        </downward>
      </grammar>
      <interpretation>0.5</interpretation>
    </language>
  </WordSpace>
  <data>
    <dataset>inline</dataset>
    <sentences>
      <sentence head="teacher">the teacher helped the student</sentence>
      <sentence head="doctor">the doctor healed the patient</sentence>
      <sentence head="mentor">the mentor encouraged the apprentice</sentence>
      <sentence head="bully">the bully punched the kid</sentence>
      <sentence head="thief">the thief robbed the merchant</sentence>
    </sentences>
  </data>
</model>
```

- [ ] **Step 4: Add `<sentences>`/`<sentence>` + `<downwardGeneration>` to XSD**

In `data/model.xsd`:

```xml
      <xs:element name="downwardGeneration" type="xs:boolean" minOccurs="0"/>

  <xs:complexType name="dataType">
    <xs:sequence>
      <xs:element name="dataset" type="xs:string" minOccurs="0"/>
      <xs:element name="sentences" type="sentencesType" minOccurs="0"/>
    </xs:sequence>
  </xs:complexType>

  <xs:complexType name="sentencesType">
    <xs:sequence>
      <xs:element name="sentence" maxOccurs="unbounded">
        <xs:complexType>
          <xs:simpleContent>
            <xs:extension base="xs:string">
              <xs:attribute name="head" type="xs:string" use="required"/>
            </xs:extension>
          </xs:simpleContent>
        </xs:complexType>
      </xs:element>
    </xs:sequence>
  </xs:complexType>
```

(If `dataType` already exists, just append the `sentences` element to its sequence rather than redefining it. Check with `grep 'name="data"' data/model.xsd` first.)

- [ ] **Step 5: Wire inline loader + head prediction hook**

In `bin/data.py`, find `Data.load_inline` (or equivalent path for `<dataset>inline</dataset>`). Extend it to read `<sentences>`: parse each `<sentence head="X">text</sentence>` into `(text, head_word)` pairs.

In `bin/Models.py` `MentalModel.forward`, after compose finishes and the deep symbolic state is available, call `reconstruct` and stash:

```python
        # Downward head emission (MVP) — runs only when the flag is on.
        self._predicted_head = None
        try:
            gen_on = bool(TheXMLConfig.get("WordSpace.downwardGeneration"))
        except (KeyError, AttributeError, TypeError):
            gen_on = False
        if gen_on:
            final_state = symbols[..., 0, :]  # [B, D] — position 0 is the chart fold
            result = self.wordSpace.reconstruct(final_state, self.symbolicSpace)
            self._predicted_head = result['heads']
```

(`symbols` is the final per-batch symbol tensor from the forward pass; if the fold lives at a different position, use the slot recorded in the last trace entry.)

- [ ] **Step 6: Run**

Expected: test PASSES — the plumbing is wired; actual head correctness is an xfail training problem.

- [ ] **Step 7: Commit**

---

## Task 12: `test_universality` update + config defaults + full regression

**Files:**
- Modify: `test/test_universality.py` — `_get_svo_and_luminosity` reads `syntacticLayer.last_svo` + `model._universality_score`.
- Modify: `data/MentalModel.xml` — already has `<chartCompose>true</chartCompose>`; confirm + add `<downwardGeneration>false</downwardGeneration>` so universality runs without the head-emission side effect.

- [ ] **Step 1: Replace `_get_svo_and_luminosity` stub**

```python
    def _get_svo_and_luminosity(self, sentence):
        """Run forward; SVO is now derived from the chart-compose trace.

        After Phase D, SyntacticLayer._extract_svo_from_trace populates
        `wordSpace.syntacticLayer.last_svo`; the MentalModel hook computes
        the universality score and stores it on `model._universality_score`.
        """
        _run_forward(self.model, [sentence])
        sl = self.model.wordSpace.syntacticLayer
        svo = getattr(sl, 'last_svo', None)
        score = getattr(self.model, '_universality_score', None)
        if svo is None:
            return None, None
        return svo, (score.item() if torch.is_tensor(score) else score)
```

- [ ] **Step 2: Confirm MentalModel.xml knobs**

Ensure `data/MentalModel.xml` contains:

```xml
    <chartCompose>true</chartCompose>
    <downwardGeneration>false</downwardGeneration>
```

- [ ] **Step 3: Run**

Run:
```bash
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel" && .venv/bin/python -m pytest test/test_universality.py -v
```
Expected: xfail-as-before or pass (xfail strict=False is tolerant).

- [ ] **Step 4: Full regression sweep**

Run:
```bash
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel" && .venv/bin/python -m pytest test/ -q
```
Expected: every test passes or xfails as documented.

- [ ] **Step 5: Hand off**

Present diff + test status. UniversalityWeight bump (Phase E) is a training-time decision for the user.

- [ ] **Step 6: Commit**

---

## Self-review

- **Spec coverage:** Both halves of the user's scope are covered —
  (a) reconstruction/generation via `<downward>`: Task 2 (grammar split), Task 10 (emit_head), Task 11 (inline corpus + head loss).
  (b) LearnedSVO phases: Task 1 (A), Tasks 3–6 (B), Tasks 7–8 (C), Task 9 (D), Task 12 (F).
- **Placeholder scan:** Every step has actual code or an exact command. One gray area is the `bin/data.py` inline-loader extension (Task 11, Step 5) — it references a path that may not exist. If `Data` doesn't already have a `load_inline` method, the task should grep for `<dataset>inline` usage first and bolt on from whatever shape exists today. This is flagged rather than invented.
- **Type consistency:** `_derivation_trace` is always `[[] for _ in range(B)]` or a list-of-lists-of-5-tuples. `last_svo` is either `None` or `(S, V, O)` where each element is `[B, 1, D]` to match `TruthLayer.universality`'s existing expectation. `emit_head` returns `(best_idx: LongTensor[B], contained: [B, D], residual: [B, D])`. `reconstruct` returns a dict with stable keys.
- **Ambiguity:** "final state" in Task 11 Step 5 is defined as `symbols[..., 0, :]` (position 0 is where chart compose folds); if the model geometry differs, the task note says to use the last trace entry's `merged_slot` instead — an explicit fallback, not a `TODO`.
