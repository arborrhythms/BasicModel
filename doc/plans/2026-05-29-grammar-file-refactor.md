# Grammar File Refactor — .grammar XML + Layer Consolidation

> **Status (2026-05-29): planning.**
> - Move grammar rule classes out of `Layers.py` into `Language.py`
> - Replace inline grammar blocks and `.cfg` text files with named `.grammar` XML files
> - Wire grammar loading so `<grammar>name.grammar</grammar>` in any model config resolves to `data/name.grammar`

---

## Goal

Three problems to fix together:

1. **Grammar is scattered.** Rule classes (IntersectionLayer, UnionLayer, LiftLayer, …) live in `Layers.py` alongside non-grammar layers (TruthLayer, EmbeddingLayer, etc.). Language-specific code should be in `Language.py`.

2. **Grammar format is inconsistent.** Some configs use `<useGrammar>shamathaSpeech</useGrammar>` (a string enum), others use inline `<grammar><S>S = not(S)</S></grammar>` blocks (MM_xor.xml, MM_grammar.xml), and the old `.cfg` text files still exist on disk. None of these load from a clean, structured external file.

3. **Grammar rules carry no metadata.** Tier (C vs S), arity, invertibility, and future POS constraints are either hardcoded in `Language.py` or absent. A structured grammar file is the right place for them.

---

## 1. Grammar File Format

New extension: `.grammar`. Location: `data/`. Loaded via XML parser.

```xml
<?xml version="1.0"?>
<grammar name="default">

  <!-- Subsymbolic tier (C): operate on continuous activations -->
  <rule name="union"        tier="C" arity="2" invertible="false"/>
  <rule name="intersection" tier="C" arity="2" invertible="false"/>
  <rule name="lift"         tier="C" arity="1" invertible="true"/>
  <rule name="lower"        tier="C" arity="1" invertible="true"/>

  <!-- Symbolic tier (S): operate on resolved codebook activations -->
  <rule name="not"          tier="S" arity="1" invertible="true"/>
  <rule name="conjunction"  tier="S" arity="2" invertible="false"/>
  <rule name="disjunction"  tier="S" arity="2" invertible="false"/>
  <rule name="part"         tier="S" arity="2" invertible="false"/>
  <rule name="equals"       tier="S" arity="2" invertible="false"/>
  <rule name="query"        tier="S" arity="1" invertible="false"/>

</grammar>
```

**Attribute spec:**

| Attribute | Values | Required | Meaning |
|-----------|--------|----------|---------|
| `name` | string matching a layer class name | yes | Identifies the `GrammarLayer` subclass to instantiate |
| `tier` | `C` or `S` | yes | Subsymbolic (C) or symbolic (S). Controls which host layer routes the rule. |
| `arity` | `1` or `2` | yes | Unary or binary. Determines whether `forward(x)` or `forward(left, right)` is called. |
| `invertible` | `true` or `false` | yes | Whether a `generate` / reverse pass is defined. |
| `pos` | space-separated POS tags | no | Optional: POS categories this rule may produce. Used by `pos_head` scoring (future). |

---

## 2. Files to Create

### `data/default.grammar`
Migrated from `grammar2.cfg`. Subsymbolic and symbolic rules, no POS specification.
Rules: union, intersection, lift, lower, not, conjunction, disjunction, part, equals, query.

### `data/shamatha.grammar`
Rules appropriate for Shamatha Speech mode. Extends default with contiguity-aware ops
as documented in `doc/plans/2026-04-28-shamatha-speech-contiguity-handoff.md`.
Start as a copy of `default.grammar`; add contiguity rules when that plan is implemented.

### `data/xor.grammar`
Minimal grammar for XOR test model (currently inline in `MM_xor.xml`):
union, intersection, not. No POS constraints.

---

## 3. Config XML Change

Replace all existing grammar references with:

```xml
<grammar>default.grammar</grammar>
```

**Before → After mapping:**

| File | Old tag | New tag |
|------|---------|---------|
| `data/MM_xor.xml` | inline `<grammar><S>…</S></grammar>` block | `<grammar>xor.grammar</grammar>` |
| `data/MM_grammar.xml` | inline `<grammar><compose>…</compose></grammar>` | `<grammar>default.grammar</grammar>` |
| Any file with `<useGrammar>shamathaSpeech</useGrammar>` | enum string | `<grammar>shamatha.grammar</grammar>` |
| Any file with `<useGrammar>default</useGrammar>` or absent | default fallback | `<grammar>default.grammar</grammar>` |

The `<useGrammar>` tag is deprecated. The parser should warn if it encounters it and fall back to `default.grammar`.

---

## 4. Grammar Loading Code

Add a module-level loader in `Language.py`:

```python
import xml.etree.ElementTree as ET
from pathlib import Path

_GRAMMAR_DIR = Path(__file__).parent.parent / "data"

def load_grammar(filename: str) -> list[dict]:
    """Load a .grammar XML file from data/ and return a list of rule dicts.

    Each dict has keys: name, tier, arity, invertible, pos (list[str]).
    """
    path = _GRAMMAR_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Grammar file not found: {path}")
    root = ET.parse(path).getroot()
    rules = []
    for elem in root.findall("rule"):
        rules.append({
            "name":       elem.attrib["name"],
            "tier":       elem.attrib["tier"],           # "C" or "S"
            "arity":      int(elem.attrib["arity"]),     # 1 or 2
            "invertible": elem.attrib.get("invertible", "false") == "true",
            "pos":        elem.attrib.get("pos", "").split(),
        })
    return rules
```

`LanguageLayer.__init__` receives the filename string (read from the model config `<grammar>` tag),
calls `load_grammar(filename)`, and uses the returned list to attach rule layers via `_attach_ops()`.

The existing `TheGrammar` / `grammar.cfg` lookup path is replaced by this loader.
`TheGrammar` may be kept as a compatibility shim that calls `load_grammar("default.grammar")` if
nothing else touches it — remove when all call sites are migrated.

---

## 5. Layer Consolidation — Move Grammar Layers to Language.py

The following classes move from `Layers.py` to `Language.py` (bottom of file, before `LanguageLayer`):

| Class | Current location |
|-------|----------------|
| `GrammarLayer` (base) | `Layers.py:1532` |
| `NotLayer` | `Layers.py:2391` |
| `NonLayer` | `Layers.py:2445` |
| `IntersectionLayer` | `Layers.py:2498` |
| `UnionLayer` | `Layers.py:2665` |
| `LiftLayer` | `Layers.py:2828` |
| `LowerLayer` | `Layers.py:3013` |
| `SymbolizeLayer` | `Layers.py:3147` |
| `ConjunctionLayer` | `Layers.py:3465` |
| `DisjunctionLayer` | `Layers.py:3579` |
| `IsEqualLayer` | `Layers.py:3744` |
| `PartLayer` | `Layers.py:3851` |
| `QueryLayer` | `Layers.py:4115` |

`SigmaLayer` and `PiLayer` stay in `Layers.py` — they are subsymbolic computation layers,
not grammar rule implementations.

**Procedure:**
1. Copy each class body verbatim into `Language.py` (after imports, before `LanguageLayer`).
2. In `Layers.py`, replace each class body with a shim import and deprecation warning:
   ```python
   # Moved to Language.py — import kept for backward compatibility.
   from Language import IntersectionLayer  # noqa: F401
   ```
3. Run the test suite. Fix any circular import issues (Language.py imports from Layers.py;
   Layers.py importing back creates a cycle — resolve by extracting any shared base class
   into a new `grammar_base.py` if needed, or by using lazy imports).
4. After tests pass, remove shim imports from Layers.py in a follow-up commit.

---

## 6. Tier Routing Integration

Once grammar layers are in `Language.py` and each rule's tier is declared in the `.grammar` file,
`BinaryStructuredReductionLayer` can gate rule eligibility by tier per position:

- Track a per-position tier bit alongside the activation tensor.
- Initial tier: C (all positions start subsymbolic).
- After a `lift` reduce: position tier flips to S.
- After a `lower` reduce: position tier flips to C.
- C-tier rules (union/intersection) only score when both positions are C-tier.
- S-tier rules only score when both positions are S-tier.

This replaces the ad-hoc `_SUBSYMBOLIC` set in `_forward_with_rule_dispatch` with a data-driven
tier check driven by the `.grammar` file.

---

## 7. Implementation Steps

1. **Create `.grammar` files** — `default.grammar`, `shamatha.grammar`, `xor.grammar` in `data/`.
2. **Add `load_grammar()`** to `Language.py`.
3. **Wire config loading** — update wherever `<useGrammar>` / `<grammar>` is parsed in
   model config reading code (likely `Models.py` or `Spaces.py`) to call `load_grammar()`.
4. **Move grammar layer classes** from `Layers.py` to `Language.py`. Add shims.
5. **Update `_attach_ops()`** in `LanguageLayer` to use the loaded rule list (tier, arity)
   instead of the existing grammar cfg lookup.
6. **Add tier tracking** to `BinaryStructuredReductionLayer` (per-position tier bit).
7. **Delete deprecated tags** — remove `<useGrammar>` handling after all configs updated.
8. **Delete `.cfg` files** — `grammar2.cfg`, `grammar_legacy.cfg` once `default.grammar` is verified.
9. **Run full test suite** and confirm no regressions.

---

## 8. Tests

- `load_grammar("default.grammar")` returns correct rule list with all fields.
- Unknown grammar filename raises `FileNotFoundError` with a clear message.
- `LanguageLayer` constructed from a model config that uses `<grammar>default.grammar</grammar>`
  attaches the same ops as the current default configuration.
- Tier gating: a C-tier position does not score S-tier rules (score is -inf or masked).
- Deprecation warning fires when `<useGrammar>` is encountered.
- Existing tests pass — no regression from the layer move.

---

## 9. Acceptance Criteria

- `data/default.grammar`, `data/shamatha.grammar`, `data/xor.grammar` exist and are valid XML.
- `<grammar>filename.grammar</grammar>` in any model config loads correctly.
- `<useGrammar>` is deprecated and logs a warning.
- All grammar layer classes are defined in `Language.py`; `Layers.py` has only shim imports.
- `BinaryStructuredReductionLayer` applies tier gating based on `.grammar` metadata.
- All existing tests pass.
