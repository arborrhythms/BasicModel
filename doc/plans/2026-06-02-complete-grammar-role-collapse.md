# Complete Grammar Role-Collapse Rewrite

> **Status (2026-06-02): spec only. Do not implement as part of this plan.**
>
> Target file: `data/complete.grammar`
>
> Related plans:
> - `doc/plans/2026-05-29-grammar-file-refactor.md`
> - `doc/plans/2026-05-30-subsymbolic-analyzer-terminal-emitter.md`

---

## Goal

Rewrite `complete.grammar` so grammar categories are no longer part-of-speech
or surface-marker categories. The grammar should define only operator roles.
Categories such as `NP3`, `AP`, `VP1`, `N3`, `DET`, `CONJ_L45`, and
`QRIGHT_AP` should disappear from the grammar file. They become learned
taxonomic/POS structure inferred from stable participation in operator roles.

The grammar file should answer this question only:

```text
Which operator is being applied, and which role in that operator does each
operand occupy?
```

It should not answer:

```text
What part of speech is this operand?
What syntactic category did this token previously occupy?
Which marker token licensed this role?
```

---

## Current Problem

The current `complete.grammar` still carries a transitional layer of role-like
categories that are actually renamed POS/category states:

```xml
<rule>QLEFT_NP3 = NP3</rule>
<rule>QRIGHT_AP = AP</rule>
<rule>QRIGHT_NP3 = NP3</rule>
<rule>QRIGHT_S34 = S34</rule>
<rule>CONJ_L45 = S45</rule>
<rule>CONJ_R45 = S45</rule>
```

The generate section mirrors those renames:

```xml
<rule>NP3 = QLEFT_NP3</rule>
<rule>AP = QRIGHT_AP</rule>
<rule>NP3 = QRIGHT_NP3</rule>
<rule>S34 = QRIGHT_S34</rule>
<rule>S45 = CONJ_L45</rule>
<rule>S45 = CONJ_R45</rule>
```

This keeps the old POS/category grammar alive indirectly. It also forces
relation rules to be specialized by operand category:

```xml
<rule query="true">REL_T = isEqual.forward(QLEFT_NP3, QRIGHT_AP)</rule>
<rule query="true">REL_T = isEqual.forward(QLEFT_NP3, QRIGHT_NP3)</rule>

<rule query="true">QLEFT_NP3, QRIGHT_AP = isEqual.reverse(REL_T)</rule>
<rule query="true">QLEFT_NP3, QRIGHT_NP3 = isEqual.reverse(REL_T)</rule>
```

After the role collapse, there should be one equality relation over equality
roles, not separate equality rules for each POS/category pairing.

---

## Role Naming Contract

Operator role names use this convention:

```text
operator_I1
operator_O1
operator_O2
```

Meaning:

```text
I = input/result/parent role for the operator
O = output/operand/child role for the operator
number = argument position
```

The canonical binary contract is:

```xml
<rule>op_I1 = op.forward(op_O1, op_O2)</rule>
<rule>op_O1, op_O2 = op.reverse(op_I1)</rule>
```

The canonical unary contract is:

```xml
<rule>op_I1 = op.forward(op_O1)</rule>
<rule>op_O1 = op.reverse(op_I1)</rule>
```

Examples:

```xml
<rule query="true">isEqual_I1 = isEqual.forward(isEqual_O1, isEqual_O2)</rule>
<rule query="true">isEqual_O1, isEqual_O2 = isEqual.reverse(isEqual_I1)</rule>

<rule query="false">not_I1 = not.forward(not_O1)</rule>
<rule query="false">not_O1 = not.reverse(not_I1)</rule>

<rule query="false">lift_I1 = lift.forward(lift_O1, lift_O2)</rule>
<rule query="false">lift_O1, lift_O2 = lift.reverse(lift_I1)</rule>
```

Apply this convention everywhere in `complete.grammar`.

---

## Equality Rewrite

Collapse the existing category-specific equality rules:

```xml
<rule query="true">QLEFT_NP3, QRIGHT_AP = isEqual.reverse(REL_T)</rule>
<rule query="true">QLEFT_NP3, QRIGHT_NP3 = isEqual.reverse(REL_T)</rule>
<rule>NP_EQ345, NP345 = isEqual.reverse(REL_T)</rule>
<rule>NP_EQ4, AP4 = isEqual.reverse(REL_T)</rule>
```

into role-based equality rules:

```xml
<rule query="true">isEqual_I1 = isEqual.forward(isEqual_O1, isEqual_O2)</rule>
<rule query="true">isEqual_O1, isEqual_O2 = isEqual.reverse(isEqual_I1)</rule>

<rule query="false">isEqual_I1 = isEqual.forward(isEqual_O1, isEqual_O2)</rule>
<rule query="false">isEqual_O1, isEqual_O2 = isEqual.reverse(isEqual_I1)</rule>
```

`query="true"` means the rule asks an equality question and dispatches to
answer-producing equality semantics.

`query="false"` means the rule asserts or composes equality as a relation.

The `query` attribute must be explicit for both forms.

---

## Parthood Rewrite

Replace both `queryPart` and `assertPart` with one relation name:

```text
isPart
```

The current split:

```xml
<rule>REL_T = queryPart.forward(QLEFT_PART34, QRIGHT_S34)</rule>
<rule>REL_T = assertPart.forward(S_PART34, S34)</rule>

<rule>QLEFT_PART34, QRIGHT_S34 = queryPart.reverse(REL_T)</rule>
<rule>S_PART34, S34 = assertPart.reverse(REL_T)</rule>
```

becomes:

```xml
<rule query="true">isPart_I1 = isPart.forward(isPart_O1, isPart_O2)</rule>
<rule query="true">isPart_O1, isPart_O2 = isPart.reverse(isPart_I1)</rule>

<rule query="false">isPart_I1 = isPart.forward(isPart_O1, isPart_O2)</rule>
<rule query="false">isPart_O1, isPart_O2 = isPart.reverse(isPart_I1)</rule>
```

Runtime support required:

```text
isPart + query=true   -> query/parthood-answer semantics
isPart + query=false  -> assertive parthood semantics
```

This mirrors the existing equality pattern, where `isEqual` remains one grammar
relation and `query="true"` selects answer-producing behavior.

---

## Start States

The existing top-level starts are SymbolicSpace-specific:

```xml
<start name="absolute_truth">ABS_T</start>
<start name="relative_truth">REL_T</start>
```

They should not remain at the top level after the file has both
`PerceptualSpace` and `SymbolicSpace` sections.

Target shape:

```xml
<PerceptualSpace>
  <start name="everything">U</start>
  ...
</PerceptualSpace>

<SymbolicSpace>
  <start name="relative_truth">isEqual_I1</start>
  <start name="relative_truth">isPart_I1</start>
  <start name="absolute_truth">exist_I1</start>
  ...
</SymbolicSpace>
```

`U` is the singular PerceptualSpace start state. It means the whole perceptual
surface, or "everything" currently visible to the perceptual analyzer.

The SymbolicSpace start set should include every role state that counts as a
completed symbolic expression. Exact SS starts should be finalized during
implementation, but the old top-level `ABS_T` / `REL_T` states should not be
kept as grammar-wide starts.

Loader support required:

```text
Current loader behavior:
  <start> is parsed as global Grammar metadata.

Required loader behavior:
  PerceptualSpace.start configures PS accepted starts.
  SymbolicSpace.start configures SS accepted starts.
```

The rewrite should not land until space-scoped starts are supported, or until a
temporary compatibility strategy is explicitly chosen.

---

## Delete Category Rename Rules

All rules that only rename one state into another should disappear.

Examples to delete:

```xml
<rule>QRIGHT_AP = AP</rule>
<rule>QLEFT_NP3 = NP3</rule>
<rule>QRIGHT_NP3 = NP3</rule>
<rule>QRIGHT_S34 = S34</rule>
<rule>NP_EQ3 = NP3</rule>
<rule>S_PART34 = NP34</rule>
<rule>QLEFT_PART34 = S_PART34</rule>
<rule>CONJ_L45 = S45</rule>
<rule>CONJ_R45 = S45</rule>
<rule>DISJ_L45 = S45</rule>
<rule>DISJ_R45 = S45</rule>
<rule>CONJ_L3 = NP3</rule>
<rule>CONJ_R3 = NP3</rule>
<rule>DISJ_L3 = NP3</rule>
<rule>DISJ_R3 = NP3</rule>
```

Their generate mirrors should disappear too:

```xml
<rule>AP = QRIGHT_AP</rule>
<rule>NP3 = QLEFT_NP3</rule>
<rule>NP3 = QRIGHT_NP3</rule>
<rule>S34 = QRIGHT_S34</rule>
<rule>NP3 = NP_EQ3</rule>
<rule>NP34 = S_PART34</rule>
<rule>S_PART34 = QLEFT_PART34</rule>
<rule>S45 = CONJ_L45</rule>
<rule>S45 = CONJ_R45</rule>
```

After this rewrite, POS/category membership is learned from role participation,
not declared by grammar projection rules.

---

## Operator Coverage

The role naming convention should cover at least these operators:

```text
exist
isEqual
isPart
not
non
conjunction
disjunction
intersection
union
lift
lower
```

The exact role arity comes from each `GrammarLayer` subclass and the existing
forward/reverse contract.

For binary operators:

```xml
<rule query="false">conjunction_I1 = conjunction.forward(conjunction_O1, conjunction_O2)</rule>
<rule query="false">conjunction_O1, conjunction_O2 = conjunction.reverse(conjunction_I1)</rule>
```

For unary operators:

```xml
<rule query="false">exist_I1 = exist.forward(exist_O1)</rule>
<rule query="false">exist_O1 = exist.reverse(exist_I1)</rule>
```

If an operator is intentionally lossy or non-invertible, the generate rule
should either be omitted with an explicit comment or represented by the
operator's documented pseudo-inverse. Do not silently keep stale category
mirrors as a substitute for an operator reverse.

---

## Runtime and Loader Changes Required

This grammar rewrite implies code changes outside `complete.grammar`.

1. Add an `isPart` grammar layer name.

   Options:

   ```text
   A. Rename/alias PartLayer.rule_name to isPart.
   B. Add IsPartLayer as the canonical grammar-facing class.
   C. Keep PartLayer internally but register GRAMMAR_LAYER_CLASSES["isPart"].
   ```

2. Dispatch `isPart` by `query` metadata.

   Existing equality behavior:

   ```text
   isEqual + query=true -> queryEqual
   ```

   Required parthood behavior:

   ```text
   isPart + query=true  -> queryPart-like answer semantics
   isPart + query=false -> assertPart/part-like assertion semantics
   ```

3. Replace relative-rule detection.

   Current detection includes:

   ```text
   isEqual
   queryPart
   assertPart
   part
   REL_T start
   ```

   Required detection should include:

   ```text
   isEqual
   isPart
   symbolic relative start role states
   ```

4. Support space-scoped starts.

   `PerceptualSpace` and `SymbolicSpace` should each carry independent
   start metadata.

5. Revisit identity-rule injection.

   The loader currently injects an identity rule for the primary global start
   symbol. After space-scoped starts, identity-rule injection must be scoped or
   explicitly limited to SymbolicSpace compatibility.

---

## Test Updates

Update tests that currently require `queryPart` and `assertPart`.

Expected changes:

```text
test_mental_model.py
  required ops should include isPart
  required ops should not include queryPart/assertPart

test_stm_relative_sentence_end_state.py
  relative op set should include isEqual/isPart
  references to REL_T should be replaced or scoped to SS compatibility

test_grammar_rewrite.py
  rename-rule assertions should invert:
    old: per-position categories present
    new: no POS/category projection rules remain
```

Add tests for:

```text
complete.grammar has no queryPart/assertPart method names
complete.grammar has no bare POS/category rename rules
all relation rules carry explicit query="true" or query="false"
isEqual uses isEqual_I1/isEqual_O1/isEqual_O2
isPart uses isPart_I1/isPart_O1/isPart_O2
top-level <start> is absent
PerceptualSpace has <start name="everything">U</start>
SymbolicSpace has its own starts
```

---

## Acceptance Criteria

The rewrite is complete when:

1. `data/complete.grammar` contains no top-level `<start>` tags.
2. `PerceptualSpace` has exactly the universal start:

   ```xml
   <start name="everything">U</start>
   ```

3. `SymbolicSpace` owns the symbolic completion starts.
4. `queryPart` and `assertPart` no longer appear in the grammar rule table.
5. `isPart` appears as the single parthood operator.
6. Every relation rule explicitly has `query="true"` or `query="false"`.
7. Category rename rules are gone.
8. Old POS/category names are not used as grammar state names except during a
   documented compatibility phase.
9. Compose/generate mirrors exist for every invertible role rule.
10. Existing grammar loading tests pass after the required loader/runtime updates.

---

## Non-Goals

This spec does not implement:

- the `complete.grammar` rewrite,
- the `isPart` runtime alias,
- space-scoped start handling,
- POS/category learning,
- test updates.

It only defines the target shape and the work required before the grammar file
can be safely rewritten.
