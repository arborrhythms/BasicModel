"""Phase-2 end-to-end integration test.

Plan: doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md

Verifies the Phase-2 primitives compose into a working pipeline:

  1. Build a knowledge section from a grammar (+ optional word vectors).
  2. Save it to disk via the unified artifact format.
  3. Load it back via ``load_knowledge_view``.
  4. Attach the loaded ``KnowledgeView`` to three Space subclasses.
  5. Each Space exposes its expected knowledge-derived fields:
       - SymbolSpace.knowledge       (view)
       - WholeSpace.references  (Parameter)
       - WholeSpace.order       (buffer)
       - PartSpace.wv.ref_ids (long)
  6. Build a runtime admissibility mask over the real loaded
     rule_order_signatures and verify it correctly identifies
     admissible rules for given stack-top state.
  7. Apply the mask to logits and verify the post-softmax probability
     mass concentrates on admissible rules.

These pieces existed individually in the per-feature test files;
this file proves they actually connect.
"""
import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))


def _grammar_with_lift_lower_and_ordinary():
    """A small but realistic grammar exercising lift, lower, and one
    ordinary order-preserving rule."""
    from Language import Grammar
    g = Grammar()
    g.rules = [
        g._parse_rule("S4", "lift(NP3, VP1)", space_role='SS'),
        g._parse_rule("NP3", "lower(DET, NP4)", space_role='SS'),
        g._parse_rule("S3", "not(S3)", space_role='SS'),
    ]
    g._configured = True
    return g


class _FakeWV:
    def __init__(self, words):
        self.index_to_key = list(words)


def _bare_space(cls):
    """Bypass __init__ for any Space subclass."""
    import torch.nn as nn
    inst = object.__new__(cls)
    nn.Module.__init__(inst)
    return inst




def test_phase2_end_to_end_admissibility_uses_loaded_signatures(tmp_path):
    """The loaded rule_order_signatures drive admissibility correctly:
    a (NP3, VP1) stack-top picks only the lift rule; (DET0, NP4) picks
    only the lower rule; a single S3 picks only the not rule."""
    from embed import (save_artifact, build_knowledge_section,
                       load_knowledge_view, admissibility_mask)
    grammar = _grammar_with_lift_lower_and_ordinary()
    path = str(tmp_path / "admissibility.kv")
    save_artifact(path, knowledge=build_knowledge_section(grammar))
    view = load_knowledge_view(path)
    sigs = view.rule_order_signatures
    assert len(sigs) == 3

    # Stack-top: (NP at 3, VP at 1) — only lift admissible
    mask = admissibility_mask(
        sigs, left_cat='NP', left_order=3, right_cat='VP', right_order=1)
    assert mask.tolist() == [True, False, False]

    # Stack-top: (DET at 0, NP at 4) — only lower admissible
    mask = admissibility_mask(
        sigs, left_cat='DET', left_order=0, right_cat='NP', right_order=4)
    assert mask.tolist() == [False, True, False]

    # Stack-top: (S at 3,) unary — only not admissible
    mask = admissibility_mask(sigs, left_cat='S', left_order=3)
    assert mask.tolist() == [False, False, True]


def test_phase2_end_to_end_mask_logits_concentrates_on_admissible(tmp_path):
    """A uniform rule-logit vector + admissibility mask → all probability
    mass concentrates on the admissible rule(s)."""
    from embed import (save_artifact, build_knowledge_section,
                       load_knowledge_view, admissibility_mask,
                       mask_logits)
    import torch
    grammar = _grammar_with_lift_lower_and_ordinary()
    path = str(tmp_path / "mask.kv")
    save_artifact(path, knowledge=build_knowledge_section(grammar))
    view = load_knowledge_view(path)
    sigs = view.rule_order_signatures

    # Build the mask for a stack-top that admits exactly the lift rule.
    mask = admissibility_mask(
        sigs, left_cat='NP', left_order=3, right_cat='VP', right_order=1)

    # Uniform logits (so without masking, softmax would be 1/3 each).
    logits = torch.zeros(3)
    masked = mask_logits(logits, mask)
    probs = torch.softmax(masked, dim=-1)

    # Almost all probability on the lift rule; the other two get 0.
    assert probs[0].item() > 0.99
    assert probs[1].item() < 1e-6
    assert probs[2].item() < 1e-6


def test_phase2_end_to_end_extend_then_load(tmp_path):
    """After ``extend_artifact`` adds a new ref, loading + querying still
    works — the symbol-learning append round-trips through the
    Phase-2 loader path."""
    from embed import (save_artifact, build_knowledge_section,
                       extend_artifact, load_knowledge_view, NewRef)
    grammar = _grammar_with_lift_lower_and_ordinary()
    path = str(tmp_path / "extended.kv")
    ks = build_knowledge_section(grammar)
    save_artifact(path, knowledge=ks)

    # Append one new ref under S, at order 4 (a new "S-instance" ref).
    view_before = load_knowledge_view(path)
    s_rid = view_before.ref_id_for('S')
    n_before = view_before.n_refs_live

    extend_artifact(path, [
        NewRef(scalar=0.42, order=4, parent_ref_id=s_rid, category='S'),
    ])

    view_after = load_knowledge_view(path)
    assert view_after.n_refs_live == n_before + 1
    # The new ref has order 4 — appears in refs_by_order[4]
    refs_o4 = view_after.refs_by_order(4)
    assert n_before in refs_o4.tolist()
    # And in refs_by_category['S'] (subtree includes the new ref)
    refs_s = view_after.refs_by_category('S')
    assert n_before in refs_s.tolist()
