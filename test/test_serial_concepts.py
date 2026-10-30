"""The serial open pass reads every present bracket before composition."""
from types import SimpleNamespace

import torch

from FamiliarTier import open_sentence as observe
from Spaces import _concept_alloc_of
from test_attention_promotion import _fixture, _candidates


def fixture():
    cs, rows = _fixture()
    cs._serial = True
    for index, (_cid, row) in enumerate(rows[:3]):
        cs.add_concept_feature(row, 'ps', index, 1.)
    # Two packed sentences, each with one alternative and the common context.
    spans = torch.tensor([[[0, 1], [2, 3], [4, 5], [6, 7]]])
    native = dict(indices=torch.tensor([[0, 2, 1, 2]]), part_spans=spans)
    model = SimpleNamespace(training=True, _concept_owner=lambda: cs,
        inputSpace=SimpleNamespace(_packed_sentence_ids=torch.tensor([[0, 0, 1, 1]]),
                                  _word_active_mask=torch.ones(1, 4, dtype=torch.bool)),
        perceptualSpace=SimpleNamespace(_forward_input=native),
        wholeSpace=SimpleNamespace(subspace=SimpleNamespace(what=SimpleNamespace(primitive_properties=None)),
                                   _staged_word_property_spans=torch.zeros(1, 4, 1, 2, dtype=torch.long)),
        _attention_spans=spans, _attention_words=None,
        _staged_concepts_in=torch.tensor([[[120, 32, 99, 32, 121, 32, 99]]]))
    return cs, rows, model


def test_open_pass_reads_once_before_composition_and_never_mints(monkeypatch):
    cs, rows, model = fixture()
    calls = []
    read, forward = cs.cs_read_memberships, cs.cs_forward_content
    monkeypatch.setattr(cs, 'cs_read_memberships', lambda *a, **k: (calls.append('read'), read(*a, **k))[1])
    monkeypatch.setattr(cs, 'cs_forward_content', lambda *a, **k: (calls.append('forward'), forward(*a, **k))[1])
    before = cs._concept_allocator.next_id
    first = observe(model, 0, torch.tensor([True]))
    assert calls == ['read', 'forward']
    assert set(first[0]) == {rows[0][1], rows[2][1]}
    assert len(_candidates(cs)) == 0
    assert cs._concept_allocator.next_id == before
    assert not model._last_open_attention['brackets'][0, 2:].any()


def test_open_pass_ignores_later_attention_narrowing_and_respects_inactive_streams():
    cs, rows, model = fixture()
    model._attention_words = SimpleNamespace(accepted=torch.tensor([[True, False, False, False]]))
    observe(model, 0, torch.tensor([True]))
    assert model._last_open_attention['brackets'][0, 1].tolist() == [2, 3]
    before = cs.mtm.candidate_counter
    seen = observe(model, 1, torch.tensor([False]))
    assert seen == {0: []} and cs.mtm.candidate_counter == before
    assert not model._last_open_attention['brackets'].any()


def test_update_has_one_promotion_after_placement_and_no_close_observer():
    import ast, inspect, textwrap
    from Models import BasicModel
    text = inspect.getsource(BasicModel._commit_sentence)
    tree = ast.parse(textwrap.dedent(text))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == 'promote_candidates']
    assert len(calls) == 1
    assert 'if bool(active.any()):' in text
    assert text.index('consolidate(owner') < text.index('mtm.promote_candidates') < text.index('derive_class_placements')
    assert 'SerialConcepts' not in text


def test_prime_boundary_joins_lexical_and_open_field_rows_in_one_step(monkeypatch):
    import pytest
    import Priming
    from Models import BasicModel
    cs, rows, model = fixture()
    lexical = rows[4][1]
    model.inputSpace._ar_concept_lookup_rows = torch.tensor([[lexical]])
    model.inputSpace._ar_concept_lookup_sentence_ids = torch.tensor([[0]])
    model.inputSpace._packed_sentence_slot_mask = torch.tensor([[True]])
    calls = []
    propagate = Priming.propagate
    def step(owner, surface, active):
        calls.append('propagate')
        return propagate(owner,surface,active)
    monkeypatch.setattr(Priming,'propagate',step)
    prime = cs.prime_seen
    class ReachedReconstructionBank(Exception):
        pass
    def primed(seen, **kwargs):
        assert set(seen[0].tolist()) == {lexical,rows[0][1],rows[2][1]}
        assert kwargs['propagate'] and kwargs['active_rows'].tolist() == [True]
        prime(seen,**kwargs)
        raise ReachedReconstructionBank
    monkeypatch.setattr(cs,'prime_seen',primed)
    before = cs._concept_allocator.next_id
    with pytest.raises(ReachedReconstructionBank):
        BasicModel._prime_sentence_symbols(model,0)
    assert calls == ['propagate']
    assert cs._concept_allocator.next_id == before and len(_candidates(cs)) == 0
