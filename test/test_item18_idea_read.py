"""Item 1.8 §68.54.4: the sentence idea teaches only its field-to-point read."""
from types import SimpleNamespace
import pytest
import torch


def test_idea_teacher_detaches_both_ends_and_only_steps_its_stack(tmp_path, monkeypatch):
    from test_item18_expect import lesson
    from IdeaRead import open_read, complete, expected_point
    from Layers import Error
    model = lesson(tmp_path, monkeypatch)
    try:
        owner = model._concept_owner()
        width = owner._order_caps()[0]
        source = torch.linspace(.1, .9, 4 * width).reshape(2, width, 2).requires_grad_()
        target = torch.full((2, owner.outputShape[-1]), .4, requires_grad=True)
        open_read(model, source)
        registry = Error()
        complete(model, SimpleNamespace(meaning_root=target), torch.tensor([True, False]), registry)
        assert set(registry._terms) == {'expectation.idea'}
        optimizer = model.getOptimizer(lr=.003)
        groups = model.objective_parameter_groups(optimizer)
        own = tuple(owner.idea_stack.parameters())
        assert set(own) <= set(groups['expectation'])
        before = [p.detach().clone() for p in own]
        registry.total().backward()
        assert source.grad is None and target.grad is None
        assert any(p.grad is not None and bool(p.grad.any()) for p in own)
        assert all(p.grad is None for p in model.parameters() if id(p) not in {id(q) for q in own})
        optimizer.step()
        assert any(not torch.equal(a, b) for a, b in zip(before, own))
        assert expected_point(owner, streams=1).shape == (1, owner.outputShape[-1])
    finally:
        model.End(); model.symbolSpace.soft_reset()


def test_native_kept_sentence_books_the_observed_field_and_semantic_root(tmp_path, monkeypatch):
    from test_item18_expect import lesson, read
    import IdeaRead
    model = lesson(tmp_path, monkeypatch)
    # MTM nominates a form on recurrence (sigma-pi §21, rule 20).
    from nomination_fixtures import nominated_sentences; nominated_sentences(model, ['alpha gamma'])
    calls = []
    complete = IdeaRead.complete
    def capture(host, record, present, registry):
        owner = host._concept_owner()
        prediction = owner.idea_stack(host._idea_observation.transpose(-1, -2))
        target = record.meaning_root.detach()
        value = complete(host, record, present, registry)
        expected = (prediction - target).square().mean(-1)[present].mean()
        term = registry._value(registry._terms['expectation.idea'])
        torch.testing.assert_close(term.sum(), expected)
        calls.append((present.clone(), target.clone()))
        return value
    monkeypatch.setattr(IdeaRead, 'complete', capture)
    try:
        read(model, 'alpha gamma', training=True, optimizer=model.getOptimizer(lr=.003))
        assert len(calls) == 1 and calls[0][0].tolist() == [True]
        assert 'expectation.idea' in model._sentence_cost_registry._terms
    finally:
        model.End(); model.symbolSpace.soft_reset()


def test_only_sentence_concept_owners_build_the_read():
    from test_math_chain import build_model
    from recon_bench import _build_model
    sentence = build_model()
    try:
        owner = sentence._concept_owner()
        for space in sentence.conceptualSpaces:
            assert (space.idea_stack is not None) == (space is owner)
        assert owner.idea_stack.hidden == 64 and owner.idea_stack.pairs == 1
        assert owner.idea_stack.outputs == owner.outputShape[-1]
    finally:
        sentence.End(); sentence.symbolSpace.soft_reset()
    numeric, *_ = _build_model('data/MM_xor.xml')
    try:
        assert all(space.idea_stack is None for space in numeric.conceptualSpaces)
    finally:
        numeric.End(); numeric.symbolSpace.soft_reset()
