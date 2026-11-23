"""Grammar read-back chooses words from the ended state, with a detached hard inverse."""
import pytest
import torch


def test_symmetric_reverse_returns_one_least_residual_pair_without_search_gradient():
    from Language import LanguageSpace, ConjunctionLayer
    operation = ConjunctionLayer()
    basis = torch.tensor([[[.2, 0.], [0., .2]]])
    parent = torch.tensor([[-.03, .02]], requires_grad=True)
    left, right, available = LanguageSpace._bounded_binary_reconstruction(
        operation, parent, torch.zeros_like(parent), torch.tensor([False]),
        torch.tensor([False]), basis, torch.ones(1, 2, dtype=torch.bool), 2)
    assert available.tolist() == [True]
    pair = torch.stack((left[0], right[0]))
    assert torch.equal(pair, basis[0]) or torch.equal(pair, basis[0].flip(0))
    assert not left.requires_grad and not right.requires_grad
    assert parent.grad is None


def test_grammar_reconstruction_reads_the_concluded_state_before_trace_disposal(monkeypatch):
    from pathlib import Path
    import Models
    import util
    from test_mm_xor import _fresh_model
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    # The subject is the inverse traversal, not graph capture.
    def eager_while(condition, body, values):
        while bool(condition(*values)):
            values = body(*values)
        return values
    monkeypatch.setattr(torch, 'while_loop', eager_while)
    model, _, data = _fresh_model(str(Path(Models.__file__).resolve().parents[1] / 'data/XOR_grammar.xml'))
    commit = model._commit_sentence
    inverse = model.languageSpace.reverse_binary_step
    parents, decoded = [], []
    readback = False
    def reverse(parent, *args, **kwargs):
        # Training and the gate use the same free read-back, with neither
        # a known operand nor witness offsets.
        if readback and bool(args[1].any()):
            parents.append(parent.detach().clone())
            assert kwargs.get('reference') is None
            assert 'reference_side' not in kwargs
        return inverse(parent, *args, **kwargs)
    monkeypatch.setattr(model.languageSpace, 'reverse_binary_step', reverse)
    import SentenceField
    steps, followed = SentenceField.taken_steps, []
    def taken(model_, record, programs):
        result = steps(model_, record, programs)
        followed.append(result[4][1].any(-1).tolist())
        return result
    monkeypatch.setattr(SentenceField, 'taken_steps', taken)
    def observe(state, sid, active, *args):
        nonlocal readback
        readback = True
        texts, unavailable = model.reconstruct_grammar_sentence(state, sid, active)
        assert len(texts) == 4 and unavailable.shape == (4,)
        # Ported (nineteenth pass, sigma-pi 19.140): a reading whose tree is
        # in the trial's STM is put back by following its edges there, each
        # word said by the concept its row names, with no inverse searched; a
        # reading with no edges is still the grammar's inverse of its state.
        rows = followed[-1]
        assert parents or any(rows), 'the reading must traverse the grammar inverse or its edges'
        before = parents[0] if parents else None
        changed = list(state[1])
        changed[6], changed[10] = (changed[6] + .5) / 2, (changed[10] + .5) / 2
        current = list(state[0])
        current[0] = (current[0] + .5) / 2
        parents.clear()
        again, _ = model.reconstruct_grammar_sentence((tuple(current), tuple(changed), state[2]), sid, active)
        # What the STM's tree holds does not follow the state's values: a
        # followed reading is put back the same from a changed state.
        assert all(again[b] == texts[b] for b, row in enumerate(rows) if row)
        if before is not None:
            # A searched reading's inverse follows the changed state.
            assert parents and not torch.equal(parents[0], before)
        from PrependedSymbols import word_view
        torch.testing.assert_close(model._word_symbol_concept_ids(),
                                   word_view(model.inputSpace, 'concept_ids'))
        decoded.append(texts)
        readback = False
        return commit(state, sid, active, *args)
    monkeypatch.setattr(model, '_commit_sentence', observe)
    try:
        # MTM nominates a form on recurrence (sigma-pi §21, rule 20).
        from nomination_fixtures import nominated_forms
        nominated_forms(model, list(data.test_input))
        with torch.no_grad():
            model.forward(model.inputSpace.prepInput(list(data.test_input)))
        assert len(decoded) == 1
        assert not bool(model._reconstruction_stack()._choice_mask.any())
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()


def test_gate_captures_grammar_words_without_a_second_forward(monkeypatch):
    from pathlib import Path
    import Models
    import util
    import test_explicit_dimensions as gates
    from test_mm_xor import _fresh_model
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model, _, data = _fresh_model(str(Path(Models.__file__).resolve().parents[1] / 'data/XOR_grammar.xml'))
    calls = []
    def evaluate(config):
        calls.append(config)
        with torch.no_grad():
            model.forward(model.inputSpace.prepInput(list(data.test_input)))
        return [('probe', (), model)]
    monkeypatch.setattr(Models.ModelFactory, 'run', evaluate)
    try:
        assert gates._run_xor_grammar_in_process() is model
        assert calls == ['data/XOR_grammar.xml']
        assert len(model._grammar_gate_arrivals) == 4
        assert len(model._grammar_gate_reconstructions) == 4
        assert len(model._grammar_gate_unavailable) == 4
        assert not bool(model._reconstruction_stack()._choice_mask.any())
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()


@pytest.mark.parametrize('boundary', ('grammar input', 'grammar inverse'))
@pytest.mark.parametrize('wrong', (('world', 'hello'), ('hello',), ('hello', 'hello')),
                         ids=('swapped', 'omitted', 'duplicated'))
def test_xor_order_gate_detects_a_single_bad_pair_at_either_boundary(boundary, wrong):
    from test_explicit_dimensions import _assert_xor_word_order
    expected = [(first, second) for first in ('hello', 'loving') for second in ('world', 'there')]
    arrivals, returned = list(expected), list(expected)
    _assert_xor_word_order(expected, arrivals, returned, [False] * 4)
    observed = arrivals if boundary == 'grammar input' else returned
    observed[0] = wrong
    with pytest.raises(pytest.fail.Exception, match=f'{boundary}: 3/4 in order'):
        _assert_xor_word_order(expected, arrivals, returned, [False] * 4)
