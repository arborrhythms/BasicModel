"""Grammar read-back chooses words from the ended state, with a detached hard inverse."""
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
    def observe(state, sid, active, *args):
        nonlocal readback
        readback = True
        texts, unavailable = model.reconstruct_grammar_sentence(state, sid, active)
        assert len(texts) == 4 and unavailable.shape == (4,)
        assert parents, 'the reading must traverse the grammar inverse'
        before = parents[0]
        changed = list(state[1])
        changed[9], changed[13] = changed[9] + 10, changed[13] + 10
        parents.clear()
        model.reconstruct_grammar_sentence((state[0], tuple(changed), state[2]), sid, active)
        assert parents and not torch.equal(parents[0], before)
        assert model._word_symbol_concept_ids() is None
        decoded.append(texts)
        readback = False
        return commit(state, sid, active, *args)
    monkeypatch.setattr(model, '_commit_sentence', observe)
    try:
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
        assert len(model._grammar_gate_reconstructions) == 4
        assert len(model._grammar_gate_unavailable) == 4
        assert not bool(model._reconstruction_stack()._choice_mask.any())
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()
