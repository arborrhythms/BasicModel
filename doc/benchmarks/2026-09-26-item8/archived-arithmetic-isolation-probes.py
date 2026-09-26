"""Arithmetic corpus helpers cannot supply learner computation or answer seeds."""
import pytest
import torch
from types import SimpleNamespace

from What import What
from test_output_path_supervised import _native_answer_model
from test_output_walk import _capture_program_probe, _stop


def _poison_oracles(monkeypatch):
    import exact

    def forbidden(*args, **kwargs):
        pytest.fail("oracle arithmetic reached from the learner")

    # Corpus construction precedes this boundary. Patch both the module
    # evaluators and scratchpad entry points, including the old answer seed.
    for name in ("eval_expr", "linearize", "numeral_code", "referent_code"):
        monkeypatch.setattr(exact, name, forbidden)
    for name in ("from_surface", "lookup", "evaluate", "bind", "substitute",
                 "constrain", "execute"):
        monkeypatch.setattr(exact.ExactState, name, forbidden)
    monkeypatch.setattr(exact.ExactLexer, "lex", forbidden)


def test_taxonomy_query_uses_relations_and_payloads_under_native_id_renaming(monkeypatch):
    from Language import Grammar
    from Queries import GrammaticalQueryRegistry, QueryContext
    from reasoning import TruthGroundedReasoner
    from test_cs_symbol_table import _cs

    def world(prefix):
        space = _cs()
        for _ in range(prefix):
            space.new_concept()
        grammar = Grammar()
        grammar.load_from_grammar_file('complete.grammar')
        registry = GrammaticalQueryRegistry.install(space, grammar)
        left, right = (('sym', space.new_concept()) for _ in range(2))
        for ref in (left, right):
            space._csw_concept_row(0, ref[1])
        space.add_whole(left[1], right)
        context = QueryContext(TruthGroundedReasoner(
            model=SimpleNamespace(conceptualSpace=space)))
        return space, registry, left, right, context

    first, second = world(0), world(11)
    x, y = first[1].form('isPart', first[2], first[3]), second[1].form('isPart', second[2], second[3])
    assert all(a != b for a, b in zip(x.role_refs, y.role_refs))
    with torch.no_grad():
        for payload, reference in zip(x.roles, y.role_refs):
            row = second[0]._csw_concept_row(0, reference[1])
            second[0].similarity_codebook.getW()[row].copy_(payload)
    y = second[1].form('isPart', second[2], second[3])
    torch.testing.assert_close(x.roles, y.roles)
    _poison_oracles(monkeypatch)
    a, b = first[1].execute(x, first[4]), second[1].execute(y, second[4])
    assert a['support_true'] == b['support_true'] == 1
    assert a['support_false'] == b['support_false'] == 0
    reverse = second[1].execute(second[1].form('isPart', second[3], second[2]), second[4])
    assert reverse['support_true'] == 0


def test_numeric_sentence_paths_use_learned_representations_with_oracles_poisoned(tmp_path, monkeypatch):
    model = _native_answer_model(tmp_path, True)
    assert model.inputSpace.data.math_problems["train"]
    _poison_oracles(monkeypatch)
    model.eval()
    _stop(model)
    try:
        with torch.no_grad():
            understood = _capture_program_probe(model, ["1 plus 2", "3 plus 4"])
            reconstructed = model.reverseReconstruct(understood)
            resolved = model.resolveAnswer(understood, (What.supervised(0), What.supervised(1)))
            answer = model.reverseOutput(understood, resolved)
        assert all(program is not None for program in understood.answer_program)
        assert bool(torch.isfinite(answer.actual).all())
        assert not torch.equal(answer.concepts[0], answer.concepts[1])
        assert reconstructed is not None
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()


@pytest.mark.slow
def test_supervised_training_cannot_use_exact_arithmetic_or_fallback_codes(tmp_path, monkeypatch):
    model = _native_answer_model(tmp_path, True)
    assert model.inputSpace.data.math_problems["train"]
    _poison_oracles(monkeypatch)
    optimizer = model.getOptimizer(lr=1e-3)
    steps = []
    real_step = optimizer.step

    def step(*args, **kwargs):
        steps.append(1)
        return real_step(*args, **kwargs)

    monkeypatch.setattr(optimizer, "step", step)
    try:
        batch = (model.inputSpace.prepInput(["1 plus 2", "3 plus 4"]),
                 torch.zeros(2, 1, 1))
        result, _ = model.runBatch(train=True, batchSize=2, split="train",
            optimizer=optimizer, batch_override=batch,
            questions=(What.supervised(0), What.supervised(1)))
        assert steps == [1]
        assert bool(torch.isfinite(result.lossOut))
        assert model._last_answer_mask.tolist() == [True, True]
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()
