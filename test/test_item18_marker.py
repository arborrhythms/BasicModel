"""Round B's learned unary family, separate from binary case selection."""
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.assessment_rung(3)

from test_meaning_kernels import paired


def test_86_marker_preserves_form_and_has_an_exact_selected_inverse(paired):
    from Language import MarkerLayer
    from Queries import StructuralGrammarContext, ConceptualSpaceCapability
    family = MarkerLayer().double()
    assert len(family.candidates) == 8
    value = torch.linspace(.1, .85, 48, dtype=torch.float64).reshape(2, 2, 12)
    context = StructuralGrammarContext((), ConceptualSpaceCapability(12), None, 'compose')
    for index, candidate in enumerate(family.candidates):
        assert candidate.arity == 1 and candidate.kernel_index == index
        assert not hasattr(candidate, 'compose_case_meaning') and not candidate.case_head_role
        assert torch.equal(candidate.compose(value), value)
        # No case view or whole-form centroid is consulted by this operation.
        with torch.no_grad():
            for stage in candidate.stack.stages:
                stage.layer.W.zero_()
                stage.layer.W[0, 0] = 1.
            final = candidate.stack.stages[-1].layer.W
            final[0, :8] = .25
            final[0, 8:] = .05
        actual = candidate.compose_from_grammar_context((value,), context=context)
        assert torch.equal(actual[..., :4], value[..., :4])
        assert not torch.equal(actual[..., 4:], value[..., 4:])
        torch.testing.assert_close(candidate.generate(actual), value, rtol=0, atol=1e-6)
        actual.sum().backward()
        assert candidate.stack.stages[-1].layer.W.grad.abs().sum() > 0


def test_86_compose_and_generate_choose_each_kernel(tmp_path, monkeypatch):
    from Language import MarkerLayer
    from identity_experiment import make_model
    model = make_model(tmp_path, monkeypatch)
    language = model.languageSpace
    compose = [op.gl for op in language.language_layer.operation_layer.unary_ops
               if op.gl.rule_name == 'marker']
    generate = [op for op in language._generate_unary_ops if op.rule_name == 'marker']
    assert len(compose) == len(generate) == 8
    assert [op.kernel_index for op in compose] == list(range(8))
    assert all(first is second for first, second in zip(compose, generate))
    rules = [rule for rule in language._generate_unary_rules if rule.method_name == 'marker']
    assert [rule.kernel_index for rule in rules] == list(range(8))
    assert len({language._generate_rule_key(rule, 1) for rule in rules}) == 8
    assert 'marker' not in language._surface_anchors.values()
    from Language import TheGrammar
    assert len([r for r in TheGrammar.rules_upward if r.method_name == 'marker']) == 1
    # A family has one grammatical identity and one set of argument concepts.
    assert language.parts_of_speech.arguments.count('marker_argin1') == 1
    width = model._concept_owner().similarity_codebook.W.shape[-1]
    values = torch.linspace(.1, .85, width).expand(8, -1)
    with torch.no_grad():
        for index, candidate in enumerate(compose):
            for stage in candidate.stack.stages:
                stage.layer.W.zero_(); stage.layer.W[0, 0] = 1.
            final = candidate.stack.stages[-1].layer.W
            final.zero_(); final[0, :final.shape[1]//2] = .05 * (index+1)
    parents = torch.cat([op.compose(values[i:i+1]) for i,op in enumerate(compose)])
    assert not torch.equal(parents, values)
    selected, active = torch.arange(8), torch.ones(8, dtype=torch.bool)
    inverse = lambda x: language.generate_unary_step(x, selected, active, ops=generate)
    torch.testing.assert_close(inverse(parents), values, rtol=0, atol=1e-6)
    compiled = torch.compile(inverse, backend='eager', fullgraph=True)
    torch.testing.assert_close(compiled(parents), values, rtol=0, atol=1e-6)
    model.End(); model.symbolSpace.soft_reset()


def test_86_retired_unary_case_has_no_grammar_and_compound_stays():
    from Language import Grammar, GRAMMAR_LAYER_CLASSES, CompoundLayer
    assert 'compound_unary' not in GRAMMAR_LAYER_CLASSES
    for root in ('data', 'test/fixtures'):
        for path in Path(root).rglob('*.grammar'):
            assert 'compound_unary' not in path.read_text(), path
    grammar = Grammar()
    grammar.load_from_grammar_file('complete.grammar')
    assert len([rule for rule in grammar.rules_upward if rule.method_name == 'compound']) == 1
    assert callable(CompoundLayer.compose_case_meaning)



def test_86_pinned_sentence_keeps_the_selected_kernel(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from identity_experiment import make_model
    from SentenceStrata import SentenceStrata
    model = make_model(tmp_path, monkeypatch)
    language, owner = model.languageSpace, model._concept_owner()
    width = owner.similarity_codebook.W.shape[-1]
    bank = SentenceStrata(1, 1, 1, width)
    value = torch.linspace(.1, .8, width)[None]
    word = owner.new_concept(order=0)
    try:
        found = []
        for index, rule in enumerate(language._compose_unary_rules):
            if rule.method_name != 'marker':
                continue
            actions = torch.tensor([[0, -1, 0], [2, index, 0]])
            operation = language._tree_layer(1).unary_ops[index].gl
            frame = torch.stack((value[0], value[0], operation.compose(value)[0]))
            program = SimpleNamespace(actions=actions, leaves=value, concept_ids=torch.tensor([word]),
                operation_values=torch.stack((torch.zeros_like(frame), frame)))
            tree = bank.capture(program, language, 0).tree
            torch.testing.assert_close(tree.actions(), actions, rtol=0, atol=0)
            assert tree.forest() == ((2, index, 1, (0, 0)),)
            found.append(rule.kernel_index)
            bank.clear()
        assert found == list(range(8))
    finally:
        model.End(); model.symbolSpace.soft_reset()
