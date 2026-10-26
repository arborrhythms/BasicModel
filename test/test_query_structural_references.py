"""Structural thought operands need live symbols, not numerical payload rows."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from Language import Grammar
from Meaning import ConceptualMeaning
from Queries import GrammaticalThoughtRegistry, _existing_row
from test_cs_symbol_table import _cs
from test_query_vp_boundaries import _context


def _request():
    cs = _cs()
    grammar = Grammar()
    grammar.configure({name: {'rule': [
        f'isEqual_argout1 = isEqual.{face}(isEqual_argin1, isEqual_argin2)',
        f'isPart_argout1 = isPart.{face}(isPart_argin1, isPart_argin2)',
    ]} for name, face in (('compose', 'forward'), ('thought', 'thought'))})
    registry = GrammaticalThoughtRegistry.install(cs, grammar)
    left, right = cs.new_concept(), cs.new_concept()
    inner = cs.reify_concept(left, right)
    alloc = cs._concept_allocator
    alloc.symbol_types.setdefault(inner, set()).add('object')
    alloc.symbol_types.setdefault(right, set()).add('object')
    alloc.singletons.add(right)
    witness = alloc.parts.add(inner, right, truth_plus=.7, truth_minus=.2)
    # The completed parse supplies held values beside the structural handles;
    # no fixture call seats the Part operand to form its thought request.
    vp = registry._reference(('symbolic-definition', 'isEqual'))
    request = ConceptualMeaning(
        torch.stack((torch.ones(8), registry._payload(vp), torch.ones(8))),
        torch.ones(3, dtype=torch.bool),
        role_refs=(('sym', inner), vp, ('sym', right)))
    return cs, registry, request, witness


def _snapshot(cs):
    alloc = cs._concept_allocator
    return dict(
        rows={name: value.clone() for name, value in cs._fixture_occurrences.named_buffers()},
        parts={name: value.clone() for name, value in alloc.parts.named_buffers()},
        bank=cs.similarity_codebook.W.detach().clone(),
        placement=dict(alloc.placement), retired=set(alloc.retired),
        tensor_rows=deepcopy(alloc._layers[0]._tensor_rows))


def _unchanged(cs, before):
    after = _snapshot(cs)
    for group in ('rows', 'parts'):
        assert before[group].keys() == after[group].keys()
        for name, value in before[group].items():
            torch.testing.assert_close(after[group][name], value, rtol=0, atol=0)
    torch.testing.assert_close(after['bank'], before['bank'], rtol=0, atol=0)
    for name in ('placement', 'retired', 'tensor_rows'):
        assert after[name] == before[name]


def test_unseated_part_operand_reaches_is_equal_without_writes():
    cs, registry, request, witness = _request()
    inner = request.role_refs[0][1]
    assert cs._concept_allocator.parts.contains(inner)
    with pytest.raises(ValueError, match='no allocated payload row'):
        _existing_row(cs, request.role_refs[0])
    before = _snapshot(cs)
    result = registry.execute(request, _context(cs, store=cs._fixture_occurrences))
    assert result.semantic_id == 'isEqual'
    assert result.support_true == pytest.approx(.7)
    assert result.support_false == pytest.approx(.2)
    assert result.evidence['witnesses'] == (('sym', witness),)
    _unchanged(cs, before)


def test_closed_is_part_menu_accepts_an_unseated_structural_operand():
    cs, registry, request, witness = _request()
    before = _snapshot(cs)
    candidates = registry.controller_candidates(request, request, request)
    selected = next(item for item in candidates if item.semantic_id == 'isPart')
    assert selected.request.role_refs[0] == request.role_refs[0]
    result = registry.execute(selected.request, _context(cs, store=cs._fixture_occurrences))
    assert result.evidence['witnesses'] == (('sym', witness),)
    assert result.support_true == pytest.approx(.7)
    assert result.support_false == pytest.approx(.2)
    _unchanged(cs, before)


@pytest.mark.parametrize('state', ('invalid', 'retired'))
def test_structural_operand_refuses_unavailable_handles_without_writes(state):
    cs, registry, request, _ = _request()
    if state == 'retired':
        cs.retire_concept(request.role_refs[0][1])
    else:
        request = replace(request, role_refs=(
            ('sym', cs._concept_allocator.next_id), *request.role_refs[1:]))
    before = _snapshot(cs)
    with pytest.raises(ValueError, match='reference is unavailable'):
        registry.execute(request, _context(cs, store=cs._fixture_occurrences))
    with pytest.raises(ValueError, match='reference is unavailable'):
        _existing_row(cs, request.role_refs[0])
    assert not any(item.semantic_id == 'isPart'
                   for item in registry.controller_candidates(request, request, request))
    _unchanged(cs, before)
