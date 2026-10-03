"""Reading derives word concepts and symbols under both supported bindings."""
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize('configuration', ['MM_xor.xml', 'XOR_grammar.xml', 'aligned'])
def test_every_read_word_has_a_concept_and_symbol(tmp_path, monkeypatch, configuration):
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    if configuration == 'aligned':
        from test_compiled_word_chunk import _tiny_canonical_model
        model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8')
        model.reconstruct_in_loop = False
        model.loss.reconstruction_scale = 0.
        model._tensor_peer_while_eager = True
        model._chart_compose_per_word = lambda: None
    else:
        from test_mm_xor import _fresh_model
        model, _, _ = _fresh_model(str(Path(__file__).resolve().parents[1] / 'data' / configuration))
    model.eval()
    sentences = ['hi no', 'hi go', 'we no', 'we go']
    try:
        inputs = model.inputSpace.prepInput(sentences)
        with torch.no_grad():
            _, symbols, _, _ = model(inputs)
        owner = model._concept_owner()
        words = sorted(set(' '.join(sentences).split()))
        assert all(owner.word_concepts(word) for word in words), {
            word: owner.word_concepts(word) for word in words}
        assert torch.isfinite(symbols).all()
        assert symbols.abs().sum() > 0, 'derived concepts must have symbol activations'
        if configuration != 'aligned':
            from ConceptEvidence import decode
            carrier = model.symbol_cache
            evidence = carrier._concept_activations
            assert evidence.any(), 'symbol bands cannot stand in for concept evidence'
            expected = decode(evidence, carrier._concept_codes)
            torch.testing.assert_close(symbols[..., :expected.shape[-1]], expected)
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()


def test_word_publication_preserves_the_composed_concept_field():
    from types import SimpleNamespace
    from Models import BasicModel
    from ConceptEvidence import decode
    # A completed symbolic phase may know both poles and higher-order rows.
    # Publishing symbols must not replace that result with a new native read.
    evidence = torch.tensor([[[[1., 0.]]], [[[0., 1.]]], [[[1., 1.]]]])
    codes = torch.eye(3)
    carrier = SimpleNamespace(_concept_activations=evidence, _concept_codes=codes)
    def read_again(*args):
        pytest.fail('completed symbolic evidence was replaced by word admission')
    owner = SimpleNamespace(_sparse_active=lambda: True, cs_read_memberships=read_again)
    host = SimpleNamespace(_reading_word_percepts=object(), _reading_word_extents=None,
        _concept_owner=lambda: owner,
        symbolSpace=SimpleNamespace(forward_concept_to_symbol=lambda value:
            decode(value._concept_activations, value._concept_codes)))
    result = BasicModel._publish_reading_symbols(host, carrier)
    torch.testing.assert_close(result, decode(evidence, codes), atol=0, rtol=0)


from functools import wraps

import pytest

import torch


def test_default_interpret_reuses_existing_kind_without_minting():
    from test_word_interpretation import _operator
    from Spaces import _concept_alloc_of
    cs, interpret = _operator()
    word = interpret.lookup_word([7], [], form='cat')
    kind = interpret.forward(word, order=2)
    before = dict(_concept_alloc_of(cs).placement)
    assert interpret.forward(word) == kind
    assert interpret.forward(word, order=1) == kind
    assert dict(_concept_alloc_of(cs).placement) == before


def test_default_interpret_reuses_a_witnessed_association():
    from test_word_interpretation import _operator
    from Spaces import _concept_alloc_of
    cs, interpret = _operator()
    word = interpret.lookup_word([7], [], form='cat')
    kind = cs.new_concept()
    _concept_alloc_of(cs).reference_orders[kind] = 2
    interpret.define(word, kind)
    cs.bind_word_concept('cat', kind)
    assert interpret.forward(word) == kind
    assert interpret.reverse(kind) == word



"""Word-to-object interpretation owns testimony, resolution, and lexicalization."""
import torch

from test_cs_sparse_weights import _cs
from Spaces import _concept_alloc_of


def _operator():
    from Language import InterpretLayer
    cs = _cs(nS=64, order=3)
    return cs, cs.interpret


def test_interpret_reuses_definition_and_reverses_by_identity():
    cs, interpret = _operator()
    word = interpret.lookup_word([7], [1], form='wug')
    obj = interpret.forward(word, occurrence=(0, 1))
    assert obj != word
    assert ('sym', word) not in cs.concept_parts(obj)
    assert cs._concept_source_order(obj) == 0
    row = cs._csw_row_of(obj)
    assert cs._csw_row_of(word) is None
    before = len(cs.definitions._store())
    for sentence in range(1, 20):
        assert interpret.forward(word, occurrence=(sentence, 1)) == obj
        interpret.boundary()
    assert len(cs.definitions._store()) == before
    assert interpret.reverse(obj) == word
    assert cs._csw_row_of(obj) == row


def test_grammar_resolution_distinguishes_particular_and_kind():
    cs, interpret = _operator()
    word = interpret.lookup_word([7], [1], form='cat')
    particular = interpret.forward(word, order=1)
    kind = cs.new_concept()
    _concept_alloc_of(cs).reference_orders[kind] = 2
    interpret.define(word, kind)
    cs.bind_word_concept('cat', kind)
    assert particular != kind
    assert cs._concept_source_order(particular) == 0
    assert cs._concept_source_order(kind) == 2
    assert interpret.forward(word, order=2) == kind
    assert interpret.reverse(particular) == interpret.reverse(kind) == word
    assert interpret.forward(word, order=0) == particular
    assert interpret.forward(word, order=2) == kind


def test_tensor_face_replaces_content_and_preserves_occurrence_coordinates():
    _, interpret = _operator()
    word = torch.tensor([[1., 2., 3., 4., 8., 9.]])
    obj = torch.tensor([[5., 6., 7., 8.]])
    activation = torch.tensor([.5])
    result = interpret.forward(word, object_atoms=obj, activation=activation)
    torch.testing.assert_close(result, torch.tensor([[2.5, 3., 3.5, 4., 8., 9.]]))




def test_ordered_word_lookup_keeps_anagrams_distinct():
    cs, interpret = _operator()
    ab = interpret.lookup_word([1, 2], [], form='ab')
    ba = interpret.lookup_word([2, 1], [], form='ba')
    assert ab != ba
    assert interpret.lookup_word([1, 2], [], form='ab') == ab




@pytest.mark.usefixtures('eager_reading')
def test_every_serial_word_runs_interpret_before_composition(tmp_path, monkeypatch):
    from test_packed_reconstruction_parity import build_model
    model = build_model(tmp_path, word_capacity=8)
    owner = model._concept_owner()
    operator = owner.interpret
    calls = []
    # Run the identical recurrence bodies eagerly so instrumentation can
    # observe ordering. The normal compiled schedule has separate parity tests.
    def eager_loop(cond, body, carries):
        while bool(cond(*carries)):
            carries = body(*carries)
        return carries
    monkeypatch.setattr(torch, 'while_loop', eager_loop)
    for space in model.conceptualSpaces:
        forward = space.interpret.forward
        def observed(word, _forward=forward, **kwargs):
            result = _forward(word, **kwargs)
            if torch.is_tensor(word):
                calls.append(result.detach().clone())
            return result
        monkeypatch.setattr(space.interpret, 'forward', observed)
    choose = model.languageSpace.choose_operation
    def composition(*args, **kwargs):
        assert calls, 'composition ran before the mandatory interpretation'
        return choose(*args, **kwargs)
    monkeypatch.setattr(model.languageSpace, 'choose_operation', composition)
    with torch.no_grad():
        model.understand(model.inputSpace.prepInput(['the wug sat']))
    ids = model.inputSpace._ar_word_concept_ids
    objects = model.inputSpace._ar_word_object_ids
    active = model.inputSpace._word_active_mask
    assert len(calls) >= int(active.sum())
    assert (ids[active] != objects[active]).all()
    for word, obj in zip(ids[active].tolist(), objects[active].tolist()):
        assert operator.reverse(obj) == word
    model.End()




def test_selected_generic_grammar_ends_the_interpreted_kinds(monkeypatch):
    """The selected generic reading of 'cats are animals' owns kind refs.

    This tests the declared grammar interface, not untrained English parsing.
    The closing receives resolved objects and never performs lexical resolution.
    """
    from dataclasses import replace
    from Layers import TernaryTruthStore
    from Models import _append_observed_meaning
    from test_selected_relation_meaning import _program_owner
    cs, _, registry, language, _, make_program, _, _ = _program_owner(monkeypatch)
    words = [cs.interpret.lookup_word([part], [], form=form)
             for part, form in ((7, 'cats'), (8, 'animals'))]
    objects = [cs.interpret.forward(word, order=1) for word in words]
    # Grammar resolves existing associations; it cannot mint a second object
    # merely because the first has a different order.
    for form in ('cats', 'animals'):
        kind = cs.new_concept()
        _concept_alloc_of(cs).reference_orders[kind] = 2
        cs._csw_concept_row(2, kind)  # the supplied kind owns a payload row
        cs.interpret.define(cs.definitions.word(form=form), kind)
        cs.bind_word_concept(form, kind)
    leaves = torch.stack([registry._payload(('sym', obj)) for obj in objects])
    entry = make_program(values=leaves, refs=tuple(('sym', obj) for obj in objects))
    rules = list(language._compose_binary_rules)
    selected = int(entry.actions[-1, 1])
    rules[selected] = rules[selected]._replace(reference_orders=(('I1', 2), ('I2', 2)))
    language._compose_binary_rules = rules
    word_rows = torch.tensor([cs._csw_row_of(obj) for obj in objects])
    refs, orders = language.resolve_lexical_references(cs, word_rows, entry.concept_ids, entry.actions)
    assert orders.tolist() == [2, 2]
    assert refs.tolist() == [cs.interpret.forward(word, order=2) for word in words]
    assert all(int(kind) != particular for kind, particular in zip(refs, objects))
    entry = replace(entry, word_rows=word_rows, reference_ids=refs, reference_orders=orders)
    meaning = language.program_meaning(entry, registry)
    assert meaning.role_refs[0] == ('sym', int(refs[0]))
    assert meaning.role_refs[2] == ('sym', int(refs[1]))
    store = TernaryTruthStore(leaves.shape[-1], capacity=8)
    from ClauseRow import attach_clause_index
    from types import SimpleNamespace
    host = SimpleNamespace(languageSpace=language, grammatical_thoughts=registry)
    attach_clause_index(host, store, cs)
    from reading_fixtures import finish_reading
    row = _append_observed_meaning(store, finish_reading(language, entry, registry=registry), trust=.9)
    assert store.row(row)['meaning'].role_refs == meaning.role_refs
    for word, kind in zip(words, refs.tolist()):
        assert cs.interpret.reverse(kind) == word
    torch.testing.assert_close(entry.leaves, leaves)


def test_testimony_and_its_later_definition_are_read_by_the_parallel_field():
    cs, interpret = _operator()
    cs._concept_binding = 'aligned'
    cs._serial = True
    word = interpret.lookup_word([7], [], form='wug')
    obj = interpret.forward(word, occurrence=(0, 0))
    for tick in range(1, 20):
        assert interpret.forward(word, occurrence=(tick, 0)) == obj
        interpret.boundary()
    row = cs._csw_row_of(obj)
    original = tuple(cs.concept_parts(obj))
    cs._serial = False  # witness the later definition through the parallel field
    witnessed = interpret.forward(interpret.lookup_word([8], [], form='witnessed'))
    cs.assert_concept_relation(obj, sym_part=witnessed)
    assert all(part in cs.concept_parts(obj) for part in original)
    assert ('sym', witnessed) in cs.concept_parts(obj)
    before = tuple(cs.concept_parts(obj))
    cs._serial = False
    spans = torch.tensor([[[0, 1], [1, 2]]])
    bracket = torch.tensor([[[0, 2]]])
    read = cs.cs_read_memberships((torch.tensor([[7, 8]]), spans, None,
                                  torch.tensor([[65, 65]]), spans), bracket)
    _, field = cs.cs_forward_content(read, cs.similarity_codebook.getW())
    slot = (cs._cs_field_concept_ids == obj).nonzero().flatten().item()
    assert field[slot, 0, 0, 0] > 0
    assert cs._csw_row_of(obj) == row and tuple(cs.concept_parts(obj)) == before
    assert interpret.reverse(obj) == word
