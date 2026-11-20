"""Reading derives word concepts and symbols under both supported bindings."""
from order_fixtures import force_order
from pathlib import Path

import pytest
import torch
from nomination_fixtures import recurrent_word, readings_before_nomination


@pytest.mark.parametrize('configuration', ['MM_text.xml', 'XOR_grammar.xml', 'aligned'])
def test_every_read_word_has_a_concept_and_symbol(tmp_path, monkeypatch, configuration):
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    if configuration == 'aligned':
        from test_compiled_word_chunk import _tiny_canonical_model
        model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8', concept_rows=256, batch_size=4)
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
        # A form is a word from the reading in which MTM nominates it.
        readings_before_nomination(model, lambda: model(inputs))
        with torch.no_grad():
            _, symbols, _, _ = model(inputs)
        owner = model._concept_owner()
        words = sorted(set(' '.join(sentences).split()))
        assert all(owner.word_concepts(word) for word in words), {
            word: owner.word_concepts(word) for word in words}
        # Word admission supplies the leaves. Their operator's final set can
        # be empty (e.g. an intersection of disjoint word concepts).
        assert sum(field.evidence.abs().sum() for field in model._sentence_open_fields.values()) > 0, \
            'derived concepts must have activations'
        # The published symbolic state is the symbols' two lanes, no code.
        # A reading that holds no symbol publishes the empty leg [B, 0, 2].
        assert symbols is not None and symbols.ndim == 3 and symbols.shape[-1] == 2
        if symbols.shape[1]:
            assert torch.isfinite(symbols).all() and symbols.min() >= 0 and symbols.max() <= 1
            assert model.symbol_cache._symbol_ids.shape == symbols.shape[:2]
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()


def test_word_publication_preserves_the_composed_concept_field():
    from types import SimpleNamespace
    from Models import BasicModel
    # A completed symbolic phase may know both poles and higher-order rows.
    # Publishing symbols must not replace that result with a new native read.
    evidence = torch.tensor([[[[1., 0.]]], [[[0., 1.]]], [[[1., 1.]]]])
    carrier = SimpleNamespace(_concept_activations=evidence)
    def read_again(*args):
        pytest.fail('completed symbolic evidence was replaced by word admission')
    owner = SimpleNamespace(_sparse_active=lambda: True, cs_read_memberships=read_again)
    host = SimpleNamespace(_reading_word_percepts=object(), _reading_word_extents=None,
        _concept_owner=lambda: owner,
        symbolSpace=SimpleNamespace(forward_concept_to_symbol=lambda value:
            value._concept_activations))
    result = BasicModel._publish_reading_symbols(host, carrier)
    assert result is evidence


from functools import wraps

import pytest

import torch


def test_default_interpret_reuses_existing_kind_without_minting():
    from test_word_interpretation import _operator
    from Spaces import _concept_alloc_of
    cs, interpret = _operator()
    word = recurrent_word(interpret, [7], [], form='cat')
    kind = interpret.forward(word, order=2)
    before = dict(_concept_alloc_of(cs).concepts)
    assert interpret.forward(word) == kind
    assert interpret.forward(word, order=1) == kind
    assert dict(_concept_alloc_of(cs).concepts) == before


def test_default_interpret_reuses_a_witnessed_association():
    from test_word_interpretation import _operator
    from Spaces import _concept_alloc_of
    cs, interpret = _operator()
    word = recurrent_word(interpret, [7], [], form='cat')
    kind = cs.new_concept()
    force_order(cs, kind, 2)
    cs._csw_concept_row(2, kind)
    assert interpret.define(word, kind) is not None
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
    word = recurrent_word(interpret, [7], [1], form='wug')
    assert interpret.forward(word) == interpret.reverse(word) == word
    assert interpret.no_object(word)
    supplied = cs.new_concept()
    force_order(cs, supplied, 0)
    cs._csw_concept_row(0, supplied)
    interpret.define(word, supplied)
    obj = interpret.forward(word, occurrence=(0, 1))
    assert obj != word
    assert ('sym', word) not in cs.concept_parts(obj)
    assert cs._concept_source_order(obj) == 0
    row = cs._concept_row_of(obj)
    assert not interpret.no_object(word)
    before = len(cs.definitions._store())
    for sentence in range(1, 20):
        assert interpret.forward(word, occurrence=(sentence, 1)) == obj
        interpret.boundary()
    assert len(cs.definitions._store()) == before
    assert interpret.reverse(obj) == word
    assert cs._concept_row_of(obj) == row


def test_grammar_resolution_distinguishes_particular_and_kind():
    cs, interpret = _operator()
    word = recurrent_word(interpret, [7], [1], form='cat')
    supplied = cs.new_concept()
    force_order(cs, supplied, 0)
    cs._csw_concept_row(0, supplied)
    interpret.define(word, supplied)
    particular = interpret.forward(word, order=1)
    kind = cs.new_concept()
    force_order(cs, kind, 2)
    cs._csw_concept_row(2, kind)
    assert interpret.define(word, kind) is not None
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
    result = interpret.forward(word, object_atoms=obj, presence=activation,
                               evidence=torch.tensor([[.5,0.]]))
    torch.testing.assert_close(result, torch.tensor([[2.5, 3., 3.5, 4., 8., 9.]]))




def test_ordered_word_lookup_keeps_anagrams_distinct():
    cs, interpret = _operator()
    ab = recurrent_word(interpret, [1, 2], [], form='ab')
    ba = recurrent_word(interpret, [2, 1], [], form='ba')
    assert ab != ba
    assert recurrent_word(interpret, [1, 2], [], form='ab') == ab




@pytest.mark.usefixtures('eager_reading')
def test_every_serial_word_runs_interpret_before_composition(tmp_path, monkeypatch):
    from test_packed_reconstruction_parity import build_model
    model = build_model(tmp_path, word_capacity=8)
    owner = model._concept_owner()
    operator = owner.interpret
    # A form is a word from the reading in which MTM nominates it.
    readings_before_nomination(model, lambda: model.understand(model.inputSpace.prepInput(['the wug sat'])))
    calls = []
    readings = []
    converting = []
    # Run the identical recurrence bodies eagerly so instrumentation can
    # observe ordering. The normal compiled schedule has separate parity tests.
    def eager_loop(cond, body, carries):
        while bool(cond(*carries)):
            carries = body(*carries)
        return carries
    monkeypatch.setattr(torch, 'while_loop', eager_loop)
    from Comprehension import Evaluation
    leaf = Evaluation.leaf
    def converted(evaluation, word):
        converting.append(True)
        try:
            return leaf(evaluation, word)
        finally:
            converting.pop()
    monkeypatch.setattr(Evaluation, 'leaf', converted)
    for space in model.conceptualSpaces:
        reading = space.interpret.reading
        def identified(word, _reading=reading, **kwargs):
            result = _reading(word, **kwargs)
            readings.append((word,result))
            return result
        monkeypatch.setattr(space.interpret, 'reading', identified)
        forward = space.interpret.forward
        def observed(word, _forward=forward, **kwargs):
            result = _forward(word, **kwargs)
            if torch.is_tensor(result) and 'object_atoms' in kwargs:
                assert converting, 'numerical word interpretation ran before conversion'
                calls.append(result.detach().clone())
            return result
        monkeypatch.setattr(space.interpret, 'forward', observed)
    choose = model.languageSpace.choose_operation
    def composition(*args, **kwargs):
        assert readings, 'composition ran before the mandatory lexical reading'
        return choose(*args, **kwargs)
    monkeypatch.setattr(model.languageSpace, 'choose_operation', composition)
    with torch.no_grad():
        model.understand(model.inputSpace.prepInput(['the wug sat']))
    ids = model.inputSpace._ar_word_concept_ids
    objects = model.inputSpace._ar_word_object_ids
    active = model.inputSpace._ar_grammar_leaf_mask
    assert len(readings) >= int(active.sum())
    assert len(calls) >= int(active.sum())
    # The grammar leaf mask covers presented words; skip the MTM prefix.
    base = ids.shape[1] - active.shape[1]
    for word, obj in zip(ids[:,base:][active].tolist(), objects[:,base:][active].tolist()):
        if not owner.definitions.objects(word):
            token = owner.mtm.token_of(obj)
            assert token is not None and owner.mtm.entries[token].pending
            assert any(name.reading == word and name.token == token for name in owner.mtm.names.values())
        else:
            assert obj in owner.definitions.objects(word)
            assert operator.reverse(obj) == word
    model.End()




def test_selected_relation_grammar_ends_the_interpreted_kinds(monkeypatch):
    """The selected relation reading of 'cats are animals' owns kind refs.

    This tests the declared grammar interface, not untrained parsing.
    The closing receives resolved objects and never performs lexical resolution.
    """
    from dataclasses import replace
    from Memory import LTM0
    from Models import _append_observed_meaning
    from test_selected_relation_meaning import _program_owner
    cs, _, registry, language, _, make_program, _, _ = _program_owner(monkeypatch, order_limit=2)
    words = [recurrent_word(cs.interpret, [part], [], form=form)
             for part, form in ((7, 'cats'), (8, 'animals'))]
    for word in words:
        obj = cs.new_concept()
        force_order(cs, obj, 0)
        cs._csw_concept_row(0, obj)
        cs.interpret.define(word, obj)
    objects = [cs.interpret.forward(word, order=1) for word in words]
    # Grammar resolves existing associations; it cannot mint a second object
    # merely because the first has a different order.
    for form in ('cats', 'animals'):
        kind = cs.new_concept()
        force_order(cs, kind, 2)
        cs._csw_concept_row(2, kind)  # the supplied kind owns a payload row
        cs.interpret.define(cs.definitions.word(form=form), kind)
        cs.bind_word_concept(form, kind)
    leaves = torch.stack([registry._payload(('sym', obj)) for obj in objects])
    entry = make_program(values=leaves, refs=tuple(('sym', obj) for obj in objects))
    rules = list(language._compose_binary_rules)
    selected = int(entry.actions[-1, 1])
    rules[selected] = rules[selected]._replace(reference_roles=('argin1', 'argin2'))
    language._compose_binary_rules = rules
    word_rows = torch.tensor([cs._concept_row_of(obj) for obj in objects])
    refs, orders = language.resolve_lexical_references(cs, word_rows, entry.concept_ids, entry.actions,
                                                     leaf_orders=torch.tensor([2, 2]))
    assert orders.tolist() == [2, 2]
    assert refs.tolist() == [cs.interpret.forward(word, order=2) for word in words]
    assert all(int(kind) != particular for kind, particular in zip(refs, objects))
    entry = replace(entry, word_rows=word_rows, reference_ids=refs, reference_orders=orders)
    meaning = language.program_meaning(entry, registry)
    assert meaning.role_refs[0] == ('sym', int(refs[0]))
    assert meaning.role_refs[2] == ('sym', int(refs[1]))
    store = LTM0(leaves.shape[-1], capacity=8)
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
    word = recurrent_word(interpret, [7], [], form='wug')
    from definition_fixtures import witness_object
    member = witness_object(cs, word)
    # The later concept member belongs to a class, seated one order above
    # its members from construction. A basic-percept collector cannot
    # silently acquire a different order while keeping its physical row.
    obj = cs.new_concept(order=1)
    cs.add_part(obj, ('sym', member))
    interpret.define(word, obj)
    assert interpret.forward(word, occurrence=(0, 0)) == obj
    for tick in range(1, 20):
        assert interpret.forward(word, occurrence=(tick, 0)) == obj
        interpret.boundary()
    row = cs._concept_row_of(obj)
    original = tuple(cs.concept_parts(obj))
    cs._serial = False  # witness the later definition through the parallel field
    witnessed = witness_object(cs, recurrent_word(interpret, [8], [], form='witnessed'))
    cs.add_part(obj, ('sym', witnessed))
    cs._populate_concept_weights(obj)
    assert all(part in cs.concept_parts(obj) for part in original)
    assert ('sym', witnessed) in cs.concept_parts(obj)
    before = tuple(cs.concept_parts(obj))
    cs._serial = False
    spans = torch.tensor([[[0, 1], [1, 2]]])
    bracket = torch.tensor([[[0, 2]]])
    read = cs.cs_read_memberships((torch.tensor([[7, 8]]), spans, None,
                                  torch.tensor([[65, 65]]), spans), bracket)
    field = cs.cs_forward_content(read)
    slot = (cs._cs_field_concept_ids == obj).nonzero().flatten().item()
    assert field[slot, 0, 0, 0] > 0
    assert cs._concept_row_of(obj) == row and tuple(cs.concept_parts(obj)) == before
    assert interpret.reverse(obj) == word


def test_known_word_survives_full_inventory_without_partial_alternative(monkeypatch):
    from copy import deepcopy
    cs, interpret = _operator()
    word = recurrent_word(interpret, [7], [1], form='cat')
    obj = interpret.forward(word)
    before = deepcopy(cs.definitions.description(word))
    ids = dict(_concept_alloc_of(cs).concepts)
    def full(*args):
        raise RuntimeError('concept row capacity exhausted before word admission; no identity was minted')
    monkeypatch.setattr(cs, '_preflight_concept_row', full)
    assert recurrent_word(interpret, [8], [1], form='cat', word_reading=True) == word
    assert interpret.forward(word) == obj
    assert cs.definitions.description(word) == before
    assert dict(_concept_alloc_of(cs).concepts) == ids
    assert cs.concept_admission_stats()['dropped']['word alternative'] == 1
    with pytest.raises(RuntimeError, match='capacity exhausted'):
        recurrent_word(interpret, [8], [1], form='cat')
