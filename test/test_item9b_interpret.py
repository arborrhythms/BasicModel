"""Word-to-object interpretation owns testimony, resolution, and lexicalization."""
import torch

from test_cs_sparse_weights import _cs
from Spaces import _concept_alloc_of


def _operator():
    from Language import InterpretLayer
    cs = _cs(nS=64, order=3)
    return cs, InterpretLayer(conceptualSpace=cs)


def test_interpret_reuses_provisional_testimony_and_reverses_by_identity():
    cs, interpret = _operator()
    word = interpret.lookup_word([7], [1], form='wug')
    obj = interpret.forward(word, order=1, occurrence=(0, 1))
    assert obj != word
    assert cs.concept_parts(obj) == [('sym', word)]
    assert cs._concept_source_order(obj) == 1
    row = cs._csw_row_of(obj)
    store = _concept_alloc_of(cs).layer()
    assert bool(store.provisional[row])
    interpret.boundary()
    assert bool(store.provisional[row]), 'one occurrence must not admit testimony'
    assert interpret.forward(word, order=1, occurrence=(1, 1)) == obj
    assert interpret.reverse(obj) == word
    for sentence in range(2, 20):
        interpret.forward(word, order=1, occurrence=(sentence, 1))
        interpret.boundary()
    assert not bool(store.provisional[row])


def test_grammar_resolution_distinguishes_particular_and_kind():
    cs, interpret = _operator()
    word = interpret.lookup_word([7], [1], form='cat')
    particular = interpret.forward(word, order=1)
    kind = cs.new_concept()
    _concept_alloc_of(cs).reference_orders[kind] = 2
    cs.bind_word_concept('cat', kind)
    assert particular != kind
    assert cs._concept_source_order(particular) == 1
    assert cs._concept_source_order(kind) == 2
    assert interpret.forward(word, order=2) == kind
    assert interpret.reverse(particular) == interpret.reverse(kind) == word
    assert interpret.forward(word, order=1) == particular
    assert interpret.forward(word, order=2) == kind


def test_tensor_face_replaces_content_and_preserves_occurrence_coordinates():
    _, interpret = _operator()
    word = torch.tensor([[1., 2., 3., 4., 8., 9.]])
    obj = torch.tensor([[5., 6., 7., 8.]])
    activation = torch.tensor([.5])
    result = interpret.forward(word, object_atoms=obj, activation=activation)
    torch.testing.assert_close(result, torch.tensor([[2.5, 3., 3.5, 4., 8., 9.]]))


def test_interpret_replaces_the_host_triple_creation_api():
    from Spaces import ConceptualSpace
    assert not hasattr(ConceptualSpace, 'create_word_object_meta')
    assert not hasattr(ConceptualSpace, '_automatic_word_object_meta')


def test_ordered_word_lookup_keeps_anagrams_distinct():
    cs, interpret = _operator()
    ab = interpret.lookup_word([1, 2], [], form='ab')
    ba = interpret.lookup_word([2, 1], [], form='ba')
    assert ab != ba
    assert interpret.lookup_word([1, 2], [], form='ab') == ab


def test_unused_one_off_testimony_is_forgotten_at_the_boundary():
    cs, interpret = _operator()
    word = interpret.lookup_word([7], [], form='wugg')
    obj = interpret.forward(word, occurrence=(0, 0))
    interpret.boundary()
    assert obj not in _concept_alloc_of(cs).retired
    interpret.boundary()
    assert obj in _concept_alloc_of(cs).retired
    assert (word, 1) not in _concept_alloc_of(cs).interpretations
    assert interpret.reverse(obj) == word, 'captured programs keep their owned inverse'


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


def test_native_parallel_pass_and_boundary_do_not_interpret_words(tmp_path, monkeypatch):
    from test_grounded_xor import grounded_model
    model, _ = grounded_model(tmp_path, inventory=128, field_slots=32)
    def forbidden(*args, **kwargs):
        raise AssertionError('parallel perception dispatched serial interpretation or composition')
    for cs in model.conceptualSpaces:
        monkeypatch.setattr(cs.interpret, 'forward', forbidden)
    from Spaces import ConceptualSpace
    monkeypatch.setattr(ConceptualSpace, '_automatic_joint_concept', forbidden)
    with torch.no_grad():
        model.understand(model.inputSpace.prepInput(['the wug sat']))
        model.dispatch_per_row_reset([True])
        model.dispatch_soft_reset()
        model.post_tick_compact()
        model.End()


def test_selected_generic_grammar_seals_the_interpreted_kinds(monkeypatch):
    """The selected generic reading of 'cats are animals' owns kind refs.

    This tests the declared grammar interface, not untrained English parsing.
    The seal receives resolved objects and never performs lexical resolution.
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
        cs.bind_word_concept(form, kind)
    leaves = torch.stack([registry._payload(('sym', obj)) for obj in objects])
    entry = make_program(values=leaves, refs=tuple(('sym', obj) for obj in objects))
    rules = list(language._compose_binary_rules)
    selected = int(entry.actions[-1, 1])
    rules[selected] = rules[selected]._replace(reference_orders=(('I1', 2), ('I2', 2)))
    language._compose_binary_rules = rules
    word_rows = torch.tensor([cs._csw_row_of(word) for word in words])
    refs, orders = language.resolve_lexical_references(cs, word_rows, entry.concept_ids, entry.actions)
    assert orders.tolist() == [2, 2]
    assert refs.tolist() == [cs.interpret.forward(word, order=2) for word in words]
    assert all(int(kind) != particular for kind, particular in zip(refs, objects))
    entry = replace(entry, word_rows=word_rows, reference_ids=refs, reference_orders=orders)
    meaning = language.program_meaning(entry, registry)
    assert meaning.role_refs[0] == ('sym', int(refs[0]))
    assert meaning.role_refs[2] == ('sym', int(refs[1]))
    store = TernaryTruthStore(leaves.shape[-1], capacity=8)
    row = _append_observed_meaning(store, entry.end_state, 1, meaning=meaning, trust=.9)
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
    witnessed = cs.new_concept()
    cs.add_part(witnessed, 8)
    cs._populate_concept_weights(witnessed)
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
