"""Document-relative addresses, re-witnessing, and lossless reference migration."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from Layers import TernaryTruthStore
from Meaning import ConceptualMeaning
from Occurrence import address_key, document_digest, sentence_key


def meaning(value=1.):
    return ConceptualMeaning.from_description(torch.full((4,), float(value)))


def write(store, document='book', position=0, value=1., evidence=(0., 0.), timestamp=1.):
    return store.append_meaning(meaning(value), kind='observation', document_key=document,
        sentence_index=position, content_key=sentence_key((b'the', b'cat')),
        evidence=evidence, timestamp=timestamp)


def test_address_is_full_int64_content_based_and_rng_independent():
    rng = torch.get_rng_state().clone()
    code = sentence_key((b'the', b'cat'))
    key = address_key(document_digest(('corpus', 23)), 8, code)
    source = ('import sys; sys.path.insert(0,"bin"); from Occurrence import *; print(address_key(document_digest(("corpus",23)),8,'
              'sentence_key((b"the",b"cat"))))')
    assert key == int(subprocess.check_output([sys.executable, '-c', source], text=True))
    assert torch.equal(rng, torch.get_rng_state())
    left, right = TernaryTruthStore(4, 8), TernaryTruthStore(4, 8)
    a, b = write(left), write(right)
    assert left.occurrence_of(a) == right.occurrence_of(b)
    assert not {'occurrence_id', '_next_occurrence', '_occurrence_namespace'} & set(left.state_dict())
    assert sentence_key((b'cat', b'the')) != code


def test_four_occurrences_after_four_hundred_presentations_and_full_upsert():
    store = TernaryTruthStore(4, 4)
    for epoch in range(400):
        for row in range(4):
            assert write(store, ('xor', row), value=epoch + 1., timestamp=epoch) == row
    assert len(store) == 4
    assert store.witness_count.tolist() == [400] * 4
    assert store.timestamp.tolist() == [399.] * 4
    assert store.slots[:, 0].eq(400).all()
    with pytest.raises(OverflowError, match='forgetting'):
        write(store, 'new-document')
    assert len(store) == 4


def test_rereading_replaces_vectors_refreshes_time_and_joins_poles():
    store = TernaryTruthStore(4, 4)
    row = write(store, value=2., evidence=(.8, .1), timestamp=7.)
    band = store.when[row].clone()
    assert write(store, value=3., evidence=(.2, .7), timestamp=12.) == row
    torch.testing.assert_close(store.slots[row, 0], torch.full((4,), 3.))
    assert store.row(row)['evidence'] == pytest.approx((.8, .7))
    assert float(store.timestamp[row]) == 12.
    assert torch.equal(band, store.when[row])
    assert write(store, 'other-document') == 1
    assert write(store, position=1) == 2
    assert list(store._index_occurrences) == store.address_keys[:3].tolist()


def test_full_store_can_rewitness_both_existing_prediction_pair_addresses():
    store = TernaryTruthStore(4, 2)
    arguments = dict(presence_logits=torch.zeros(3), document='turn')
    first = store.append_expectation_pair(meaning(.5), meaning(1.), **arguments)
    assert first == (0, 1) and len(store) == store.capacity
    assert store.append_expectation_pair(meaning(.5), meaning(1.), **arguments) == first
    assert store.witness_count.tolist() == [2, 2]


def test_when_is_unit_bracket_with_coarse_fine_roundtrip():
    from Spaces import WhenEncoding, WhenStartDurationEncoding
    encoding = WhenEncoding(n_when=4).set_capacity(2**28)
    onset = WhenStartDurationEncoding(n_when=4).set_capacity(2**28)
    positions = torch.tensor([0, 1, 5, 2**24 + 1, 2**28 - 1])
    band = encoding.encode(positions)
    torch.testing.assert_close(band, (onset.encode(positions) + onset.encode(positions + 1)) / 2)
    torch.testing.assert_close(encoding.decode_index(band), positions)
    assert encoding.maxVal > int(positions.max()) + 1


def test_references_and_content_keys_survive_compaction_and_checkpoint():
    store = TernaryTruthStore(4, 8)
    removed = write(store, 'discard')
    store.set_origin(removed, store.ORIGIN_USER)
    source = write(store, 'kept')
    child = ConceptualMeaning(meaning(2.).roles, meaning().role_mask,
                             role_refs=(store.occurrence_of(source), None, None))
    target = store.append_meaning(child, kind='observation', document_key='reference')
    store.refs[target, 0] = store.address_keys[source]
    reference = store.occurrence_of(source)
    assert store.clear_origin(store.ORIGIN_USER) == 1
    assert store._index_occurrences[reference] == 0
    assert store.meaning_of(1).role_refs[0] == reference
    assert int(store.refs[1, 0]) == int(store.address_keys[0])
    restored = TernaryTruthStore(4, 8)
    restored.load_state_dict(copy.deepcopy(store.state_dict()))
    restored.load_semantic_extras(copy.deepcopy(store.semantic_extras()))
    assert restored._index_occurrences[reference] == 0
    assert restored.content_key(0) == sentence_key((b'the', b'cat'))


def test_legacy_checkpoint_rekeys_native_and_typed_edges():
    store = TernaryTruthStore(4, 8)
    source = write(store)
    child = ConceptualMeaning(meaning(2.).roles, meaning().role_mask,
                             role_refs=(store.occurrence_of(source), None, None))
    target = store.append_meaning(child, kind='observation', document_key='child')
    state, extras = copy.deepcopy(store.state_dict()), copy.deepcopy(store.semantic_extras())
    namespace = '1234567890abcdef1234567890abcdef'
    old_ref = ('ltm', namespace, 0)
    extras['version'], extras['namespace'] = 4, namespace
    for i, record in enumerate(extras['records']):
        record['id'] = i
        if i == target:
            record['context']['role_refs'] = (old_ref, None, None)
        state['semantic_fingerprint'][i] = torch.tensor(store._context_fingerprint(
            record['context'], record['text'], record['expectation']), dtype=torch.uint8)
    for name in ('address_keys', 'document_keys', 'sentence_content_keys', 'sentence_index', 'witness_count'):
        state.pop(name)
    state['occurrence_id'] = torch.tensor([0, 1, -1, -1, -1, -1, -1, -1])
    state['_next_occurrence'] = torch.tensor(2)
    state['_occurrence_namespace'] = torch.tensor(list(bytes.fromhex(namespace)), dtype=torch.uint8)
    state['row_ids'][:2] = torch.tensor([2**62, 2**62 + 1])
    state['refs'][target, 0] = 2**62
    state['timestamp'][:2] = torch.tensor([100., 500.])
    restored = TernaryTruthStore(4, 8)
    restored.load_state_dict(state)
    restored.load_semantic_extras(extras)
    reference = restored.occurrence_of(0)
    assert restored.meaning_of(1).role_refs[0] == reference
    assert int(restored.refs[1, 0]) == reference[2]
    assert restored.timestamp[:2].tolist() == [100., 500.]
    assert restored.remap_legacy_references({'history': (old_ref,)}) == {'history': (reference,)}
    from Spaces import WhenEncoding
    encoder = WhenEncoding(n_when=4).set_capacity(8)
    assert encoder.decode_index(restored.when[:2]).tolist() == [0, 1]


def test_live_gate_reread_changes_clock_but_not_address_or_when():
    from test_mm_xor import _fresh_model
    model, _, data = _fresh_model('data/XOR_grammar.xml')
    try:
        optimizer = model.getOptimizer(lr=.01)
        model.runEpoch(optimizer=optimizer, batchSize=4, split='train')
        store = model.symbolSpace.ltm_store
        rows = (store.rel_type[:len(store)] != store.REL_DEF).nonzero().flatten()
        assert len(rows) == 4
        expected = {sentence_key(tuple(text.split())) for text in
                    (b'hello world', b'hello there', b'loving world', b'loving there')}
        assert {store.content_key(int(row)) for row in rows} == expected
        addresses, bands = store.address_keys[rows].clone(), store.when[rows].clone()
        timestamps = store.timestamp[rows].clone()
        model.runEpoch(optimizer=optimizer, batchSize=4, split='train')
        assert int((store.rel_type[:len(store)] != store.REL_DEF).sum()) == 4
        assert torch.equal(store.address_keys[rows], addresses)
        assert torch.equal(store.when[rows], bands)
        assert (store.timestamp[rows] > timestamps).all()
        assert store.witness_count[rows].tolist() == [2] * 4
    finally:
        model.End()


def test_thought_references_are_source_addresses_not_history_counters():
    from Layers import WhatInteractionMemory
    first, second = WhatInteractionMemory(), WhatInteractionMemory()
    question = ConceptualMeaning(meaning().roles, meaning().role_mask, mode='interrogative')
    for owner in (first, second):
        owner._address_sources = {0: (('conversation', 'alice', 7), 0)}
    left = first.begin_thought_episode(question)
    right = second.begin_thought_episode(question)
    assert first.thought_reference(left) == second.thought_reference(right)
    assert left.address != left.id


def test_when_reader_inventory_covers_current_attribute_and_dynamic_readers():
    import ast
    root = Path(__file__).resolve().parents[1]
    fixture = json.loads((root / 'test/fixtures/when-readers-round4a0.json').read_text())
    attributes, actual = set(fixture['attributes']), set()
    class Scan(ast.NodeVisitor):
        def __init__(self, path):
            self.path, self.scope = path, []
        def visit_ClassDef(self, node):
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()
        def visit_FunctionDef(self, node):
            self.scope.append(node.name)
            for item in ast.walk(node):
                direct = (isinstance(item, ast.Attribute) and isinstance(item.ctx, ast.Load)
                          and item.attr in attributes)
                dynamic = (isinstance(item, ast.Call) and isinstance(item.func, ast.Name)
                    and item.func.id == 'getattr' and len(item.args) >= 2
                    and isinstance(item.args[1], ast.Constant) and item.args[1].value in attributes)
                if direct or dynamic:
                    actual.add((self.path, '.'.join(self.scope)))
            self.scope.pop()
    for path in (root / 'bin').glob('*.py'):
        Scan(str(path.relative_to(root))).visit(ast.parse(path.read_text()))
    assert actual == {(row['file'], row['function']) for row in fixture['readers']}
    assert {'_event_split', '_lift_when', '_lower_when'} <= {
        row['function'] for row in fixture['dynamic_readers']}
    for row in fixture['readers'] + fixture['dynamic_readers']:
        assert isinstance(row['previous_absolute_time'], bool) and row['now']


def test_prepared_dataset_value_retains_its_source_after_cursor_advances():
    from test_mm_xor import _fresh_model
    from Occurrence import stage_sources
    model, _, data = _fresh_model('data/XOR_grammar.xml')
    try:
        loader = data.data_loader(split='train', num_streams=1)
        cursor = iter(loader)
        raw, _ = next(cursor)
        first = model.inputSpace.prepInput(raw)
        raw, _ = next(cursor)
        model.inputSpace.prepInput(raw)
        split, rows = first._sentence_source
        stage_sources(model, split, rows, first)
        assert rows == [0]
        original = model._sentence_sources
        model.inputSpace.prepInput(['a different input'])
        stage_sources(model, split, rows, first)
        assert model._sentence_sources == original
        assert original[0][0][1] == 1
    finally:
        model.End()


def test_rewitness_keeps_poles_and_the_round3a_net_existence_read():
    from reasoning import TruthGroundedReasoner
    store = TernaryTruthStore(4, 2)
    value = meaning()
    row = store.append_meaning(value, kind='fact', evidence=(.8, 0.))
    assert store.append_meaning(value, kind='fact', evidence=(0., .9)) == row
    result = TruthGroundedReasoner(store=store).existence_evidence(value)
    assert store.row(row)['evidence'] == pytest.approx((.8, .9))
    assert result['support_true'] == 0.
    assert result['support_false'] == pytest.approx(.1)
    assert len(result['candidates']) == 1


def test_checkpoint_can_grow_capacity_without_changing_an_address():
    source = TernaryTruthStore(4, 2)
    write(source, 'one', position=1)
    write(source, 'two', position=1)
    restored = TernaryTruthStore(4, 8)
    restored.load_state_dict(copy.deepcopy(source.state_dict()))
    restored.load_semantic_extras(copy.deepcopy(source.semantic_extras()))
    assert restored.capacity == 8 and len(restored) == 2
    assert restored.occurrence_of(1) == source.occurrence_of(1)
    from Spaces import WhenEncoding
    encoder = WhenEncoding(n_when=4).set_capacity(8)
    torch.testing.assert_close(restored.when[:2], encoder.encode(torch.tensor([1, 1])))
    assert write(restored, 'three') == 2


def test_serving_source_is_conversation_and_turn_and_replay_is_stable():
    from serve import _request_document
    body = dict(conversation_id='chat', turn_id=7,
                messages=[dict(role='user', content='hello world')])
    assert _request_document(body) == ('conversation', 'chat', 7)
    assert _request_document(dict(body, turn_id=8)) != _request_document(body)
    raw = dict(messages=body['messages'])
    assert _request_document(copy.deepcopy(raw)) == _request_document(raw)
    assert _request_document(dict(messages=raw['messages'] * 2)) != _request_document(raw)


def test_explicit_rewitness_rejects_a_self_reference_without_mutating_postings():
    store = TernaryTruthStore(4, 2)
    row = write(store)
    before = copy.deepcopy(store.state_dict())
    reference = store.occurrence_of(row)
    value = ConceptualMeaning(meaning().roles, meaning().role_mask,
                              role_refs=(reference, None, None))
    with pytest.raises(ValueError, match='cycle'):
        store.append_meaning(value, kind='observation', document_key='book',
            content_key=sentence_key((b'the', b'cat')))
    for name, tensor in before.items():
        torch.testing.assert_close(store.state_dict()[name], tensor)


def test_local_reference_tag_distinguishes_a_journal_slot_from_the_same_signed_address():
    from ReferenceContext import ReferenceBank, resolve_operand
    bank = ReferenceBank(torch.empty(1, 0, dtype=torch.long), torch.empty(1, 0, 4),
        torch.empty(1, 0, dtype=torch.bool), torch.empty(1, 0, dtype=torch.bool),
        torch.ones(1, 4), torch.zeros(1, dtype=torch.bool))
    live = (torch.empty(1, 0, 4), torch.empty(1, 0, dtype=torch.long),
            torch.empty(1, 0, dtype=torch.long), torch.empty(1, 0, dtype=torch.bool),
            torch.empty(1, 0, dtype=torch.long), torch.empty(1, 0, dtype=torch.bool))
    for local in (False, True):
        result = resolve_operand(torch.ones(1, 1, 4), torch.tensor([[-26]]),
            torch.zeros(1, 1, dtype=torch.long), mode=None, bank=bank, live=live,
            active=torch.ones(1, 1, dtype=torch.bool), local=torch.tensor([[local]]))
        assert bool(result[3]) is (not local)
        assert int(result[1]) == -26
