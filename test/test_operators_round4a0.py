"""Document-relative addresses, re-witnessing, and versioned layout checkpoints."""
from Memory import STM0
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from Memory import LTM0
from Meaning import ConceptualMeaning


def meaning(value=1.):
    return ConceptualMeaning.from_description(torch.full((12,), float(value)))


def write(store, document='book', position=0, value=1., evidence=(0., 0.)):
    return store.append_meaning(meaning(value),

        evidence=evidence)










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
    store = LTM0(12, 8, percept_width=8)
    removed = write(store, 'discard')
    source = write(store, 'kept')
    child = ConceptualMeaning(meaning(2.).roles, meaning().role_mask,
                             role_refs=(store.occurrence_of(source), None, None))
    target = store.append_meaning(child)
    reference = store.occurrence_of(source)
    store.compact_rows((source,target))
    assert store.location(reference) == 0
    assert store.meaning_of(1).role_refs[0] == reference
    assert not hasattr(store, 'refs')
    restored = LTM0(12, 8, percept_width=8)
    restored.load_state_dict(copy.deepcopy(store.state_dict()))
    assert restored.location(reference) == 0
    assert restored.meaning_of(1).role_refs[0] == reference


def test_pre_layout_checkpoint_is_rejected_without_migration():
    store = LTM0(12, 8, percept_width=8)
    write(store)
    state = copy.deepcopy(store.state_dict())
    state['truth_schema'].fill_(10)
    restored = LTM0(12, 8, percept_width=8)
    before = copy.deepcopy(restored.state_dict())
    with pytest.raises(ValueError, match='unsupported LTM0 checkpoint layout'):
        restored.load_state_dict(state)
    for name, value in before.items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)




def test_thought_references_are_source_addresses_not_history_counters():
    from Layers import WhatInteractionMemory
    first, second = WhatInteractionMemory(), WhatInteractionMemory()
    question = ConceptualMeaning(meaning().roles, meaning().role_mask, sentence_kind='relation')
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
    assert {'_event_split'} <= {
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




def test_checkpoint_rejects_a_different_constructed_capacity():
    source = LTM0(12, 2, percept_width=8)
    write(source, 'one', position=1)
    write(source, 'two', position=1)
    restored = LTM0(12, 8, percept_width=8)
    with pytest.raises(ValueError, match='capacity/width'):
        restored.load_state_dict(copy.deepcopy(source.state_dict()))
    assert restored.capacity == 8 and not len(restored)


def test_serving_source_is_conversation_and_turn_and_replay_is_stable():
    from serve import _request_document
    body = dict(conversation_id='chat', turn_id=7,
                messages=[dict(role='user', content='hello world')])
    assert _request_document(body) == ('conversation', 'chat', 7)
    assert _request_document(dict(body, turn_id=8)) != _request_document(body)
    raw = dict(messages=body['messages'])
    assert _request_document(copy.deepcopy(raw)) == _request_document(raw)
    assert _request_document(dict(messages=raw['messages'] * 2)) != _request_document(raw)


def test_explicit_rewitness_keeps_a_self_reference_without_inventing_evidence():
    store = LTM0(12, 2, percept_width=8)
    row = write(store)
    before = copy.deepcopy(store.state_dict())
    reference = store.occurrence_of(row)
    value = ConceptualMeaning(meaning().roles, meaning().role_mask,
                              role_refs=(reference, None, None))
    kept = store.append_meaning(value,  )
    assert kept != row and store.meaning_of(kept).role_refs[0] == reference
    # The self identity is a binding, not a composing cycle.
    assert store.read_structure(reference)['incomplete'] == ()
    torch.testing.assert_close(store.trust[:1],before['trust'][:1],rtol=0,atol=0)
    assert store.row(kept)['evidence'] == (0.,0.)


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
