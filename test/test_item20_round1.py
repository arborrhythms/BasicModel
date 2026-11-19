"""Direct certificates for the behavior-neutral memory and journal move."""
from types import SimpleNamespace

import pytest
import torch

from Memory import MemoryStore, ShortTermMemory, MediumTermMemory, LTM0, LTM1, STMChunks as chunks
from Meaning import ConceptualMeaning
from ThoughtChunks import put, read, release
from ThoughtReferences import bindings, with_slots


def idea():
    return with_slots(ConceptualMeaning.from_description(torch.ones(8)), (), pair=(.7, .2))


def test_thought_deposits_advance_when_with_empty_where_on_one_model_clock():
    from RowTime import encoder
    stm = ShortTermMemory(concept_dim=8)
    from clock_fixtures import for_stm
    model = for_stm(stm, tick=1)
    sentence_when = encoder().encode(1)
    state, sentence = chunks.deposit(stm.state, torch.ones(1, 8), torch.tensor([True]),
        where=(.1,.2,.3,.4), when=sentence_when)
    stm.set_state(state)
    first, second = (put(stm, idea(), row=0) for _ in range(2))
    a, b = (bindings(value)['_chunk_when'] for value in (first, second))
    assert a != b and model.present() == 3
    for when in (a, b):
        index = int(chunks.locate(stm.state, when, band=chunks.WHEN)[0])
        assert index >= 0 and index != int(sentence[0])
        assert stm.state[chunks.WHERE][0, index].tolist() == [-1.] * 4
        assert torch.equal(stm.state[chunks.WHEN][0, index], torch.tensor(when))
        assert int(encoder().decode_index(torch.tensor(when))) in (2, 3)
        torch.testing.assert_close(read(stm, when, 0).roles, idea().roles)
    assert int(chunks.locate(stm.state, sentence_when, band=chunks.WHEN)[0]) == int(sentence[0])
    put(stm, first, a, 0, existing=True)
    assert model.present() == 3


def test_thought_eviction_is_absent_by_when_for_read_and_release():
    stm = ShortTermMemory(capacity=1, concept_dim=8)
    from clock_fixtures import for_stm
    for_stm(stm)
    old = bindings(put(stm, idea(), row=0))['_chunk_when']
    put(stm, idea(), row=0)
    with pytest.raises(ValueError, match=r'absent at its .when'):
        read(stm, old, 0)
    with pytest.raises(ValueError, match=r'absent at its .when'):
        release(SimpleNamespace(_thought_stm=stm), old, 0)


def test_thought_clock_restores_with_the_trial_and_advances_only_on_deposit():
    from WalkTrials import ThoughtTrialState
    stm = ShortTermMemory(capacity=1, concept_dim=8)
    from clock_fixtures import for_stm
    model = for_stm(stm)
    trial = ThoughtTrialState(model)
    key = bindings(put(stm, idea(), row=0, pin=True))['_chunk_when']
    with pytest.raises(OverflowError, match='capacity'):
        put(stm, idea(), row=0)
    assert model.present() == 1
    trial.restore()
    assert model.present() == 0
    assert bindings(put(stm, idea(), row=0, pin=True))['_chunk_when'] == key
    release(SimpleNamespace(_thought_stm=stm), key, 0)
    from RowTime import HORIZON
    model.when_time.fill_(HORIZON)
    with pytest.raises(OverflowError, match='model time'):
        put(stm, idea(), row=0)


@pytest.mark.parametrize('leaves', [(0,), (0, 1)])
def test_symbol_description_has_no_own_occurrence_even_for_one_leaf(leaves):
    from ClauseRow import Clause
    from test_reference_close import fixture
    resolver, store, basis = fixture()
    node = Clause(ConceptualMeaning.from_description(basis[0]), point=basis[0])
    plan = dict(items=[dict(written_node=node, mode='new', leaves=leaves)], prepared_root=node)
    with pytest.raises(ValueError, match='description .* has no own occurrence'):
        resolver.commit(plan, 0, {('mention', id(node)): ('sym', 17)},
                        document='round-one', position=0, training=False)
    assert not resolver.components.owner.mtm.entries


@pytest.mark.parametrize('kind', [0, 3])
def test_reading_nonoperation_has_empty_rows_and_no_row_zero_gradient(kind):
    from Models import BasicModel
    values = torch.arange(1., 49.).reshape(1, 2, 24).requires_grad_()
    journal = SimpleNamespace(_choice_journal_columns=torch.tensor([[1, 0]]),
        _choice_values=values, _choice_refs=torch.tensor([[[11, 12], [21, 22]]]),
        _choice_ref_relations=torch.ones(1, 2, 2, dtype=torch.bool),
        _choice_log_probabilities=torch.tensor([[-.3, -.7]]),
        _case_state=(None, None, None, torch.full((1, 2, 8), .7), torch.ones(1, 2, dtype=torch.bool)))
    # Since the tables round the entries index their reading (ReadingIndex):
    # it reads the declared unary rule and accepts only a reading a walk can
    # record, so a transposition needs two operands on the stack. The
    # language therefore declares its one unary rule, the transposition
    # stands where it is legal, and the rows are named by their role.
    language = SimpleNamespace(_compose_unary_rules=(SimpleNamespace(method_name='generic'),),
                               _compose_binary_rules=())
    owner = SimpleNamespace(symbolSpace=SimpleNamespace(ltm_store=SimpleNamespace(model_clock=0)), _reconstruction_stack=lambda: journal,
                            _concept_owner=lambda: None, languageSpace=language,
                            conceptualSpace=SimpleNamespace(stm=SimpleNamespace(
                                strata=SimpleNamespace(capture=lambda *args, **kwargs: None))))
    actions = ([[0, -1, 0], [2, 0, -1], [0, -1, 1]] if kind == 0 else
               [[0, -1, 0], [0, -1, 1], [3, -1, -1], [2, 0, -1]])
    operation, = [row for row, action in enumerate(actions) if action[0] == 2]
    nonoperations = [row for row, action in enumerate(actions) if action[0] in (0, 3)]
    assert kind in [actions[row][0] for row in nonoperations]
    program = (torch.tensor([[0, 1]]), torch.tensor([actions]), torch.tensor([[0]]),
               torch.tensor([[0 if row == operation else -1 for row in range(len(actions))]]))
    entry, = BasicModel._program_entries(owner, program, torch.ones(1, 2, 8),
        torch.tensor([[1, 2]]), torch.tensor([[1, 2]]), torch.ones(1, 2), torch.ones(1, 3, 8))
    for name in ('operation_values', 'operation_refs', 'operation_relations', 'case_values', 'case_selected'):
        table = getattr(entry, name)
        fill = -1 if name == 'operation_refs' else 0
        assert (table[nonoperations] == fill).all()
        assert not torch.equal(table[nonoperations[0]], table[operation])
    torch.testing.assert_close(entry.operation_values[operation].flatten(), values[0, 1])
    assert entry.operation_probabilities[nonoperations].tolist() == [0.] * len(nonoperations)
    entry.operation_values.sum().backward()
    assert not values.grad[0, 0].any() and (values.grad[0, 1] == 1).all()


@pytest.mark.parametrize('store', [ShortTermMemory, MediumTermMemory, LTM0, LTM1])
def test_memory_subclasses_share_snapshot_ownership_without_state(store):
    assert issubclass(store, MemoryStore)
    value = torch.tensor([.2, .7], requires_grad=True)
    committed = store._snapshot_tensor(value, cpu=True)
    trial = store._snapshot_tensor(value, detach=False)
    assert committed.data_ptr() != value.data_ptr() != trial.data_ptr()
    assert not committed.requires_grad and trial.requires_grad
    torch.testing.assert_close(committed, value, rtol=0, atol=0)
    trial.sum().backward()
    assert value.grad.tolist() == [1., 1.]
