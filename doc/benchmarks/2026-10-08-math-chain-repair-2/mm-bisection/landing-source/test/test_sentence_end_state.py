"""September 28 decision: durable rows own only the completed field."""
from dataclasses import fields, replace
import copy
from types import SimpleNamespace

import torch
import pytest

from ClauseRow import Clause
from test_clause_storage import clause_store, idea_clause, part_clause


def test_reading_an_idea_returns_its_one_occupied_slot():
    store, _ = clause_store()
    ended = idea_clause()
    row = store.write_clause(ended, evidence=(.75, .25))
    meaning = store.meaning_of(row)
    assert meaning.role_mask.tolist() == [True, False, False]
    torch.testing.assert_close(meaning.roles[0], ended.point, rtol=0, atol=0)
    assert not meaning.roles[1:].any()
    assert store.refs[row].tolist() == [1, 3, -1]
    assert store.row(row)['evidence'] == (.75, .25)


def test_relation_stores_the_supplied_contents_without_rereading_references():
    store, points = clause_store()
    ended = part_clause()
    # Reference identities can outlive a change to the native dictionary.
    # Such a change cannot replace the values held when this clause ended.
    for cid in points:
        points[cid] = torch.full((4,), float(cid + 10))
    row = store.write_clause(ended)
    torch.testing.assert_close(store.slots[row], ended.meaning.roles, rtol=0, atol=0)
    assert store.refs[row].tolist() == list(ended.refs)


def test_no_clause_derivation_or_factored_target_is_checkpointed():
    store, _ = clause_store()
    ended = idea_clause()
    row = store.write_clause(ended, trust=.5)
    assert 'derivation' not in {field.name for field in fields(Clause)}
    assert not hasattr(store, 'clause_derivation')
    extras = store.semantic_extras()
    assert all('clause' not in record for record in extras['records'])
    restored, _ = clause_store()
    restored.load_state_dict(copy.deepcopy(store.state_dict()))
    restored.load_semantic_extras(copy.deepcopy(extras))
    torch.testing.assert_close(restored.slots, store.slots, rtol=0, atol=0)
    torch.testing.assert_close(restored.refs, store.refs, rtol=0, atol=0)
    assert restored.meaning_of(row).role_mask.tolist() == [True, False, False]


def test_finishing_uses_actual_operands_and_result_without_running_an_operator():
    from ClauseJournal import finish_clause
    from Understanding import AnswerProgram

    def forbidden(*args, **kwargs):
        raise AssertionError('finishing replayed an operator')

    language = SimpleNamespace(
        _compose_binary_rules=[SimpleNamespace(method_name='lift', clause_form='S',
            head_role=0, reference_kinds=(), lhs='S')], _compose_unary_rules=[],
        forward_binary_step=forbidden, forward_unary_step=forbidden)
    left, right, result = torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.],
                                      [-4., -3., -2., -1.]])
    frames = torch.zeros(3, 3, 4)
    frames[2] = torch.stack((left, right, result))
    program = AnswerProgram(rows=torch.tensor([0, 1]), word_rows=torch.tensor([0, 1]),
        activations=torch.ones(2), leaves=torch.zeros(2, 4), concept_ids=torch.tensor([1, 2]),
        actions=torch.tensor([[0, -1, 0], [0, -1, 1], [1, 0, -1]]),
        targets=torch.tensor([0, 1, 1]), end_state=torch.stack((result, result * 0, result * 0)),
        operation_values=frames)
    ended = finish_clause(language, program)
    torch.testing.assert_close(ended.point, result, rtol=0, atol=0)
    torch.testing.assert_close(ended.meaning.roles[:2], torch.stack((left, right)), rtol=0, atol=0)
    assert ended.refs == (1, 2, -1)


def test_numerical_journal_is_fullgraph_and_keeps_the_gradient_path():
    from Models import BasicModel

    buffer = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).requires_grad_()
    candidate = torch.tensor([[.5, .25, .125, .0625], [1., 1., 1., 1.]], requires_grad=True)
    choice = SimpleNamespace(kind=torch.tensor([1, 2]), position=torch.tensor([0, 1]),
        applied=torch.tensor([True, False]), candidate=candidate)
    journal = torch.zeros(2, 5, 12)
    def record(journal,slot,buffer,candidate):
        rows=torch.arange(buffer.shape[0])
        left=buffer[rows,(choice.position+1).clamp_max(2)]
        right=buffer[rows,choice.position]
        frame=torch.cat((left,right,candidate),-1)
        return BasicModel._tensor_record_selected_values(journal,slot,frame,choice.applied)
    compiled = torch.compile(record, backend='eager', fullgraph=True)
    recorded = compiled(journal, torch.tensor(2), buffer, candidate)
    expected = torch.cat((buffer[0, 1], buffer[0, 0], candidate[0]))
    torch.testing.assert_close(recorded[0, 2], expected, rtol=0, atol=0)
    assert not recorded[1].any()
    recorded.sum().backward()
    torch.testing.assert_close(buffer.grad[0, :2], torch.ones(2, 4), rtol=0, atol=0)
    torch.testing.assert_close(candidate.grad[0], torch.ones(4), rtol=0, atol=0)
    assert not candidate.grad[1].any()


def test_answer_and_recall_own_completed_fields_without_programs():
    from Meaning import ConceptualMeaning
    from Output import AnswerDerivation
    from Understanding import SentenceEndState, Understanding

    meaning = ConceptualMeaning.from_description(torch.tensor([1., 2., 3., 4.]))
    ended = SentenceEndState(meaning, refs=(1, 2, -1))
    understanding = Understanding(sentence_states=(ended,), sentence_fields={0: (ended,)})
    answer = AnswerDerivation(answer_symbol=None, sentence_states=(ended,),
                              conceptual_answer=ended.end_state[None])
    recalled = ended.detached()
    meaning.roles.zero_()
    assert not hasattr(understanding, 'answer_program')
    assert not hasattr(understanding, 'sentence_programs')
    assert not hasattr(answer, 'program')
    for field in (understanding.sentence_states[0], answer.sentence_states[0], recalled):
        assert not hasattr(field, 'actions') and not hasattr(field, 'leaves')
        assert field.refs == (1, 2, -1)
        torch.testing.assert_close(field.end_state[0], torch.tensor([1., 2., 3., 4.]), rtol=0, atol=0)


def test_closing_discards_operations_without_changing_other_live_rows():
    from Models import BasicModel

    lang = tuple(torch.ones(2, 4) for _ in range(25)) + (
        torch.ones(2, 4, dtype=torch.long), torch.ones(2, 4, dtype=torch.bool),
        torch.ones(2, 4))  # selected operation log-probabilities
    _stm, cleared, _feedback = BasicModel._discard_sentence_record(((), lang, ()),
                                                                  torch.tensor([True, False]))
    for index in (4, 5, 6, 7, 8, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27):
        expected = -1 if index in (4, 5, 15, 16, 17, 18, 23, 25) else 0
        assert (cleared[index][0] == expected).all()
        torch.testing.assert_close(cleared[index][1], lang[index][1], rtol=0, atol=0)
    for index in (0, 9, 13, 14, 20):
        torch.testing.assert_close(cleared[index], lang[index], rtol=0, atol=0)


def test_writer_takes_relation_kind_from_the_rel_identity():
    store, _ = clause_store()
    store.configure_clause_index(allocate=store._allocate_clause_row,
        concept_point=store._concept_point,
        predicate_kind=lambda reference: 'part' if reference == 3 else 'operator')
    # The reading's annotation is not authoritative at the row boundary.
    ended = replace(part_clause(), relation='operator')
    row = store.write_clause(ended)
    assert int(store.rel_type[row]) == store.REL_PARTOF
    assert ended.slots.shape == (3, 4)
    idea = idea_clause()
    assert idea.slots.shape == (1, 4)
    assert store.clause_relation(idea) is None


def test_idea_caches_the_operand_addresses_recorded_before_fusion():
    from ClauseJournal import finish_clause
    from Understanding import AnswerProgram

    language = SimpleNamespace(_compose_binary_rules=[SimpleNamespace(method_name='lift',
        clause_form='S', head_role=0, reference_kinds=(), lhs='S')], _compose_unary_rules=[])
    left, right, result = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    frames = torch.zeros(3, 3, 4)
    frames[2] = torch.stack((left, right, result))
    entry = AnswerProgram(rows=torch.tensor([0, 1]), word_rows=torch.tensor([0, 1]),
        activations=torch.ones(2), leaves=torch.stack((left, right)), concept_ids=torch.tensor([1, 2]),
        reference_ids=torch.tensor([7, 8]), actions=torch.tensor([[0, -1, 0], [0, -1, 1], [1, 0, -1]]),
        targets=torch.tensor([0, 1, 1]), end_state=torch.stack((result, result * 0, result * 0)),
        operation_values=frames, operation_refs=torch.tensor([[-1, -1], [-1, -1], [1, -1]]))
    ended = finish_clause(language, entry)
    assert ended.refs == (1, -1, -1)




def test_one_native_sentence_owns_only_its_end_state_after_the_boundary(tmp_path, monkeypatch):
    from test_meronomy_ladder import _build_ladder_variant
    import util

    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model = _build_ladder_variant(tmp_path, 'end_state_storage', [
        ('<architecture>', '<architecture><ltmConsolidation>true</ltmConsolidation>')])
    model._tensor_peer_while_eager = True
    model._chart_compose_per_word = lambda: None
    model._install_unit_span_fn()
    model.reconstruct_in_loop = False
    model.loss.reconstruction_scale = 0.
    try:
        inputs = model.inputSpace.prepPackedInput([['1 plus 2']])
        model.runBatch(train=False, batchSize=1, split='runtime',
                       batch_override=(inputs, torch.zeros(1, 1, 0)))
        ended = model._sentence_fields[0][0]
        store = model.symbolSpace.ltm_store
        row = store.index_of_row(ended.row_id)
        assert row is not None
        torch.testing.assert_close(ended.meaning.roles, store.slots[row], rtol=0, atol=0)
        trace = model._reconstruction_stack()
        assert not trace._choice_mask.any()
        assert not trace._choice_values.any()
        assert (trace._choice_refs == -1).all()
        assert model._open_sentence_slot is None
        assert not hasattr(ended, 'actions')
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()


from dataclasses import replace
import torch
from Language import LanguageSpace
from test_clause_acceptance import SentenceFixture


def test_program_meaning_reads_ended_points_from_their_rows(monkeypatch):
    f = SentenceFixture(monkeypatch)
    row = f.store.write_clause(f.clause('earlier'), trust=.8)
    identity = int(f.store.row_ids[row])
    assert f.cs._csw_row_of(identity) is None
    entry = f.program(('part', 'it', 'animals'))
    refs = entry.concept_ids.clone()
    refs[0] = identity
    entry = replace(entry, reference_ids=refs)
    meaning = LanguageSpace.program_meaning(f.language, entry, f.registry)
    assert meaning is not None
    torch.testing.assert_close(meaning.roles[0], f.store.point_of_row(identity))
    torch.testing.assert_close(meaning.roles[2], f.registry._payload(('sym', int(refs[1]))))


@pytest.mark.parametrize("unary", [False, True])
def test_deep_clause_retains_exact_values_references_and_gradients(unary):
    from ClauseJournal import finish_clause
    from Understanding import AnswerProgram
    words = 1200
    rows=[(0,-1,0)]
    if unary:
        rows.extend((2,0,-1) for _ in range(words))
        rows.extend(((0,-1,1),(1,0,-1)))
        words=2
    else:
        for w in range(1,words):rows.extend(((0,-1,w),(1,0,-1)))
    actions=torch.tensor(rows)
    frames=torch.ones(len(rows),3,2)
    frames[-1]=torch.tensor([[4.,5.],[6.,7.],[8.,9.]])
    end=torch.tensor([[8.,9.],[0.,0.],[0.,0.]],requires_grad=True)
    program=AnswerProgram(rows=torch.arange(words),word_rows=torch.arange(words),
        leaves=torch.ones(words,2),activations=torch.ones(words),actions=actions,
        concept_ids=torch.arange(1,words+1),targets=torch.zeros(len(rows),dtype=torch.long),
        end_state=end,operation_values=frames)
    language=SimpleNamespace(_compose_binary_rules=[SimpleNamespace(method_name='sum')],
        _compose_unary_rules=[SimpleNamespace(method_name='not')])
    result=finish_clause(language,program)
    torch.testing.assert_close(result.point,end[0],rtol=0,atol=0)
    torch.testing.assert_close(result.meaning.roles[:2],frames[-1,:2],rtol=0,atol=0)
    assert result.children==() and result.relation is None
    assert result.refs==(-1,words,-1)
    result.point.sum().backward()
    torch.testing.assert_close(end.grad,end.new_tensor([[1.,1.],[0.,0.],[0.,0.]]))
