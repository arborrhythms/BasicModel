"""The compiled reducer carries clause scope and local references per slot."""
from types import SimpleNamespace

import torch

from Language import Grammar


def scope():
    from ClauseScope import ClauseScope
    grammar = Grammar()
    grammar.load_from_grammar_file('complete.grammar')
    binary = [r for r in grammar.rules_upward if r.arity == 2 and r.space_role == 'CS']
    unary = [r for r in grammar.rules_upward if r.arity == 1]
    return ClauseScope(binary, unary), binary, unary


def choice(rules, name, *, unary=False, position=0):
    return SimpleNamespace(kind=torch.tensor([2 if unary else 1]), position=torch.tensor([position]),
        local_op=torch.tensor([next(i for i, r in enumerate(rules) if r.method_name == name)]),
        applied=torch.tensor([True]))


def test_13_relative_clause_pushes_a_scoped_reference_and_taints_its_enclosing_s():
    policy, binary, _ = scope()
    state = policy.empty(torch.zeros(1, 8, 8))
    for ref in (10, 11, 12, 13):  # he said cats animals
        state = policy.push(state, torch.tensor([True]), torch.tensor([ref]))
    state, ended = policy.apply(state, choice(binary, 'part'), torch.tensor(0))
    assert ended.tolist() == [3]
    assert int(state[0, 0, 1]) == -2  # trial-local reference; no durable ID was minted
    state, ended = policy.apply(state, choice(binary, 'verb'), torch.tensor(1))
    assert ended.tolist() == [0]
    assert policy.slots(state, torch.tensor([2])).tolist() == [1]
    assert int(state[0, 0, 0]) & policy.RELATIVE
    state, ended = policy.apply(state, choice(binary, 'lift'), torch.tensor(2))
    assert ended.tolist() == [3] and int(state[0, 0, 1]) == -4


def test_absolute_embedding_pushes_its_point_without_a_reference():
    policy, binary, _ = scope()
    state = policy.empty(torch.zeros(1, 8, 8))
    for ref in (10, 11, 12, 13):
        state = policy.push(state, torch.tensor([True]), torch.tensor([ref]))
    state, ended = policy.apply(state, choice(binary, 'lift'), torch.tensor(0))
    assert ended.tolist() == [1] and int(state[0, 0, 1]) == -1
    state, _ = policy.apply(state, choice(binary, 'verb'), torch.tensor(1))
    state, ended = policy.apply(state, choice(binary, 'lift'), torch.tensor(2))
    assert ended.tolist() == [1] and policy.slots(state, torch.tensor([1])).tolist() == [1]


def test_clause_scope_is_fullgraph_and_keeps_other_batch_rows():
    policy, binary, _ = scope()
    state = policy.empty(torch.zeros(1, 8, 8))
    for ref in (1, 2):
        state = policy.push(state, torch.tensor([True]), torch.tensor([ref]))
    op = choice(binary, 'part')
    expected = policy.apply(state, op, torch.tensor(7))
    compiled = torch.compile(policy.apply, backend='eager', fullgraph=True)
    actual = compiled(state, op, torch.tensor(7))
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_committed_stm_uses_one_idea_slot_or_three_relative_slots():
    from Models import BasicModel
    from test_item7_storage import idea_clause, part_clause
    buffer = torch.randn(2, 8, 4)
    stm = (buffer, torch.ones(2, dtype=torch.long),
           torch.zeros(2, 8, dtype=torch.long), torch.zeros(2, 8, dtype=torch.long),
           torch.full((2, 8), -1, dtype=torch.long), torch.ones(2, 8))
    lang = [torch.zeros(2)] * 22
    lang[9] = torch.zeros(2, 1, 4)
    lang[13] = torch.zeros(2, 1, 12)
    lang[14] = torch.ones(2, 1, dtype=torch.long)
    lang[20] = torch.zeros(2, 8, 2, dtype=torch.long)
    idea, relation = idea_clause(), part_clause()
    state = BasicModel._clause_end_state((stm, tuple(lang), ()), 0, (idea, relation), (10, 11))
    assert state[0][1].tolist() == [1, 3]
    torch.testing.assert_close(state[0][0][0, 0], idea.point)
    torch.testing.assert_close(state[0][0][1, :3], relation.meaning.roles[[2, 0, 1]])
    assert not state[0][0][0, 1:].any()
    assert state[1][14].tolist() == [[1], [3]]
