"""A sentence tree retains its trace, its occurrence names and its live points."""
from types import SimpleNamespace

import torch

from OperationSeats import Seat
from SentenceStrata import SentenceStrata
from Understanding import Reading


def test_tree_reads_the_trace_without_accessing_relation_rows(monkeypatch):
    from Memory import LTM1
    from MemberRead import read

    def forbidden(*args, **kwargs):
        raise AssertionError('a sentence tree accessed a relation row')

    words = torch.arange(8, dtype=torch.float32).reshape(2, 4).requires_grad_()
    product = words[0] * words[1]
    frames = torch.stack((words.new_zeros(3, 4), words.new_zeros(3, 4),
                          torch.stack((words[0], words[1], product))))
    actions = torch.tensor([[0, 0, 0], [0, 0, 1], [1, 0, 0]])
    program = Reading(rows=torch.arange(2), word_rows=torch.arange(2),
                      activations=torch.ones(2), leaves=words, actions=actions,
                      targets=words, end_state=product, concept_ids=torch.tensor([11, 12]),
                      operation_values=frames)
    rule = SimpleNamespace(method_name='product', canonical='product', surface_name=None, kernel_index=None)
    seat = Seat(1, 'product', False, torch.mul)
    language = SimpleNamespace(parts_of_speech=SimpleNamespace(seats={1: seat, 'product': seat}),
                               _compose_binary_rules=(rule,), _compose_unary_rules=())
    strata = SentenceStrata(1, 2, 2, 4)
    owner = SimpleNamespace(stm=SimpleNamespace(strata=strata, _thought_stores=()))
    with monkeypatch.context() as patch:
        for name in ('add', 'references', 'contains'):
            patch.setattr(LTM1, name, forbidden)
        strata.capture(program, language, 0, after=10)
        assert program.actions.tolist() == [[0, -1, 0], [0, -1, 1], [1, 0, -1]]
        assert program.tree.forest() == ((1, 0, 2, (0, 0), (0, 1)),)
        node = program.tree.composition(action=2)
        assert node.address == 13 and [v.address for v in node.children] == [11, 12]
        torch.testing.assert_close(node.point, product)
        torch.testing.assert_close(read(owner, ('ltm', 'stm-v1', 13)), product)
        torch.testing.assert_close(program.leaves, words)
        assert program.tree.semantic_start == 14
    gradient, = torch.autograd.grad(node.point.sum(), words)
    torch.testing.assert_close(gradient, words.flip(0))
