"""Fork, exact alternatives and shape contracts for the second repair."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch


def test_no_legal_thought_operation_finishes_with_the_reference_open(monkeypatch):
    from test_item6_2_thinking import world, run
    from ThoughtReferences import question, open_slots
    import ThoughtStream
    model, registry, _store, (a, _b, c) = world()
    goal = question(registry.form('isPart', a, c))
    monkeypatch.setattr(ThoughtStream, 'candidates', lambda *_args, **_kwargs: ())
    result = run(model, goal, work_budget=32)
    assert open_slots(result.meaning) == open_slots(goal)
    assert result.evidence['incomplete'] == ('no_legal_operation',)
    assert [record.kind for record in result.records] == ['begin', 'finish']
    assert all(record.operation != 'conclude' for record in result.records)
    assert result.work.spent == 1


def test_rebuilt_reference_bank_is_detached_at_the_fork():
    from ReferenceContext import ReferenceBank, ReferenceTypes
    point = torch.randn(2, 3, 4, requires_grad=True)
    ids = torch.arange(6).reshape(2, 3)
    interpreter = object()
    bank = ReferenceBank(ids, point, ids >= 0, ids < 0, point[:, 0], ids[:, 0] >= 0,
        ReferenceTypes(ids, torch.tensor([1]), ids[:, :, None], point[:, :, None]),
        interpreter=interpreter)
    fork = bank.detached()
    assert type(fork) is ReferenceBank and type(fork.types) is ReferenceTypes
    assert fork.interpreter is interpreter
    assert not fork.values.requires_grad and not fork.query.requires_grad
    assert not fork.types.values.requires_grad
    torch.testing.assert_close(fork.values, point)
    with torch.no_grad():
        point.add_(1)
    assert not torch.equal(fork.values, point)


def test_binding_layout_is_row_local_with_all_masked_candidates():
    from Interpret import InterpretLayer
    value = torch.randn(2, 1, 4)
    arguments = dict(identity=torch.tensor([[19], [19]]), scope=torch.zeros(2, 1, dtype=torch.long),
        candidate_ids=torch.tensor([[[19, 31]], [[19, 31]]]),
        candidate_values=torch.randn(2, 1, 2, 4), candidate_relations=torch.zeros(2, 1, 2, dtype=torch.bool),
        available=torch.ones(2, 1, 2, dtype=torch.bool), associations=torch.empty(0, 2, dtype=torch.long),
        mode=None, active=torch.ones(2, 1, dtype=torch.bool))
    first = InterpretLayer.bind(None, value, **arguments)
    arguments['available'][1].zero_()
    second = InterpretLayer.bind(None, value, **arguments)
    arguments['available'][0].zero_()
    third = InterpretLayer.bind(None, value, **arguments)
    assert len(first) == len(second) == len(third) == 5
    for a, b, c in zip(first, second, third):
        torch.testing.assert_close(a[1], b[1])
        torch.testing.assert_close(a[1], c[1])
        torch.testing.assert_close(a[3][0], b[3][0])
    assert [bool(option[3][0, 0]) for option in third] == [False, False, True, True, False]


@pytest.mark.parametrize('repeated', [False, True])
@pytest.mark.parametrize('padded', [False, True])
def test_large_menu_exact_representatives_match_pairwise_contract(monkeypatch, repeated, padded):
    from Language import OperationSelectionLayer
    B, N, D, R = 4, 4, 4, 50
    x = torch.randn(B, N, D)
    depth = torch.tensor([4, 3, 1, 0])
    if repeated:
        x[:, 1:] = x[:, :1]
    if padded:
        x = torch.where((torch.arange(N)[None] < depth[:, None])[..., None], x, 0.)
    nb, nu = (N - 1) * R, N * R
    candidates = torch.randn(B, nb + nu + 1, D)
    candidates[:, 1:20] = candidates[:, :1]
    candidates[:, 111:125] = x[:, 0, None]
    for position in range(N-1):
        candidates[:, position*R+30:position*R+40] = x[:, position, None]
        candidates[:, position*R+40:position*R+50] = x[:, position+1, None]
    for position in range(N):
        candidates[:, nb+position*R:nb+position*R+20] = x[:, position, None]
    logits = torch.randn(B, nb + nu + 1)
    logits[0, 1:4] = -torch.inf
    ids = torch.full((B, nb + nu + 1, 2), -1, dtype=torch.long)
    ids[:, 3:10, 0] = 2**60
    ids[:, 10:20, 0] = 2**60 + 1  # distinct even at identical float payload
    args = (x, depth, candidates, logits, torch.tensor([0, 1, nb+1, nb+nu]), nb, nu, R, R)
    kwargs = dict(reference_ids=ids, stop_exact=torch.tensor([False, True, True, False]))
    grouped = OperationSelectionLayer._distinct_departures(*args, **kwargs)
    monkeypatch.setattr(torch.compiler, 'is_compiling', lambda: True)
    paired = OperationSelectionLayer._distinct_departures(*args, **kwargs)
    assert torch.equal(grouped, paired)


def test_reference_variant_batching_preserves_values_and_gradients():
    from Language import OperationSelectionLayer, ContextualBindLayer, _BinaryGrammarOpAdapter
    class Product(torch.nn.Module):
        def forward(self, left, right):
            return left * right
    layer = OperationSelectionLayer(d_model=4,
        ops=[Product(), _BinaryGrammarOpAdapter(ContextualBindLayer(4, 4))])
    x = torch.randn(2, 3, 4, requires_grad=True)
    left, right = [torch.randn(2, 2, 6, 4, requires_grad=True) for _ in range(2)]
    operations = (0, 1, 0, 1, 0, 1)
    refs = dict(left=left, right=right, binary_ops=operations)
    grouped = layer._stacked_reduced(x, reference_data=refs)
    per_column = []
    for column, operation in enumerate(operations):
        per_column.append(layer._stacked_reduced(x, reference_data=dict(
            binary_ops=(operation,), left=left[:, :, column:column+1],
            right=right[:, :, column:column+1])))
    separate = torch.cat(per_column, 2)
    torch.testing.assert_close(grouped, separate, rtol=0, atol=0)
    a = torch.autograd.grad(grouped.square().sum(), (x, left, right), retain_graph=True)
    b = torch.autograd.grad(separate.square().sum(), (x, left, right))
    for first, second in zip(a, b):
        torch.testing.assert_close(first, second)


def test_pending_query_shape_keeps_its_missing_operand_open():
    from Meaning import ConceptualMeaning
    from ThoughtReferences import open_slots, with_slots
    value = ConceptualMeaning(torch.randn(3, 4), torch.tensor([True, True, False]),
        role_refs=(('sym', 1), ('sym', 2), None), sentence_kind='relation')
    value = with_slots(value, (('relation', 1),), pair=(1., 0.))
    assert open_slots(value) == (('referent', 2),)


def test_expectation_cannot_change_departure_or_keep():
    from SentenceCredit import comparison
    parts = torch.tensor([[[2., 100., 0.], [2., 0., 0.]],
                          [[2., 100., 0.], [3., 0., 0.]],
                          [[2., 0., 2.], [2., 100., 0.]]])
    audit = comparison(parts, torch.ones(3, dtype=torch.bool))
    assert audit['costs'].tolist() == [[2., 2.], [2., 3.], [4., 2.]]
    assert audit['wins'].tolist() == [False, False, False]
    assert not audit['delta'][:, 1].eq(0).any()
    assert audit['deciding'] == ['tie:greedy', 'reconstruction', 'answer']


def test_eight_corpus_sentences_train_at_distinct_fork_rounds(tmp_path, eager_reading):
    from math_chain_corpus import MathChainCorpus
    from math_chain_ordinary import ordinary_model, train_documents, episode_observer
    model = ordinary_model(tmp_path)
    footprint = episode_observer()
    differences = []
    def episode(original, model, *args, **kwargs):
        before = footprint.snapshot(model) if not differences else None
        result = original(model, *args, **kwargs)
        if before is not None:
            differences.append(footprint.difference(before, footprint.snapshot(model)))
        return result
    try:
        docs = MathChainCorpus().counting()[::2][:8]
        observations = train_documents(model, docs, tmp_path, episode=episode)
        assert len(observations) == 8
        assert len({row['departure'] for row in observations if row['departure'] >= 0}) > 1
        assert differences, 'no ordinary episode exercised the footprint'
        trail = {'.'+name for name in ('_last_thought_comparison', '_last_thought_score_function',
                 '_thought_ordinals', '_walk_audit', '_walk_observations', '_walk_previous')}
        trail.update('what_memory.'+name for name in (
            '_address_sources', '_episode_live', '_thought_next_id', '_what_slots', '_what_closure_pressure'))
        unexpected = [key for key in differences[0] if '._ltm_store.' not in key and key not in trail]
        assert not unexpected, unexpected
        assert not any(hasattr(model, key) for key in (
            '_open_thought_rows', '_pending_thought_credit', '_closing_images'))
    finally:
        model.End()


def test_determiner_scope_cannot_reopen_at_an_enclosing_bare_operation():
    from Interpret import InterpretLayer
    from ClauseScope import ClauseScope
    value = torch.randn(2, 1, 4)
    options = InterpretLayer.bind(None, value, identity=torch.tensor([[19], [19]]),
        scope=torch.full((2, 1), ClauseScope.DETERMINED, dtype=torch.long),
        candidate_ids=torch.tensor([[[31]], [[31]]]), candidate_values=value[:, :, None],
        candidate_relations=torch.zeros(2, 1, 1, dtype=torch.bool),
        available=torch.ones(2, 1, 1, dtype=torch.bool), mode=None, associations=None,
        active=torch.ones(2, 1, dtype=torch.bool))
    assert [bool(option[3][0, 0]) for option in options] == [False, False, False, True]
    assert int(options[-1][1][0, 0]) == 19
