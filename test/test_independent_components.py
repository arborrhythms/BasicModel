"""Item 6.5 mechanism certificates; no passing seed is selected."""
import itertools
from types import SimpleNamespace
from dataclasses import replace
from IdentityRows import witness_count

import pytest
import torch


def test_definedness_keeps_direction_input_uncertainty_and_evidence_separate():
    from MeaningCodes import defined_code, definedness, compose
    from Interpret import activate_code
    direction = torch.tensor([1., 2., 0.])
    counts = torch.tensor([0., 1., 4., 16., 10000.], dtype=torch.float64)
    magnitudes = definedness(counts)
    assert (magnitudes.diff() > 0).all()
    torch.testing.assert_close(1 - magnitudes, 4 / (counts + 4))
    for n, pair in itertools.product([0, 1, 4, 16], itertools.product([0., 1.], repeat=2)):
        code = defined_code(direction, n)
        m = n / (n + 4)
        torch.testing.assert_close(code.norm(), code.new_tensor(m))
        if n:
            torch.testing.assert_close(code / code.norm(), direction / direction.norm())
        atoms = torch.cat((torch.tensor([1., 0.]), code, torch.zeros(3)))[None]
        leaf = activate_code(atoms, torch.tensor([.7]), 2, evidence=torch.tensor([pair]))
        torch.testing.assert_close(leaf[0, :2], torch.tensor([.7, 0.]))
        torch.testing.assert_close(leaf[0, 2:5], code * pair[0])
        torch.testing.assert_close(leaf[0, 5:], code * pair[1])
        conjunction = compose('conjunction', leaf[:, 2:], torch.ones(1, 6))
        assert conjunction[:, :3].norm() <= m + 1e-7
        input_value = .3 * direction / direction.norm()
        torch.testing.assert_close(input_value @ code, code.new_tensor(.3 * m))


def test_row_definedness_survives_rewitness_and_checkpoint_without_changing_truth():
    from Layers import LTM0
    from Meaning import ConceptualMeaning
    from MeaningCodes import sentence_key
    store = LTM0(8, capacity=4)
    meaning = ConceptualMeaning.from_description(torch.ones(8))
    key = sentence_key([b'a', b'cat'])
    args = dict(document_key='fixture', sentence_index=0, content_key=key, evidence=(1., 1.))
    row = store.append_meaning(meaning, **args)
    first = store.identity_code(row, 64, 3)
    assert first.norm().item() == pytest.approx(1 / 5)
    store.append_meaning(meaning, **args)
    second = store.identity_code(row, 64, 3)
    torch.testing.assert_close(second / second.norm(), first / first.norm())
    assert second.norm().item() == pytest.approx(2 / 6)
    assert store.c_plus[row] == store.c_minus[row] == 1
    restored = LTM0(8, capacity=4)
    restored.load_state_dict(store.state_dict())
    torch.testing.assert_close(restored.identity_code(row, 64, 3), second)


def test_word_magnitude_counts_containing_rows_and_excludes_definition_rows(monkeypatch):
    from Layers import LTM0
    from Meaning import ConceptualMeaning
    from MeaningCodes import sentence_key
    from MereologicalCodes import MereologicalCodes
    from test_clause_acceptance import SentenceFixture
    f = SentenceFixture(monkeypatch)
    store, owner = f.store, f.cs
    a, b = f.noun('a'), f.noun('b')
    root = store.parts.add(a, b, truth_plus=1.)
    codes = MereologicalCodes(owner, torch.zeros(2, 20), percept_width=4, percept_event_width=4)
    for position in range(2):
        row = store.append_meaning(replace(ConceptualMeaning.from_description(torch.ones(8)),
            role_refs=(('sym', root), None, None)),
            document_key='fixture', sentence_index=position,
            content_key=sentence_key([str(position).encode()]))
    count, value = codes.occurrence_terms()[owner._csw_row_of(a)]
    assert count == 2 and value.norm().item() == pytest.approx(2 / 6)
    assert not value[8:].any() and not value.requires_grad


def dictionary(width=4, sources=2):
    from IndependentComponents import SparseDictionary
    return SparseDictionary(width, max_sources=sources, prior_scale=.05,
                            mint_threshold=.2, recurrence=4)


def witness(dictionary, value, start=0, count=4):
    for occurrence in range(start, start + count):
        dictionary.observe(value, witness=occurrence)


def test_change_column_admission_requires_recurring_innovation():
    columns = dictionary()
    first, second = torch.eye(4)[:2]
    witness(columns, first, count=3)
    assert columns.ids == ()
    columns.observe(first, witness=3)
    assert len(columns.ids) == 1
    original = columns.ids[0]
    witness(columns, second, start=4, count=3)
    assert columns.ids == (original,)
    columns.observe(second, witness=7)
    assert len(columns.ids) == 2
    stream = torch.stack((first, second, first, torch.zeros_like(first), second))
    result = columns.encode(stream)
    assert result.codes[:, 0].ne(0).tolist() == [True, False, True, False, False]
    assert result.codes[:, 1].ne(0).tolist() == [False, True, False, False, True]
    assert columns.ids[0] == original
    torch.testing.assert_close(columns.matrix().norm(dim=-1), torch.ones(2))


def test_shared_change_features_seed_only_the_unexplained_remainder():
    columns = dictionary()
    # Coordinate zero is shared change content; the next two are distinct effects.
    first = torch.tensor([1., 1., 0., 0.]) / 2**.5
    second = torch.tensor([1., 0., 1., 0.]) / 2**.5
    witness(columns, first)
    witness(columns, second, start=10, count=3)
    assert len(columns.ids) == 1
    columns.observe(second, witness=13)
    assert len(columns.ids) == 2
    codes = columns.encode(torch.stack((first, second, first, first + second))).codes
    # The second admission is the unexplained remainder, not another whole
    # change. Its known content still reuses the first column.
    assert codes.ne(0).tolist() == [[True, False], [True, True], [True, False], [True, True]]
    assert not torch.allclose(columns.matrix()[1], second)
    assert columns.encode(second).residual.norm() < columns.mint_threshold


def test_mint_threshold_duplicate_witness_and_relevance_do_not_delete_columns():
    columns = dictionary()
    first, second = torch.eye(4)[:2]
    for _ in range(9):
        columns.observe(first, witness='same occurrence')
    assert not columns.ids
    witness(columns, first)
    assert len(columns.ids) == 1
    witness(columns, .01 * second, start=10)
    assert len(columns.ids) == 1
    with torch.no_grad():
        columns.relevance[str(columns.ids[0])].zero_()
    assert columns.pruning_candidates() == columns.ids
    assert len(columns.ids) == 1
    assert not columns.encode(first[None]).codes.any()


def test_population_loss_is_batch_partition_invariant_and_has_one_backward():
    columns = dictionary()
    witness(columns, torch.eye(4)[0])
    witness(columns, torch.eye(4)[1], start=10)
    population = torch.tensor([[1., 0., .2, 0.], [0., .7, 0., .1], [1., .8, 0., 0.]])
    whole = columns.loss(population)
    singles = torch.stack([columns.loss(row[None]) for row in population]).mean()
    chunks = (columns.loss(population[:2]) * 2 + columns.loss(population[2:])) / 3
    torch.testing.assert_close(whole, singles)
    torch.testing.assert_close(whole, chunks)
    calls = []
    handle = columns.directions[str(columns.ids[0])].register_hook(lambda grad: calls.append(grad))
    whole.backward()
    handle.remove()
    assert len(calls) == 1 and calls[0].abs().sum() > 0


def test_change_dictionary_is_one_sparse_per_sentence():
    changes = dictionary(width=12, sources=1)
    first, second = torch.eye(12)[:2]
    witness(changes, first)
    witness(changes, second, start=10)
    result = changes.encode(torch.stack((first, second, first, first + second)))
    assert result.codes.ne(0).sum(-1).tolist() == [1, 1, 1, 1]
    assert result.codes[:3].argmax(-1).tolist() == [0, 1, 0]


def test_dictionary_checkpoint_preserves_pending_recurrence_and_learned_parameters():
    import copy
    source = dictionary()
    first, second = torch.eye(4)[:2]
    witness(source, first)
    witness(source, second, start=10, count=2)
    restored = dictionary()
    restored.load_state_dict(copy.deepcopy(source.state_dict()))
    assert restored.ids == source.ids
    torch.testing.assert_close(restored.encode(first[None]).codes, source.encode(first[None]).codes)
    witness(restored, second, start=12, count=2)
    assert len(restored.ids) == 2 and len(source.ids) == 1


def test_identity_candidates_share_the_operation_softmax_and_live_credit():
    from ReferenceContext import ReferenceBank, prepare_operands
    from Language import OperationSelectionLayer
    class Identity(torch.nn.Module):
        def forward(self, x):
            return x
    rule = SimpleNamespace(reference_roles=('argin1',),
                           reference_modes=(('argin1', 'particular'),))
    word = torch.tensor([[[0., 0., 1., 0.]]])
    columns = torch.eye(4)[:2][None]
    bank = ReferenceBank(torch.tensor([[11, 12]]), columns,
                         torch.ones(1, 2, dtype=torch.bool),
                         torch.zeros(1, 2, dtype=torch.bool), word[:, 0], torch.tensor([True]))
    live = (word, torch.tensor([[3]]), torch.tensor([[0]]),
            torch.tensor([[False]]), torch.tensor([[0]]))
    args = dict(orders=torch.tensor([[1]]), rules=(), unary_rules=(rule,), bank=bank, live=live,
                active=torch.tensor([[True]]))
    candidates = prepare_operands(word, torch.tensor([[3]]), torch.tensor([[0]]),
                                  torch.tensor([[0]]), **args)
    refs = candidates['unary_refs'][0, 0, :, 0]
    legal = candidates['unary_valid'][0, 0]
    assert set(refs[legal].tolist()) == {-1, 11, 12}
    layer = OperationSelectionLayer(d_model=4, ops=(), unary_ops=(Identity(),), chooser='mlp')
    for identity in (11, 12, -1):
        action = ((refs == identity) & legal).nonzero()[0, 0].reshape(1)
        from operation_fixtures import selected_action
        with selected_action(layer, action):
            _, _, route = layer(word, reference_data=candidates)
        assert route['refs'][0, 0] == identity and route['op'][0] == 0
        torch.testing.assert_close(route['probabilities'], route['logits'].softmax(-1))
        credit = route['logits'].log_softmax(-1).gather(1, action[:, None]).sum()
        grads = torch.autograd.grad(credit, tuple(layer.chooser.parameters()), allow_unused=True)
        assert any(g is not None and bool(g.abs().sum() > 0) for g in grads)
    # A pronoun has precisely the bounded noun columns, with no dictionary
    # decision or mint fallback. Selection remains the same global action.
    rule.reference_modes = (('argin1', 'pronoun'),)
    pronoun = prepare_operands(word, torch.tensor([[3]]), torch.tensor([[0]]),
                               torch.tensor([[0]]), **args)
    assert set(pronoun['unary_refs'][0, 0, pronoun['unary_valid'][0, 0], 0].tolist()) == {11, 12}


def test_event_chart_excludes_address_bands_and_preserves_evidence_poles():
    from IndependentComponents import EventChart
    # The active meaning width may be narrower than the reserved pole stride.
    codes = torch.zeros(2, 16)
    codes[0, 4], codes[1, 5] = 1., 1.
    chart = EventChart(torch.tensor([1, 3]), codes, capacity=6, content_width=14,
                       meaning_start=4, meaning_pairs=2, pole_stride=5)
    value = torch.arange(16, dtype=torch.float32, requires_grad=True)
    torch.testing.assert_close(chart.project(value), value[[4, 5, 9, 10]])
    chart.project(value).square().sum().backward()
    assert value.grad.nonzero().flatten().tolist() == [4, 5, 9, 10]
    restored = chart.render(chart.project(value))
    assert restored.nonzero().flatten().tolist() == [4, 5, 9, 10]


def component_fixture():
    from test_cs_sparse_weights import _cs
    from IdentityRows import mint
    owner = _cs(nS=32, order=2, dim=4)
    model = owner.components
    for value in torch.eye(4)[:2]:
        mint(owner, value)
    return model


def test_unretained_innovation_cannot_change_population_cost(monkeypatch):
    import Spaces
    from Layers import LTM0
    from Meaning import ConceptualMeaning
    columns = component_fixture()
    store = LTM0(4, capacity=8)
    meaning = ConceptualMeaning.from_description(torch.eye(4)[0])
    row = store.append_meaning(meaning, document_key='retained', sentence_index=0, kind='observation')
    owner = columns.owner
    owner._closed_clause_store = lambda: store
    owner._order0_inventory_row = lambda physical: physical < 4
    for i in range(4):
        cid = owner.new_concept()
        seat = owner._csw_concept_row(0, cid)
        with torch.no_grad(): owner.similarity_codebook.W[seat].copy_(torch.eye(4)[i])
    prediction = SimpleNamespace(roles=torch.zeros(3, 4))
    columns.begin_forward()
    expected = columns.cost([meaning], [prediction], torch.tensor([True]))
    retained = int(store.row_ids[row])
    absent = retained + 100
    columns._innovations[absent] = torch.ones(3 * columns.capacity)
    columns.begin_forward()
    actual = columns.cost([meaning], [prediction], torch.tensor([True]))
    torch.testing.assert_close(actual, expected)
    assert absent not in columns._innovations
    columns._innovations[retained] = torch.ones(3 * columns.capacity)
    columns.begin_forward()
    assert retained in columns._innovations
    assert not torch.allclose(columns.cost([meaning], [prediction], torch.tensor([True])), expected)


def test_continuity_is_binding_evidence_and_never_a_column_signature():
    from IndependentComponents import EventChart
    from ReferenceContext import SituationFrame
    from test_cs_sparse_weights import _cs
    from IdentityRows import mint
    owner = _cs(nS=32, order=2, dim=12)
    columns = owner.components
    value = torch.zeros(12); value[4] = 1.
    identity = mint(owner, value)
    object.__setattr__(owner.similarity_codebook, 'mereology', SimpleNamespace(
        percept_width=0, percept_event_width=4))
    identity = columns.individual_ids[0]
    original = columns.column_point(identity).detach().clone()
    frame = SituationFrame(1, torch.zeros(3, 12), torch.tensor([True, False, False]),
                           91, None, where=torch.tensor([2., 3.]), when=torch.tensor([8., 9.]))
    located = columns.column_point(identity)
    torch.testing.assert_close(located, original)
    # Changing only the retained model-time column cannot enter composition.
    frame = replace(frame, when=torch.tensor([99., 100.]))
    changed = columns.column_point(identity)
    torch.testing.assert_close(changed, located, rtol=0, atol=0)
    torch.testing.assert_close(columns.column_point(identity), original)


def test_temporal_shuffle_reports_loss_of_predictability_without_changing_identity(record_property):
    columns = dictionary(sources=1)
    first, second = torch.eye(4)[:2]
    witness(columns, first)
    witness(columns, second, start=10)
    sequence = torch.stack((first, first, first, second, second, second, first, first))
    shuffled = sequence[torch.tensor([0, 3, 1, 4, 6, 5, 2, 7])]
    series, control = columns.encode(sequence).codes, columns.encode(shuffled).codes
    record_property('ordered_persistence_error', float(series.detach().diff(dim=0).square().mean()))
    record_property('shuffled_persistence_error', float(control.detach().diff(dim=0).square().mean()))
    # This is a reported temporal diagnostic, not a learned-identity gate.
    torch.testing.assert_close(columns.loss(sequence), columns.loss(shuffled))


def test_native_admission_optimizer_ownership_and_checkpoint(tmp_path, monkeypatch):
    import copy
    from pathlib import Path
    import Language
    import Models
    import util
    from data import TheData
    from Spaces import _concept_alloc_of
    from IndependentComponents import EventChart
    source = Path(__file__).resolve().parents[1] / 'data' / 'XOR_grammar.xml'
    config = tmp_path / 'components.xml'
    config.write_text(source.read_text().replace('<nVectors>6</nVectors>', '<nVectors>32</nVectors>').replace('<architecture>',
        '<architecture><ltmConsolidation>true</ltmConsolidation>', 1))
    util.init_config(path=str(config), defaults_path=str(source.parent / 'model.xml'))
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    Language.TheGrammar._configured = False
    cfg = Models.BaseModel.load_config(str(config))
    TheData.load('xor', dat=dict(cfg['architecture']['data']))
    model, _ = Models.BaseModel.from_config(str(config), data=TheData)
    try:
        owner = model._concept_owner()
        columns = owner.components
        event = owner.new_concept()
        event_row = owner._csw_concept_row(0, event)
        width = owner.similarity_codebook.W.shape[-1]
        code = torch.zeros(1, width)
        derived = owner.similarity_codebook.mereology
        code[0, derived.percept_event_width] = 1.
        columns._chart = EventChart(torch.tensor([event_row]), code, capacity=owner.nVectors,
            content_width=owner.nWhat, meaning_start=derived.percept_event_width,
            meaning_pairs=derived.meaning_pairs, pole_stride=derived.reserved_pairs)
        from ClauseRow import attach_clause_index
        attach_clause_index(model, owner._closed_clause_store(), owner)
        from test_reference_close import close
        close(columns.reference_close, code[0], torch.zeros_like(code[0]), 'mint', (),
              document='native-columns', sentence=0)
        identity, = columns.individual_ids
        assert _concept_alloc_of(owner).order_of(identity) == 1
        assert owner._csw_row_of(identity) != event_row
        read = owner.similarity_codebook.lookup_rows(torch.tensor(owner._csw_row_of(identity)))
        torch.testing.assert_close(read[derived.percept_event_width:],
                                   columns.column_point(identity)[derived.percept_event_width:])
        optimizer = model.getOptimizer(lr=.01)
        owners = model.objective_parameter_groups(optimizer)
        ids = {id(p) for p in columns.parameters()}
        assert ids <= {id(p) for p in owners['expectation']}
        assert all(not ids.intersection(id(p) for p in params)
                   for name, params in owners.items() if name != 'expectation')
        state = copy.deepcopy(columns.state_dict())
        from IndependentComponents import IndependentComponents
        restored = IndependentComponents(owner, weight=.1, prior_scale=.05, mint_threshold=.2)
        restored.load_state_dict(state)
        assert restored.individual_ids == (identity,)
        torch.testing.assert_close(restored.column_point(identity), columns.column_point(identity))
        assert witness_count(owner, identity) == 1
        path = tmp_path / 'admitted-columns.ckpt'
        model.save_weights(path)
        restored_model, _ = Models.BaseModel.from_config(str(config), data=TheData)
        try:
            assert restored_model.load_weights(path, strict=True, require_match=True)
            restored_owner = restored_model._concept_owner()
            assert restored_owner.components.individual_ids == (identity,)
            assert restored_owner._csw_row_of(identity) == owner._csw_row_of(identity)
            assert witness_count(restored_owner, identity) == 1
            torch.testing.assert_close(restored_owner.components.column_point(identity),
                                       columns.column_point(identity))
        finally:
            restored_model.End()
            restored_model.symbolSpace.soft_reset()
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_known_column_in_new_sentence_witnesses_without_pending_prototype():
    columns = dictionary()
    known, novel = torch.eye(4)[:2]
    witness(columns, known)
    identity = columns.ids[0]
    before = columns.witness_count(identity)
    # A new occurrence can be wholly surprising while already explainable.
    columns.observe(known, witness=('new sentence', 1), prediction=torch.zeros(4))
    assert columns.witness_count(identity) == before + 1
    assert columns._pending == [] and columns.ids == (identity,)
    # A correct prediction has no surprise but still witnesses the whole row.
    columns.observe(known, witness=('new sentence', 2), prediction=known)
    assert columns.witness_count(identity) == before + 2
    assert columns._pending == []
    # Only novel surprise becomes a prototype; it needs distinct recurrences.
    for occurrence in range(3):
        columns.observe(known + novel, witness=('novel sentence', occurrence), prediction=known)
    assert columns.ids == (identity,) and len(columns._pending) == 1
    torch.testing.assert_close(columns._pending[0]['direction'], novel)
    columns.observe(known + novel, witness=('novel sentence', 3), prediction=known)
    assert len(columns.ids) == 2
    torch.testing.assert_close(columns.matrix()[1], novel)


def test_greedy_support_stops_when_an_extra_column_worsens_remainder():
    columns = dictionary(sources=2)
    a, b = torch.eye(4)[:2]
    witness(columns, a)
    witness(columns, b, start=10)
    # Relevance can over-amplify an explanation: the second column's best
    # projected coefficient increases the residual and must not be retained.
    with torch.no_grad():
        columns.relevance[str(columns.ids[1])].fill_(5.)
    encoded = columns.encode(a + .1 * b)
    assert encoded.codes.ne(0).tolist() == [True, False]
    assert encoded.residual.norm() < (a + .1 * b).norm()


def test_sparse_coding_is_confined_to_verbs_and_cannot_train_noun_signatures():
    from Meaning import ConceptualMeaning
    columns = component_fixture()
    assert not hasattr(columns, 'nouns')
    roles = torch.eye(4)[:3].clone().requires_grad_()
    meaning = ConceptualMeaning(roles, torch.ones(3, dtype=torch.bool))
    prediction = SimpleNamespace(roles=torch.zeros_like(roles))
    assert columns.cost([meaning], None, torch.tensor([True])).item() == 0
    innovation = columns.innovation(roles.detach(), prediction.roles)
    witness(columns.verbs, innovation)
    columns.cost([meaning], [prediction], torch.tensor([True])).sum().backward()
    assert roles.grad is not None and roles.grad.abs().sum() > 0
    assert all("nouns" not in name for name, _ in columns.named_parameters())
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in columns.verbs.parameters())
