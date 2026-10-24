"""Retained content selection, bilateral folds and unknown phase answers."""
import torch

from CaseSelection import CaseView


def view():
    # Three observed cases; each has its own positive and negative evidence.
    codes = torch.tensor([[[.2, .8, .1, .7],
                           [.6, .4, .5, .3],
                           [.9, .1, .8, .2]]], dtype=torch.float64)
    weights = torch.tensor([[[1., 0.], [1., .5], [1., 1.]]], dtype=torch.float64)
    return CaseView(codes, weights, torch.ones(1, 4, dtype=torch.float64),
                    torch.zeros(1, dtype=torch.bool))


def test_repeated_content_selection_is_idempotent_and_stacks():
    cases = view()
    first = torch.tensor([[.5, .8, .4, .9]], dtype=torch.float64)
    second = torch.tensor([[.7, .3, .9, .1]], dtype=torch.float64)
    selected = cases.select(first)
    torch.testing.assert_close(selected.select(first).fold(meaning_start=0),
                               selected.fold(meaning_start=0), rtol=0, atol=0)
    torch.testing.assert_close(selected.select(second).fold(meaning_start=0),
                               cases.select(torch.minimum(first, second)).fold(meaning_start=0),
                               rtol=0, atol=0)
    torch.testing.assert_close(cases.test, torch.ones_like(cases.test), rtol=0, atol=0)


def test_fold_keeps_bilateral_evidence_and_retained_value_gradients():
    cases = view()
    cases.codes.requires_grad_()
    # With no positive evidence, negative evidence remains negative content;
    # it cannot become a signed vector or positive support.
    weights = cases.weights.clone()
    weights[..., 0] = 0.
    cases = cases._replace(weights=weights)
    result = cases.fold(meaning_start=0)
    torch.testing.assert_close(result[..., :2], torch.zeros_like(result[..., :2]), rtol=0, atol=0)
    torch.testing.assert_close(result[..., 2:], torch.tensor([[.7, 7/30]], dtype=torch.float64))
    result.sum().backward()
    assert cases.codes.grad[..., :2].count_nonzero() == 0
    assert cases.codes.grad[0, 1:, 2:].abs().sum() > 0


def test_no_selection_is_unknown_and_partial_content_cannot_certify_completion():
    cases = view()
    request = cases.fold(meaning_start=0)
    assert cases.coverage(request, meaning_start=0).count_nonzero() == 0
    selected = cases.select(torch.tensor([[.2, .2, .1, .1]], dtype=torch.float64))
    assert (selected.coverage(request, meaning_start=0) < 1.).all()
    torch.testing.assert_close(cases.select(cases.test).coverage(request, meaning_start=0),
                               torch.ones(1, 2, dtype=torch.float64))


def test_case_view_compiles_without_a_memory_reader():
    cases = view()
    def select(test):
        selected = cases.select(test)
        return selected.fold(meaning_start=0), selected.coverage(cases.fold(meaning_start=0), meaning_start=0)
    test = torch.tensor([[.5, .8, .4, .9]], dtype=torch.float64, requires_grad=True)
    actual = torch.compile(select, backend='aot_eager', fullgraph=True)(test)
    expected = select(test)
    for observed, wanted in zip(actual, expected):
        torch.testing.assert_close(observed, wanted, rtol=0, atol=0)
    actual[0].sum().backward()
    assert test.grad is not None and test.grad.abs().sum() > 0


def test_paired_compound_uses_one_learned_test_and_preserves_its_head_order(monkeypatch):
    from test_meaning_kernels import paired
    paired.__wrapped__(monkeypatch)
    from Language import CompoundLayer, invoke_structural_face
    from Queries import StructuralGrammarContext, ConceptualSpaceCapability
    from ReferenceContext import result_order
    op = CompoundLayer().double()
    codes = torch.tensor([[[.1, .2, .3, .4, 1., 0., 0., 0., 0., 1., 0., 0.],
                           [.2, .3, .4, .5, 0., 1., 0., 0., 0., 0., 1., 0.]]], dtype=torch.float64)
    cases = CaseView(codes, torch.ones(1, 2, 2, dtype=torch.float64),
                     torch.ones(1, 12, dtype=torch.float64), torch.zeros(1, dtype=torch.bool))
    modifier = codes[:, 1].clone().requires_grad_()
    head = torch.cat((codes[..., :4].mean(1), cases.fold(meaning_start=4)), -1)
    context = StructuralGrammarContext((), ConceptualSpaceCapability(12), None, 'compose', case_view=cases)
    result = invoke_structural_face(op, (modifier, head), context=context)
    torch.testing.assert_close(result[..., 4:], codes[:, 1, 4:], rtol=0, atol=0)
    torch.testing.assert_close(result[..., :4], (modifier[..., :4]+head[..., :4])/2)
    selected = op.select_view(modifier, cases)
    second = op.compose(modifier, result, case_view=selected)
    torch.testing.assert_close(second[..., 4:], result[..., 4:], rtol=0, atol=0)
    assert result_order(op, (0, 2)) == 2
    assert all(name.startswith('meaning_kernel.') for name, _ in op.named_parameters())
    result[..., 4:].sum().backward()
    assert op.meaning_kernel.bias.grad is not None
    assert modifier.grad is not None


def test_case_flow_retains_the_selected_head_through_two_parse_steps(monkeypatch):
    from types import SimpleNamespace
    from test_meaning_kernels import paired
    paired.__wrapped__(monkeypatch)
    from CaseSelection import CaseBank
    import CaseFlow
    from Language import CompoundLayer
    op = CompoundLayer().double()
    codes = torch.zeros(4, 12, dtype=torch.float64)
    codes[1, 4] = codes[1, 8] = 1.
    codes[2, 5] = codes[2, 9] = 1.
    codes[0] = (codes[1]+codes[2])/2
    codes[3] = codes[2]
    members = torch.zeros(4, 4, 2, dtype=torch.float64)
    members[0, 1:3] = 1.
    bank = CaseBank(torch.tensor([10, 20, 30, 40]), codes, members, torch.zeros_like(members))
    state = CaseFlow.empty(codes.new_zeros(1, 3, 12), cases=4, journal=2)
    live = torch.tensor([True])
    for identity in (40, 40, 10):
        state = CaseFlow.push(state, bank, torch.tensor([identity]), live)
    rule = SimpleNamespace(head_role=2, case_head_role=2, clause_form=None)
    choice = SimpleNamespace(kind=torch.tensor([1]), local_op=torch.tensor([0]),
        position=torch.tensor([0]), applied=live)
    operands = codes[[3, 0]][None]
    def step(*values):
        return CaseFlow.apply(values, bank, choice, operands, (rule,), (), (op,), column=torch.tensor([0]))
    first = torch.compile(step, backend='aot_eager', fullgraph=True)(*state)
    second = CaseFlow.apply(first, bank, choice, operands, (rule,), (), (op,), column=torch.tensor([1]))
    positions = torch.tensor([[0]])
    a, b = (CaseFlow.view(item, bank, positions) for item in (first, second))
    torch.testing.assert_close(a.fold(meaning_start=4), b.fold(meaning_start=4), rtol=0, atol=0)
    torch.testing.assert_close(b.fold(meaning_start=4), codes[2:3, None, 4:], rtol=0, atol=0)
    assert first[4].tolist() == [[True, False]] and second[4].tolist() == [[True, True]]
    assert not state[2].any()


def test_absent_marker_face_shares_the_test_and_retains_one_form(monkeypatch):
    from test_meaning_kernels import paired
    paired.__wrapped__(monkeypatch)
    from Language import CompoundLayer, CompoundUnaryLayer
    from CaseSelection import CaseBank, inverse_unary
    base = CompoundLayer().double()
    unary = CompoundUnaryLayer(base).double()
    codes = torch.zeros(3, 12, dtype=torch.float64)
    codes[:, :4] = torch.tensor([[.2, .3, .4, .5], [.5, .4, .3, .2], [.1, .6, .2, .7]])
    codes[1, 4] = codes[1, 8] = 1.
    codes[2, 5] = codes[2, 9] = 1.
    codes[0, 4:] = (codes[1, 4:]+codes[2, 4:])/2
    weights = torch.zeros(3, 3, 2, dtype=torch.float64)
    weights[0, 1:] = 1.
    bank = CaseBank(torch.tensor([11, 12, 13]), codes, weights, torch.zeros_like(weights))
    cases = bank.view(torch.tensor([11]))
    with torch.no_grad():
        base.meaning_kernel.bias.fill_(.03)
    first = unary.select_view(codes[:1], cases)
    second = unary.select_view(codes[:1], first)
    torch.testing.assert_close(first.fold(meaning_start=4), second.fold(meaning_start=4), rtol=0, atol=0)
    value = unary.compose(codes[:1], case_view=cases)
    torch.testing.assert_close(value[:, :4], codes[:1, :4], rtol=0, atol=0)
    restored, available = inverse_unary(bank, value, unary)
    assert available.tolist() == [True] and restored.shape == value.shape
    torch.testing.assert_close(restored, codes[:1], rtol=0, atol=0)
    assert unary.meaning_kernel is base.meaning_kernel
    assert len(tuple(unary.parameters())) == len(tuple(base.parameters()))


def test_retained_case_inverse_recovers_form_without_inventing_discarded_content(monkeypatch):
    from types import SimpleNamespace
    from test_meaning_kernels import paired
    paired.__wrapped__(monkeypatch)
    from Language import CompoundLayer
    from DerivationReconstruction import inverse_step
    op = CompoundLayer().double()
    operands = torch.linspace(.1, .8, 24, dtype=torch.float64).reshape(2, 12)
    parent = op.compose_blocks(None, *operands, counts=(1, 3))[None]
    language = SimpleNamespace(_tree_layer=lambda arity: SimpleNamespace(ops=(op,)))
    left, right = inverse_step(language, 1, 0, parent, operands, counts=(1, 3))
    torch.testing.assert_close(left[0], operands[:1], rtol=0, atol=0)
    torch.testing.assert_close(right[0][:, :4], operands[1:, :4])
    assert left[1].all() and right[1][:, :4].all()
    assert not right[1][:, 4:].any() and not right[0][:, 4:].any()


def test_accessible_event_stages_before_change_after_with_bilateral_memberships(monkeypatch):
    from test_modal_timelines import fixture, recording, close
    from CaseSelection import CaseBank, stage_event_cases
    f, action, modifier = fixture(monkeypatch)
    _, frame, row, _ = close(f, recording(f, ('lift', ('lower_unary', 'subject'), 'move')),
        actions=(action, modifier), evidence=(.8, .2))
    token, entry = next(iter(f.cs.mtm.entries.items()))
    event = entry.events[0]
    verb = f.words['move'][1]
    code = f.cs.similarity_codebook.W[f.cs._csw_row_of(verb)].clone()
    bank = CaseBank(torch.tensor([verb]), code[None, None], code.new_zeros(1, 1, 2),
        code.new_zeros(1, 1, 2), torch.ones(1, 1, dtype=torch.bool))
    before = f.cs.mtm.state()
    staged = stage_event_cases(bank, f.cs.components.reference_close, [(frame,)], limit=4)
    assert staged.codes.shape == (1, 4, 8)
    for actual, expected in zip(staged.codes[0, 1:], (event.before, code, event.after)):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(staged.cases[0, 0, 1:], code.new_tensor([[.8, .2]]).expand(3, -1))
    assert staged.eligible.tolist() == [[True, False, False, False]]
    torch.testing.assert_close(staged.view(torch.tensor([verb])).reference[0], entry.value)
    assert stage_event_cases(bank, f.cs.components.reference_close, [(frame,)], limit=3) is bank
    from test_modal_timelines import same
    same(f.cs.mtm.state(), before)


def test_phase_question_uses_selected_content_and_never_the_event_clock(monkeypatch):
    import pytest
    from dataclasses import replace
    from test_modal_timelines import fixture, recording, close
    from Meaning import ConceptualMeaning
    from Queries import ThoughtLTMCapability
    from QueryWork import QueryWorkBudget
    f, action, modifier = fixture(monkeypatch)
    _, frame, row, _ = close(f, recording(f, ('lift', ('lower_unary', 'subject'), 'move')),
        actions=(action, modifier), evidence=(.8, .2))
    token, entry = next(iter(f.cs.mtm.entries.items()))
    event = entry.events[0]
    wanted = torch.tensor([1., 1., 0., 0., 1., 1., 0., 0.])
    partial = torch.tensor([1., 0., 0., 0., 1., 0., 0., 0.])
    reader = ThoughtLTMCapability(store=lambda: f.store, equal=lambda a,b: 1., tau_id=.9)
    def ask(content):
        from TimelineRead import phase_request
        question = replace(ConceptualMeaning.from_description(entry.value), bindings={
            '_query_event': (token, event.verb, 0)})
        question = phase_request(question, content)
        return reader.truth_evidence(question, max_records=16, work=QueryWorkBudget(32))
    assert ask(wanted)['support_true'] == ask(wanted)['support_false'] == 0
    f.cs.mtm.event(token, event.occurrence, verb=event.verb, role=0, when=2, case_value=partial)
    assert ask(wanted)['support_true'] == ask(wanted)['support_false'] == 0
    first = ask(partial)
    assert first['support_true'] == pytest.approx(.8) and first['support_false'] == pytest.approx(.2)
    f.cs.mtm.event(token, event.occurrence, verb=event.verb, role=0, when=20000, case_value=partial)
    second = ask(partial)
    assert second['support_true'] == first['support_true'] and second['support_false'] == first['support_false']
    f.cs.mtm.forget(token)
    expired = ask(partial)
    assert expired['support_true'] == 0 and expired['incomplete'] == ('phase_history_expired',)


def test_phase_content_checkpoint_copies_and_rejects_invalid_values():
    from copy import deepcopy
    import pytest
    from test_modal_timelines import pool, address, same
    from MediumTermMemory import MediumTermMemory
    p = pool()
    value = torch.full((8,), .3)
    p.event(address(1), address(2), verb=12, role=0, when=2, case_value=value)
    saved = p.state()
    restored = MediumTermMemory()
    restored.restore(saved)
    same(restored.state(), saved)
    value.zero_()
    assert restored.entries[address(1)].events[0].case_value.sum() > 0
    for bad_value in (torch.ones(7), torch.full((8,), 1.1), torch.full((8,), float('nan'))):
        invalid = deepcopy(saved)
        event = invalid['entries'][0]['timelines'][0][0][0]
        invalid['entries'][0]['timelines'] = ((((*event[:-1], bad_value),), torch.full((8,), .2)),)
        before = restored.state()
        with pytest.raises(ValueError):
            restored.restore(invalid)
        same(restored.state(), before)
    for version in (5, 6):
        legacy = deepcopy(saved)
        legacy['version'] = version
        entry = legacy['entries'][0]
        entry['timelines'] = tuple((tuple(event[:-1] for event in history), value)
                                  for history, value in entry['timelines'])
        restored.restore(legacy)
        assert restored.entries[address(1)].events[0].case_value is None


def test_close_transports_phase_content_as_a_query_constituent(monkeypatch):
    from dataclasses import replace
    from test_modal_timelines import fixture, recording, close
    from ThoughtClosing import closing_question
    from ThoughtReferences import bindings, validate_local_references
    f, action, modifier = fixture(monkeypatch)
    _, frame, row, _ = close(f, recording(f, ('lift', ('lower_unary', 'subject'), 'move')),
        actions=(action, modifier), evidence=(1., 0.))
    program = recording(f, ('lift', ('bind', 'subject'), 'move'), (frame,))
    values = torch.zeros(len(program.actions), 8)
    selected = torch.zeros(len(program.actions), dtype=torch.bool)
    root = len(program.actions)-1
    values[root, 1] = values[root, 5] = .6
    selected[root] = True
    program = replace(program, case_values=values, case_selected=selected)
    clause, _, _, written = close(f, program, (frame,), position=1, question=True)
    query = closing_question(clause.meaning, clause, store=f.store, written_rows=written)
    validate_local_references(query)
    data = bindings(query)
    assert len(data['_query_event']) == 3
    kind, index = data['_query_case']
    assert kind == 'constituent'
    torch.testing.assert_close(query.constituents[index].roles[0], values[root])
    assert all(event.case_value is None for entry in f.cs.mtm.entries.values() for event in entry.events)


def test_implicit_selection_uses_current_content_and_explicit_selection_uses_its_object(monkeypatch):
    from test_meaning_kernels import paired
    paired.__wrapped__(monkeypatch)
    from Language import CompoundLayer, CompoundUnaryLayer
    base = CompoundLayer().double()
    unary = CompoundUnaryLayer(base).double()
    codes = torch.zeros(1, 3, 12, dtype=torch.float64)
    for i in range(3):
        codes[0, i, 4+i] = codes[0, i, 8+i] = 1.
    cases = CaseView(codes, torch.ones(1, 3, 2, dtype=torch.float64),
        torch.ones(1, 12, dtype=torch.float64), torch.zeros(1, dtype=torch.bool), codes[:, 2])
    current = unary.compose(codes.mean(1), case_view=cases)
    referred = base.compose(codes[:, 0], codes.mean(1), case_view=cases)
    torch.testing.assert_close(current[:, 4:], codes[:, 2, 4:], rtol=0, atol=0)
    torch.testing.assert_close(referred[:, 4:], codes[:, 0, 4:], rtol=0, atol=0)
    changed = cases._replace(reference=codes[:, 1])
    torch.testing.assert_close(unary.compose(codes.mean(1), case_view=changed)[:, 4:], codes[:, 1, 4:], rtol=0, atol=0)
