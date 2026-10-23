"""Fixed-operation identity certificates; no claim about learned parsing."""
from types import SimpleNamespace

import pytest
import torch

from MediumTermMemory import MediumTermMemory
from MTMOverlay import MTMOverlay
from Spaces import LanguageOperationChoice


def address(number):
    return 'ltm', 'address-v1', number


def operations():
    return MTMOverlay((SimpleNamespace(method_name='lift', clause_form='S'),),
                      (SimpleNamespace(method_name='bind', head_role=1),))


def choice(value, *, kind=2, identity=0):
    batch = len(value)
    ids = torch.zeros(batch, dtype=torch.long)
    active = torch.ones(batch, dtype=torch.bool)
    return LanguageOperationChoice(value, ids+kind, ids, ids, active,
        value.new_ones(batch), ids, active, order=ids+1, identity=ids+identity)


def push(overlay, state, value, *, word, index, order=1, resolved=True):
    return overlay.push(state, torch.ones(len(value), dtype=torch.bool), value,
        word=torch.full((len(value),), word), order=torch.full((len(value),), order),
        index=torch.tensor(index), words=8, resolved=torch.full((len(value),), resolved))


def start(value):
    overlay = operations()
    state = overlay.empty(value[:, None].expand(-1, 8, -1), journal=16, words=8)
    state = push(overlay, state, value, word=101, index=0)
    state = overlay.apply(state, choice(value), torch.stack((value, value*0), 1),
                          column=torch.tensor(0), words=8)
    return overlay, state


def event(overlay, state, value, *, index, column):
    state = push(overlay, state, torch.ones_like(value)*.2, word=202, index=index, order=0)
    before = overlay.read(state[0], state[2][:, 1, 0:1])[:, 0]
    return overlay.apply(state, choice(value, kind=1),
        torch.stack((before, torch.ones_like(value)*.2), 1),
        column=torch.tensor(column), words=8)


@pytest.mark.parametrize('compiled', (False, True))
def test_second_verb_receives_first_result_through_bind_with_derivative(compiled):
    first = torch.tensor([[.2, .7, .3]], requires_grad=True)
    scale = torch.tensor(.6, requires_grad=True)
    overlay, state = start(first)
    if compiled:
        overlay.apply = torch.compile(overlay.apply, backend='eager', fullgraph=True)
    changed = first * scale
    state = event(overlay, state, changed, index=1, column=1)
    mention = torch.zeros_like(first)
    state = push(overlay, state, mention, word=303, index=2)
    points, records, handles, _, available = overlay.menu(state, mention[:, None], torch.tensor([[0]]))
    assert available[0, 0, 1] and handles[0, 0, 1] == 1
    torch.testing.assert_close(records[:, 0, 1], changed)
    bound = points[:, 0, 1]
    state = overlay.apply(state, choice(bound, identity=1),
        torch.stack((mention, records[:, 0, 1]), 1), column=torch.tensor(2), words=8)
    second_scale = torch.tensor(.4, requires_grad=True)
    received = overlay.refresh(state, mention[:, None], torch.tensor([[0]]))[:, 0]
    second = received * second_scale
    state = event(overlay, state, second, index=3, column=3)
    torch.testing.assert_close(state[0][:, 0], first * scale * second_scale)
    state[0][:, 0].sum().backward()
    torch.testing.assert_close(first.grad, torch.full_like(first, .24))
    torch.testing.assert_close(scale.grad, (first.detach()*second_scale.detach()).sum())
    assert second_scale.grad > 0


def test_latest_verb_replaces_state_and_two_aliases_share_description_join():
    overlay, state = start(torch.tensor([[.1, .1, .1]]))
    state = event(overlay, state, torch.tensor([[.8, .1, .1]]), index=1, column=1)
    state = push(overlay, state, torch.zeros(1, 3), word=303, index=2)
    old = overlay.read(state[0], torch.tensor([[1]]))[:, 0]
    state = overlay.apply(state, choice(old, identity=1), torch.stack((old*0, old), 1),
                          column=torch.tensor(2), words=8)
    state = event(overlay, state, torch.tensor([[.1, .8, .1]]), index=3, column=3)
    torch.testing.assert_close(state[0][0, 0], torch.tensor([.1, .8, .1]))
    description = torch.tensor([[.1, .1, .9]])
    state = push(overlay, state, description, word=404, index=4)
    points, records, _, _, _ = overlay.menu(state, description[:, None], torch.tensor([[0]]))
    snapshot = records[:, 0, 1].clone()
    state = overlay.apply(state, choice(points[:, 0, 1], identity=1),
        torch.stack((description, records[:, 0, 1]), 1), column=torch.tensor(4), words=8)
    torch.testing.assert_close(snapshot, torch.tensor([[.1, .8, .1]]))
    torch.testing.assert_close(overlay.refresh(state, torch.zeros(1, 2, 3), torch.tensor([[0, 1]])),
        torch.tensor([[[.1, .8, .9], [.1, .8, .9]]]))


@pytest.mark.parametrize('keep_second', (False, True))
def test_trial_values_are_isolated_and_only_selected_values_commit(keep_second):
    from SentenceCompose import select_rows
    overlay, original = start(torch.tensor([[.2, .3]]))
    first = event(overlay, original, torch.tensor([[.8, .1]]), index=1, column=1)
    second = event(overlay, original, torch.tensor([[.1, .9]]), index=1, column=1)
    kept = select_rows(first, second, torch.tensor([keep_second]))
    torch.testing.assert_close(original[0][0, 0], torch.tensor([.2, .3]))
    pool = MediumTermMemory()
    pool.open(address(1), kind=None, when=1, value=original[0][0, 0])
    pool.update(address(1), kept[0][0, 0], when=2)
    torch.testing.assert_close(pool.value_of(address(1)), second[0][0, 0] if keep_second else first[0][0, 0])
    assert not pool.value_of(address(1)).requires_grad


def test_names_events_checkpoint_and_reclamation_have_separate_ownership():
    pool = MediumTermMemory(2)
    x, alias, event_row = address(1), address(2), address(3)
    pool.open(x, kind=None, when=1, reading=101, value=torch.tensor([.5, .1]))
    pool.bind(x, alias, kind=None, when=2, reading=102, value=torch.tensor([.1, .7]))
    pool.event(x, event_row, verb=201, role=0, when=3)
    pool.event(x, event_row, verb=201, role=0, when=4)
    pool.update(x, torch.tensor([.2, .9]), when=4)
    assert pool.entries[x].count == 2 and pool.entries[x].occurrences == (x, alias)
    assert len(pool.entries[x].events) == 1
    assert pool.reading_of(x) == 101 and pool.reading_of(alias) == 102
    restored = MediumTermMemory(2)
    restored.restore(pool.state())
    assert restored.names == pool.names and restored.entries[x].events == pool.entries[x].events
    torch.testing.assert_close(restored.value_of(alias), torch.tensor([.2, .9]))
    restored.forget(event_row)
    assert restored.token_of(alias) == x and not restored.entries[x].events
    restored.forget(x)
    assert not restored.entries and not restored.names


def test_frame_access_does_not_leak_a_native_individuals_other_stream_value():
    from dataclasses import replace
    from IdentityRows import mint
    from ReferenceContext import SituationFrame
    from test_medium_term_memory import keep, mention
    from test_reference_close import fixture
    resolver, store, atoms = fixture()
    pool = resolver.components.owner.mtm
    native = mint(resolver.components.owner, atoms[0])
    rows = [keep(resolver, mention(atoms[0], reference=native), position=index)[0] for index in (0, 1)]
    first, second = map(store.occurrence_of, rows)
    assert first != second and pool.token_of(first) != pool.token_of(second)
    pool.update(first, atoms[1], when=20)
    frames = [replace(SituationFrame.from_row(store.row(row)), symbol_ids=(native,),
                      activations=((store.occurrence_of(row), native, 0),)) for row in rows]
    staged = MTMOverlay.stage(resolver, [(frames[0],), (frames[1],)],
        torch.zeros(2, 8, 12), journal=8, words=8)
    for batch, expected in enumerate((atoms[1], atoms[0])):
        current = staged[0][batch, staged[1][batch]]
        assert len(current) > 0
        torch.testing.assert_close(current, expected.expand_as(current), rtol=0, atol=0)
    assert not resolver.individuals(resolver.accessible(()))
