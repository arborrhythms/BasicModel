"""MTM retains chosen bindings, while interpretation and trials stay read-only."""
from dataclasses import replace
import pytest
import torch

from ClauseRow import Clause, Composition
from Meaning import ConceptualMeaning
from MediumTermMemory import MediumTermMemory
from ReferenceContext import SituationFrame
from test_reference_close import close, fixture
from part_fixtures import part_references, assert_part_references


def snapshot(resolver):
    store = resolver.store
    return (part_references(store), store.slots.clone(), store.when.clone(),
            {k: v.clone() for k, v in store.parts.state_dict().items()},
            resolver.components.owner.similarity_codebook.W.detach().clone())


def unchanged(resolver, before):
    after = snapshot(resolver)
    for actual, expected in zip(after, before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def mention(point, *, mode='mint', reference=-1):
    # A supplied, identity-preserving lower retains the admitted kind.
    term = Composition(point, (Composition(point*0), Composition(point, reference=reference)),
        operation='lower', formal_name='lower', head_role=2, order=1)
    if mode != 'mint':
        term = Composition(point, order=1)
    return Clause(ConceptualMeaning.from_description(point), point=point, order=1,
        composition=term, reference_request=dict(mode=mode, leaves=(1,),
            cue=point*0, content=point, predicate=point*0, role=0))


def keep(resolver, clause, frames=(), *, position=0):
    plan = resolver.preview(clause, frames)
    before = dict(resolver.components.owner.mtm.entries)
    kept = resolver.prepare(clause, plan)
    assert resolver.components.owner.mtm.entries == before
    written = {}
    row = resolver.store.write_clause(kept, document_key='mtm',
        sentence_index=position, written_rows=written)
    assert resolver.components.owner.mtm.entries == before
    retained = snapshot(resolver)
    resolver.commit(plan, row, written, document='mtm', position=position, training=False)
    unchanged(resolver, retained)
    return row, plan


@pytest.mark.parametrize('mode', ('bind', 'pronoun', 'elision'))
def test_bound_use_reads_as_its_introduction_without_rewriting_rows(mode):
    resolver, store, a = fixture()
    first, address, _ = close(resolver, a[0], a[6], 'mint', (), document='pool', sentence=0)
    pool = resolver.components.owner.mtm
    introduction = store.occurrence_of(store.index_of_row(address))
    original = pool.entries[introduction]
    before = part_references(store), store.slots.clone()
    use, plan = keep(resolver, mention(a[0], mode=mode), (first,), position=1)
    occurrence = store.occurrence_of(use)
    assert plan['items'][0]['reference'] == introduction
    assert pool.token_of(occurrence) == introduction
    entry = pool.entries[introduction]
    assert entry.occurrences == (introduction, occurrence) and entry.count == 2
    assert entry.kind == store.raw_meaning_of(store.index_of_row(address)).role_refs[0][1]
    assert entry.when > original.when
    assert_part_references(store, before[0])
    torch.testing.assert_close(store.slots[:use], before[1][:use], rtol=0, atol=0)
    frame = SituationFrame.from_row(store.row(use))
    # Both the direct row and activation read use the retained token.
    for held in (frame, replace(first, activations=((occurrence, None, 0),))):
        context = next(context for location, context in resolver.accessible((held,)).values()
                       if location == use)
        assert context['identity'] == introduction
    before_read = dict(pool.entries)
    groups = resolver.individuals(resolver.accessible((frame,)))
    assert set(groups) == {introduction}
    assert pool.entries == before_read
    assert not resolver.accessible(()) and not resolver.individuals({})


def test_lru_eviction_keeps_rows_and_a_later_mention_opens_a_new_token():
    resolver, store, a = fixture()
    pool = resolver.components.owner.mtm = MediumTermMemory(2)
    frames, tokens = [], []
    for n in range(2):
        frame, _, plan = close(resolver, a[n], a[6], 'mint', (), document='lru', sentence=n)
        frames.append(frame); tokens.append(plan['items'][0]['reference'])
    keep(resolver, mention(a[0], mode='bind'), (frames[0],), position=2)
    last_access = pool.entries[tokens[0]].when
    old = snapshot(resolver)
    retained_rows = len(store)
    # Admission may write new rows; eviction at commit must not touch them.
    third, _, plan = close(resolver, a[2], a[6], 'mint', (), document='lru', sentence=3)
    assert_part_references(store, old[0])
    for current, before in zip((store.slots, store.when), old[1:3]):
        torch.testing.assert_close(current[:retained_rows], before[:retained_rows], rtol=0, atol=0)
    assert tokens[0] in pool.entries and tokens[1] not in pool.entries
    assert pool.token_of(tokens[1]) is None
    assert pool.entries[plan['items'][0]['reference']].when > last_access
    use, plan = keep(resolver, mention(a[1], mode='bind'), (frames[1],), position=4)
    assert plan['items'][0]['reference'] == tokens[1]
    fresh = store.occurrence_of(use)
    assert pool.token_of(fresh) == fresh and fresh != tokens[1]
    assert pool.entries[fresh].occurrences == (fresh,) and pool.entries[fresh].count == 1
    assert pool.token_of(tokens[1]) is None


@pytest.mark.parametrize('reclose', ('bind', 'mint'))
def test_reclosed_occurrence_survives_eviction_of_its_previous_token(reclose):
    resolver, store, a = fixture()
    owner = resolver.components.owner
    pool = owner.mtm = MediumTermMemory(2)
    kind = owner.new_concept()
    seat = owner._csw_concept_row(0, kind)
    with torch.no_grad():
        owner.similarity_codebook.W[seat].copy_(a[0])
    frames, tokens = [], []
    for position in range(2):
        row, _ = keep(resolver, mention(a[0], reference=kind), position=position)
        frames.append(SituationFrame.from_row(store.row(row)))
        tokens.append(store.occurrence_of(row))
    first, second = tokens
    row, _ = keep(resolver, mention(a[0], mode='bind'), (frames[0],), position=2)
    occurrence = store.occurrence_of(row)
    assert pool.token_of(occurrence) == first
    retained = snapshot(resolver)
    again, _ = keep(resolver, mention(a[0], mode=reclose, reference=kind),
                    (frames[1],), position=2)
    assert again == row
    target = second if reclose == 'bind' else occurrence
    assert pool.token_of(occurrence) == target
    assert occurrence not in pool.entries[first].occurrences
    # Re-closing the same retained description does not edit its content,
    # derivation or the bank; membership changes only in MTM.
    unchanged(resolver, retained)
    keep(resolver, mention(a[0], reference=kind), position=3)
    assert first not in pool.entries
    assert pool.token_of(occurrence) == target
    assert occurrence in pool.entries[target].occurrences
    assert all(pool.token_of(use) == token for token, entry in pool.entries.items()
               for use in entry.occurrences)


def test_native_individual_mint_and_bind_share_an_occurrence_keyed_value():
    from IdentityRows import mint
    resolver, store, a = fixture()
    native = mint(resolver.components.owner, a[0])
    row, plan = keep(resolver, mention(a[0], reference=native))
    assert plan['items'][0]['reference'] == native
    frame = replace(SituationFrame.from_row(store.row(row)), symbol_ids=(native,),
                    activations=((store.occurrence_of(row), native, 0),))
    _, plan = keep(resolver, mention(a[0], mode='bind'), (frame,), position=1)
    assert plan['items'][0]['reference'] == native
    pool = resolver.components.owner.mtm
    token = pool.token_of(native)
    assert token == store.occurrence_of(row)
    assert pool.entries[token].native == native and pool.entries[token].count == 2
    torch.testing.assert_close(pool.value_of(native), a[0], rtol=0, atol=0)


def test_same_close_new_alias_appends_once_to_the_introduction():
    resolver, store, a = fixture()
    first, second = mention(a[0]), mention(a[0], mode='bind')
    clause = replace(first, companions=(second,))
    _, plan = keep(resolver, clause)
    assert plan['items'][1]['selected'] == ('new', 0)
    token = plan['items'][0]['reference']
    assert plan['items'][1]['reference'] == token
    entry = resolver.components.owner.mtm.entries[token]
    assert len(entry.occurrences) == entry.count == 2
    assert entry.occurrences[0] == token


def test_revision_merges_are_computed_without_merging_pool_entries():
    resolver, store, a = fixture()
    frames = []
    for index, (content, mode) in enumerate(((a[0], 'mint'), (a[0], 'mint'),
            (a[0]+a[4], 'bind'), (a[0]+a[4], 'bind'))):
        frame, _, _ = close(resolver, content, a[6], mode, frames,
            document='revision', sentence=index, selected=1 if index == 3 else None)
        frames.append(frame)
    pool = resolver.components.owner.mtm
    before = dict(pool.entries)
    retained = snapshot(resolver)
    assert len(resolver.individuals(resolver.accessible(frames))) == 1
    assert pool.entries == before and len(pool.entries) == 2
    assert all(entry.count == len(entry.occurrences) == 2 for entry in pool.entries.values())
    unchanged(resolver, retained)


@pytest.mark.parametrize('compiled', (False, True))
def test_trials_leave_pool_untouched_and_only_the_kept_close_writes(compiled):
    from Language import LowerLayer
    resolver, store, a = fixture()
    pool = resolver.components.owner.mtm
    operator = LowerLayer(12, 12)
    forward = torch.compile(operator.compose, backend='eager', fullgraph=True) if compiled else operator.compose
    plans = []
    for _ in range(2):
        point = forward(a[1][None], a[0][None])[0]
        clause = mention(point)
        plan = resolver.preview(clause, ())
        kept = resolver.prepare(clause, plan)
        plans.append((kept, plan))
        assert not pool.entries
    kept, plan = plans[0]
    written = {}
    row = store.write_clause(kept, written_rows=written)
    assert not pool.entries
    retained = snapshot(resolver)
    resolver.commit(plan, row, written, document='trial', position=0, training=False)
    unchanged(resolver, retained)
    token = store.occurrence_of(row)
    assert pool.entries[token].occurrences == (token,) and pool.entries[token].count == 1
    assert not any('mtm' in key for key in resolver.components.owner.state_dict())


def test_row_reclamation_drops_a_use_and_then_its_introductions_entry():
    resolver, store, a = fixture()
    first, address, _ = close(resolver, a[0], a[6], 'mint', (), document='forget', sentence=0)
    use, _ = keep(resolver, mention(a[0], mode='bind'), (first,), position=1)
    pool = resolver.components.owner.mtm
    token = store.occurrence_of(store.index_of_row(address))
    used = store.occurrence_of(use)
    store.compact_rows([row for row in range(len(store)) if row != use])
    assert pool.token_of(used) is None and pool.entries[token].occurrences == (token,)
    introduction = store.location(token)
    store.compact_rows([row for row in range(len(store)) if row != introduction])
    assert not pool.entries and pool.token_of(token) is None


@pytest.mark.parametrize('compiled', (False, True))
def test_complete_forward_body_does_not_access_the_pool(tmp_path, monkeypatch, compiled):
    from test_compiled_word_chunk import _tiny_canonical_model, _stage_fullgraph_tensor_peer
    model = _tiny_canonical_model(tmp_path, monkeypatch)
    model._compiled_word_loop_fullgraph = True
    raw = _stage_fullgraph_tensor_peer(model, ['alpha beta gamma delta', 'epsilon zeta'])
    pool = model.conceptualSpace.mtm
    assert pool.capacity == 64 and not pool.entries
    def forbidden(*args, **kwargs):
        raise AssertionError('the tensor forward body accessed MTM')
    for name in ('token_of', 'open', 'bind', 'forget'):
        monkeypatch.setattr(pool, name, forbidden)
    forward = lambda unused: model._forward_with_compiled_sentence_state(None)
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    if compiled:
        forward = torch.compile(forward, backend='eager', fullgraph=True)
    try:
        result = forward(raw)
        model._publish_compiled_sentence_state(result)
        assert not pool.entries
        if compiled:
            assert torch._dynamo.utils.counters['stats']['unique_graphs'] > 0
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()
