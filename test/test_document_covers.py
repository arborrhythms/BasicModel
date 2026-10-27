"""Cover segments, ordinary reference access and evidence centroids."""
from chunk_fixtures import context as chunk_context
from dataclasses import replace
from types import SimpleNamespace

import torch
import pytest

from DocumentCover import begin_sentence, end_document, latest, open_cover, read, segments
from Meaning import ConceptualMeaning
from QueryWork import QueryWorkBudget
from test_reference_close import fixture as reference_fixture


def fixture(*, percept=False):
    from MemoryIndex import configure_model_index
    resolver, store, atoms = reference_fixture(percept=percept)
    configure_model_index(SimpleNamespace(conceptualSpace=resolver.components.owner), store)
    return resolver, store, atoms


def append(store, value, stream=0, position=0, document='body'):
    return store.append_meaning(ConceptualMeaning.from_description(value), kind='observation',
                               evidence=(1., 0.), stream=stream, sentence_index=position,
                               document_key=document)


def test_cover_generator_is_global_rng_neutral_advances_and_restores():
    import copy
    from Layers import LTM0
    resolver, store, _ = fixture()
    owner = resolver.components.owner
    global_state = torch.get_rng_state().clone()
    first = open_cover(owner, store, stream=0, document='first')
    checkpoint = copy.deepcopy(store.state_dict())
    metadata = copy.deepcopy(store.semantic_extras())
    second = open_cover(owner, store, stream=0, document='second')
    first_value = store.raw_meaning_of(store.location(first)).roles[0]
    second_value = store.raw_meaning_of(store.location(second)).roles[0].clone()
    assert not torch.equal(first_value, second_value)
    restored = LTM0(store.nDim, capacity=store.capacity, percept_width=store.percept_width)
    restored.load_state_dict(checkpoint)
    restored.load_semantic_extras(metadata)
    resumed = open_cover(owner, restored, stream=0, document='second')
    torch.testing.assert_close(restored.raw_meaning_of(restored.location(resumed)).roles[0],
                               second_value, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), global_state, rtol=0, atol=0)
    assert store.slots.shape == restored.slots.shape == (store.capacity, store.nDim)


def test_cover_never_certifies_a_proposition():
    from Queries import ThoughtLTMCapability
    resolver, store, _ = fixture()
    cover = open_cover(resolver.components.owner, store, stream=0, document='body')
    query = store.raw_meaning_of(store.location(cover))
    reader = ThoughtLTMCapability(store=lambda: store, equal=lambda *_: 1., tau_id=.9)
    for reference in (cover, query.role_refs[0]):
        result = reader.premise_evidence(reference, max_records=20, work=QueryWorkBudget(20))
        assert result['support_true'] == result['support_false'] == 0.
    found = store.cued_rows(query, references=(cover,), max_candidates=20,
                           work=QueryWorkBudget(20))
    assert found['value'] == ()
    with pytest.raises(ValueError, match='cover cannot certify'):
        store.accept_fact(store.location(cover))


def test_full_store_boundary_pass_precedes_the_next_document_cover(monkeypatch):
    from Models import BasicModel
    from Forgetting import ForgettingConfig
    from test_forgetting import config
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    cover = open_cover(owner, store, stream=0, document='first')
    for position in range(store.capacity-1):
        append(store, atoms[0], position=position)
    assert len(store) == store.capacity
    model = SimpleNamespace(_concept_owner=lambda: owner,
        symbolSpace=SimpleNamespace(ltm_store=store, what_memory=None, expectation=None,
                                   truth_layer=None),
        _document_covers={0: ('first', store.capacity, cover)},
        _sentence_sources=((('second', 0),),),
        _forget_document_boundary=True, _forget_training=False,
        _forget_all_streams_closed=True, _open_sentence_slot=None,
        stateless=False, conceptualSpaces=[], _detach_persistent_state=lambda: None)
    monkeypatch.setattr(ForgettingConfig, 'from_xml', classmethod(lambda cls: config()))
    # The streaming cursor performs this after the previous document's
    # backward/reset and before it calls the next runBatch/begin_sentence.
    BasicModel.post_tick_compact(model)
    report = store.last_forgetting_report
    assert report['before'] == store.capacity and report['after'] < store.capacity
    assert int(store.forget_passes) == 1
    begin_sentence(model, 0, torch.tensor([True]))
    assert len(store) == report['after']+1
    assert latest(store, 0) != cover and int(store.forget_passes) == 1


def test_cursor_reuses_the_book_cover_on_a_reread():
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    # The same recent-address owner production uses; no word or target.
    recent = SimpleNamespace(include_occurrence=lambda b, address: seen.append((b, address)))
    seen = []
    model = SimpleNamespace(_concept_owner=lambda: owner,
        symbolSpace=SimpleNamespace(expectation=recent),
        _sentence_sources=((('first', 1), ('first', 2), ('second', 1)),))
    for slot in range(3):
        begin_sentence(model, slot, torch.tensor([True]))
        append(store, atoms[slot], position=slot)
    assert [store.row(i)['kind'] for i in range(len(store))] == [
        'cover', 'observation', 'observation', 'cover', 'observation']
    first, second = (address for _, address in seen)
    assert first != second and latest(store, 0) == second
    symbols = [store.raw_meaning_of(store.location(address)).role_refs[0][1]
               for address in (first, second)]
    assert len(set(symbols)) == 2
    assert all(symbol in owner._concept_allocator.individuals for symbol in symbols)
    assert all(owner._concept_allocator.order_of(symbol) == 1 for symbol in symbols)
    assert not owner.mtm.entries
    before = store.slots.clone(), store.address_keys.clone()
    begin_sentence(model, 2, torch.tensor([True]))
    assert latest(store, 0) == second
    torch.testing.assert_close(store.slots[:5], before[0][:5], rtol=0, atol=0)
    torch.testing.assert_close(store.address_keys[:5], before[1][:5], rtol=0, atol=0)


def test_equal_time_boundary_and_two_streams_keep_exact_ownership():
    resolver, store, atoms = fixture(percept=True)
    owner = resolver.components.owner
    store._model_present = lambda: 7
    a = open_cover(owner, store, stream=0, document='first')
    a_row = append(store, atoms[0])
    b = open_cover(owner, store, stream=1, document='second')
    b_row = append(store, atoms[1], stream=1)
    c = open_cover(owner, store, stream=0, document='third')
    c_row = append(store, atoms[2], position=1)
    membership = segments(store)
    assert membership[a_row] == {a} and membership[b_row] == {b} and membership[c_row] == {c}
    assert store.stream_rows(0).tolist() == [0, a_row, 4, c_row]
    assert store.stream_rows(1).tolist() == [2, b_row]
    from Forgetting import history_age
    torch.testing.assert_close(history_age(store), torch.zeros(6))
    from Layers import SymbolExpectation
    chain = SymbolExpectation.get_stm_chain(SimpleNamespace(_ltm_store=store), b=1)
    torch.testing.assert_close(torch.stack([entry[1][0] for entry in chain]), store.slots[[2, b_row]])


def test_end_document_retires_only_tokens_introduced_in_its_segment():
    resolver, store, atoms = fixture()
    owner, pool = resolver.components.owner, resolver.components.owner.mtm
    covers, tokens = [], []
    for n in range(2):
        covers.append(open_cover(owner, store, stream=0, document=('body', n)))
        row = append(store, atoms[n], position=n)
        token = store.occurrence_of(row)
        tokens.append(token)
        pool.open(token, kind=None, value=atoms[n], when=n)
        pool.close(store, stream=0, references=(token,))
    before = store.slots.clone(), owner.similarity_codebook.W.clone()
    pool.end_document(covers[0], store=store)
    assert tokens[0] not in pool.entries and tokens[1] in pool.entries
    model = SimpleNamespace(_concept_owner=lambda: owner)
    end_document(model, 0)
    assert not pool.entries
    torch.testing.assert_close(store.slots, before[0], rtol=0, atol=0)
    torch.testing.assert_close(owner.similarity_codebook.W, before[1], rtol=0, atol=0)


def test_cover_concept_reads_only_its_segment_under_the_shared_meter():
    resolver, store, atoms = fixture()
    cover = open_cover(resolver.components.owner, store, stream=0, document='first')
    first = append(store, atoms[0])
    append(store, atoms[1], position=1)
    open_cover(resolver.components.owner, store, stream=0, document='second')
    append(store, atoms[2], position=2)
    query = store.raw_meaning_of(store.location(cover))
    meter = QueryWorkBudget(1)
    result = read(store, query, max_records=20, work=meter)
    assert [row['index'] for row in result['value']] == [first]
    assert result['records_scanned'] == meter.spent == 1
    assert result['incomplete'] == ('cover_read_limit',)
    result = read(store, query, max_records=20, work=QueryWorkBudget(20))
    assert [row['index'] for row in result['value']] == [first, first+1]


def test_cover_is_an_ordinary_recent_candidate_with_no_lexical_name():
    from ReferenceContext import SituationAddress
    from test_medium_term_memory import mention
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    cover = open_cover(owner, store, stream=0, document='body')
    # A returned cover enters the working identity pool through recall.
    store.recall_identities((dict(store.row(store.location(cover)), index=store.location(cover)),),
                            work=QueryWorkBudget(20))
    frame = SituationAddress(cover).read(store)
    symbol = store.raw_meaning_of(store.location(cover)).role_refs[0][1]
    records = resolver.individuals(resolver.accessible(chunk_context(resolver, (frame,))))
    assert symbol in records
    assert owner.definitions.words(symbol) == ()
    clause = mention(frame.point, mode='bind')
    plan = resolver.preview(clause, chunk_context(resolver, (frame,)), stream=0)
    assert symbol in plan['items'][0]['candidates']
    plan['items'][0]['selected'] = symbol
    prepared = resolver.prepare(clause, plan)
    assert prepared is not None
    written = {}
    row = store.write_clause(prepared, document_key='mention', stream=0, written_rows=written)
    resolver.commit(plan, row, written, document='mention', position=0, training=False)
    assert resolver.last_audit[0]['reference'] == symbol
    occurrence = store.occurrence_of(store.index_of_row(resolver.last_audit[0]['occurrence']))
    assert resolver._derivation_symbol(occurrence) == symbol
    assert owner.mtm.token_of(occurrence) == owner.mtm.token_of(cover)


def test_cover_lookup_uses_the_index_and_charges_before_body_reads(monkeypatch):
    resolver, store, atoms = fixture()
    cover = open_cover(resolver.components.owner, store, stream=0, document='body')
    append(store, atoms[0])
    query = store.raw_meaning_of(store.location(cover))
    def forbidden(*args, **kwargs):
        raise AssertionError('an exhausted query must not read a row payload')
    monkeypatch.setattr(store, 'raw_meaning_of', forbidden)
    monkeypatch.setattr(store, 'row', forbidden)
    result = read(store, query, max_records=20, work=QueryWorkBudget(0))
    assert result['value'] == () and result['records_scanned'] == 0
    assert result['incomplete'] == ('cover_read_limit',)


def test_cover_region_does_not_certify_unrelated_content():
    resolver, store, atoms = fixture()
    cover = open_cover(resolver.components.owner, store, stream=0, document='body')
    append(store, atoms[0])
    query = ConceptualMeaning.from_description(atoms[1])
    result = read(store, query, references=(cover,), max_records=20, work=QueryWorkBudget(20))
    assert len(result['value']) == 1 and result['value'][0]['match'] == 0.


def test_forgetting_protects_a_cover_for_the_pass_that_drops_its_last_member():
    from Forgetting import forget
    from test_forgetting import config
    resolver, store, atoms = fixture()
    cover = open_cover(resolver.components.owner, store, stream=0, document='body')
    body = store.occurrence_of(append(store, atoms[0]))
    cfg = config(high_water=2/store.capacity, low_water=1/store.capacity, min_interval=0)
    forget(store, cfg, document_boundary=True)
    assert store.location(cover) is not None and store.location(body) is None
    from DocumentCover import rows
    assert rows(store, cover) == (store.location(cover),)


def test_higher_order_fold_is_evidence_weighted_and_preserves_rows():
    from MemoryConsolidation import _fold
    resolver, store, atoms = fixture()
    rows = [store.append_meaning(ConceptualMeaning.from_description(atoms[n]),
        evidence=pair, kind='observation', stream=0, sentence_index=n)
        for n, pair in enumerate(((1., 0.), (.25, .25), (0., 0.)))]
    before = store.slots.clone()
    folded = _fold(resolver.components.owner, store, rows)
    torch.testing.assert_close(folded, (atoms[0] + .5*atoms[1]) / 1.5)
    torch.testing.assert_close(store.slots, before, rtol=0, atol=0)


def test_centroid_centers_parts_and_native_wholes_by_their_separate_counts():
    from MeaningKernels import centroid, split_centroid
    from FormCentroids import SpanWholes
    parts = torch.tensor([[.2, .6], [.8, .4]])
    codes = torch.tensor([[.9, .1], [.7, .3], [.0, 1.]])
    coefficients = torch.zeros(3, 256)
    coefficients[:2, [10, 20]] = 1.
    coefficients[2, 10] = 1.
    counts = torch.zeros(1, 2, 256)
    counts[0, 0, 10] = counts[0, 1, 20] = 1.
    bank = SpanWholes(torch.cat((counts[:, :1]*0, counts.cumsum(1)), 1), coefficients, codes)
    whole = bank.read(torch.tensor([[[0, 2]]]))
    torch.testing.assert_close(whole[0][0, 0], codes[:2].mean(0))
    assert whole[1].item() == 2
    left, right = parts[0].reshape(1, 1, -1), parts[1].reshape(1, 1, -1)
    value = centroid((left, right), (3, 1), whole=whole)
    expected = ((parts[0]*3+parts[1])/4 + codes[:2].mean(0))/2
    torch.testing.assert_close(value[0, 0], expected)
    torch.testing.assert_close(split_centroid(value, right, (3, 1), whole=whole), left)
    absent = bank.read(torch.tensor([[[-1, -1]]]))
    torch.testing.assert_close(centroid((left, right), (3, 1), whole=absent),
                               centroid((left, right), (3, 1)))


def test_whole_contribution_follows_the_composed_stamp():
    from types import SimpleNamespace
    from FormCentroids import selected
    mean, count = torch.ones(1, 2, 4), torch.ones(1, 2, 1)
    orders = (torch.tensor([[0, 1]]), torch.tensor([[0, 1]]))
    rule = SimpleNamespace(head_role=2)
    actual = selected({2: (mean, count, orders)}, 2, mean, rule)
    assert actual[1].flatten().tolist() == [1., 0.]
    rule.determiner_mode = 'kind'
    assert selected({2: (mean, count, orders)}, 2, mean, rule)[1].flatten().tolist() == [1., 0.]


@pytest.mark.parametrize('centered', (False, True))
def test_free_inverse_reads_native_wholes_from_candidates_without_a_span(monkeypatch, centered):
    from test_meaning_kernels import paired
    # Use the common paired-layout setup directly; no parse or target bytes
    # are supplied to the inverse search.
    paired.__wrapped__(monkeypatch)
    from Language import LanguageSpace, ProductLayer
    from FormCentroids import CandidateWholes
    op = ProductLayer(12, 12).double()
    a, b = torch.linspace(.1, .7, 12, dtype=torch.float64), torch.linspace(.8, .2, 12, dtype=torch.float64)
    support = torch.zeros(1, 2, 256, dtype=torch.bool)
    support[0, 0, 10] = support[0, 1, 20] = True
    properties = torch.zeros(1, 256, dtype=torch.float64)
    properties[:, [10, 20]] = 1.
    wholes = CandidateWholes(torch.stack((a[:4], b[:4]))[None], support,
        torch.ones(1, 2, dtype=torch.bool), properties, torch.tensor([[.9, .7, .6, .3]], dtype=torch.float64))
    native = wholes.read(a[None], b[None])
    value = op.compose_blocks(op.compose, a[None], b[None], counts=(3, 1),
                              whole=native if centered else None)
    left, right, valid = LanguageSpace._bounded_binary_reconstruction(op, value,
        torch.zeros_like(value), torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool),
        torch.stack((a, b))[None], torch.ones(1, 2, dtype=torch.bool), 2, whole_bank=wholes)
    assert valid.all()
    torch.testing.assert_close(left, a[None], rtol=0, atol=1e-12)
    torch.testing.assert_close(right, b[None], rtol=0, atol=1e-12)


@pytest.mark.parametrize('centered', (False, True))
def test_free_unary_inverse_uses_the_same_native_form_equations(monkeypatch, centered):
    from test_meaning_kernels import paired
    paired.__wrapped__(monkeypatch)
    from Language import LanguageSpace, SurfaceLayer, SurfaceUnaryLayer
    from FormCentroids import CandidateWholes
    op = SurfaceUnaryLayer(SurfaceLayer(12, 12)).double()
    value = torch.linspace(.15, .65, 12, dtype=torch.float64)[None]
    support = torch.zeros(1, 1, 256, dtype=torch.bool)
    support[..., 10] = True
    properties = torch.zeros(1, 256, dtype=torch.float64)
    properties[:, 10] = 1.
    bank = CandidateWholes(value[:, None, :4], support, torch.ones(1, 1, dtype=torch.bool),
        properties, torch.tensor([[.9, .7, .6, .3]], dtype=torch.float64))
    parent = op.compose_blocks(op.compose, value, whole=bank.read(value) if centered else None)
    language = SimpleNamespace(_generate_unary_ops=(op,))
    actual = LanguageSpace.generate_unary_step(language, parent, torch.tensor([0]),
        torch.tensor([True]), whole_bank=bank)
    torch.testing.assert_close(actual, value, rtol=0, atol=1e-12)


def test_free_unary_center_requires_a_valid_recovered_form():
    from FormCentroids import CandidateWholes, unary
    # A closer centered dictionary word is insufficient when uncentering
    # this different parent would leave the membership chart.
    forms = torch.tensor([[[.3, .9]]])
    support = torch.zeros(1, 1, 256, dtype=torch.bool)
    support[..., 10] = True
    properties = torch.zeros(1, 256)
    properties[:, 10] = 1.
    bank = CandidateWholes(forms, support, torch.ones(1, 1, dtype=torch.bool),
                           properties, torch.tensor([[.8, .1]]))
    parent = torch.tensor([[.35, .5, .2, .7]])
    assert (((forms + bank.codes)*.5-parent[:, None, :2]).square().sum()
            < (forms-parent[:, None, :2]).square().sum())
    torch.testing.assert_close(unary(parent, bank), parent, rtol=0, atol=0)
