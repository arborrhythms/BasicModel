"""The revised placement, permanent concepts and episodic System-1 contracts."""
import warnings

import pytest
import torch


def test_unplaced_concepts_start_at_their_own_seed_and_keep_their_position():
    from test_cs_sparse_weights import _cs
    from FamiliarTier import cover_members
    owner = _cs(nS=64, order=3, dim=8)
    members = [owner.new_concept(order=0) for _ in range(2)]
    concept = owner.new_concept(order=1)
    row = owner._csw_row_of(concept)
    before = owner.similarity_codebook.W[row].detach().clone()
    torch.testing.assert_close(before.norm(), torch.tensor(1.))
    generator = torch.Generator().manual_seed(concept)
    expected = torch.nn.functional.normalize(torch.randn(8, generator=generator).abs(), dim=0)
    torch.testing.assert_close(before, expected)
    cover_members(owner, concept, {owner._csw_row_of(cid): 1. for cid in members})
    torch.testing.assert_close(owner.similarity_codebook.W[row], before)


def test_percept_start_and_candidate_centroid_ignore_higher_orders():
    from test_cs_sparse_weights import _cs
    from FamiliarTier import cover_members, centroid
    from FormCentroids import convex_mean
    owner = _cs(nS=64, order=3, dim=8)
    concepts = [owner.new_concept(order=order) for order in (0, 0, 1, 0)]
    a, b, higher, target = [owner._csw_row_of(cid) for cid in concepts]
    with torch.no_grad():
        owner.similarity_codebook.W[a] = torch.eye(8)[0]
        owner.similarity_codebook.W[b] = torch.eye(8)[1]
        owner.similarity_codebook.W[higher] = torch.ones(8)*100
    support = {a:1., b:3., higher:9.}
    expected = convex_mean(torch.tensor([1.,3.]), torch.eye(8)[:2])[0]
    cover_members(owner, concepts[-1], support)
    torch.testing.assert_close(owner.similarity_codebook.W[target], expected)
    torch.testing.assert_close(centroid(owner, support, order=0), expected)
    with torch.no_grad():
        owner.similarity_codebook.W[higher].mul_(-1)
    torch.testing.assert_close(centroid(owner, support, order=0), expected)


def test_full_identity_promotion_warns_once_and_leaves_the_entry():
    from FamiliarTier import promote_identity
    from test_presentation_memory import fixture, write
    resolver, store, points = fixture()
    owner = resolver.components.owner
    occurrence = write(store, points[0], ('full', 0))
    token = owner.mtm.open(occurrence, kind=None, value=points[0], when=0)
    alloc = owner._concept_allocator
    for index in range(owner.nVectors):
        alloc.layer().assign_row(('occupied', index), capacity=owner.nVectors, base=0)
    before = (alloc.next_id, dict(alloc.layer()._tensor_rows), len(store.parts))
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        assert promote_identity(owner, token) is None
        assert promote_identity(owner, token) is None
    assert len(recorded) == 1 and recorded[0].category is RuntimeWarning
    assert 'order 1' in str(recorded[0].message)
    assert owner._promotion_refusals == {1:2}
    assert (alloc.next_id, alloc.layer()._tensor_rows, len(store.parts)) == before
    assert owner.mtm.entries[token].symbol is None
    from Spaces import ConceptRowCapacityError
    with pytest.raises(ConceptRowCapacityError):
        owner.new_concept(order=1)


def test_identity_starts_from_its_neighbours_objects_without_moving_them():
    from argument_fixtures import argument_inventory
    from ConceptPlacement import identity_start, positions
    from placement_fixtures import place_at
    owner, speech, words = argument_inventory(('modifier_argin1',))
    owner.object_start_neighbours = 1
    objects = [owner.new_concept(order=1) for _ in range(3)]
    for word, obj, point in zip(words[:2], objects[:2], ([0.,0.,1.,0.],[0.,0.,0.,1.])):
        owner.interpret.define(word, obj)
        place_at(owner, obj, point)
    place_at(owner, words[2], [1.,.1,0.,0.])
    before = positions(owner).clone()
    identity_start(owner, objects[2], (words[2],))
    row = owner._csw_row_of(objects[2])
    expected = before.clone()
    expected[row] = torch.tensor([0.,0.,1.,0.])
    torch.testing.assert_close(positions(owner), expected)


def test_sentence_places_words_in_addition_to_their_argument_pull():
    from types import SimpleNamespace as NS
    from argument_fixtures import argument_inventory
    from test_parts_of_speech import rule
    from PartsOfSpeech import formal_arguments
    owner, speech, words = argument_inventory(formal_arguments((rule('modifier'),)))
    bank = owner.similarity_codebook.mereology
    obj = owner.new_concept(order=0)
    owner._concept_allocator.symbol_types.setdefault(obj, set()).add('object')
    rows = [owner._csw_row_of(cid) for cid in (*words[:2], obj, *speech.concepts.values())]
    context = bank.placement.new_zeros(owner.nVectors)
    context[rows] = 1.
    owner.placement_context = lambda batch: context
    owner._model = NS(languageSpace=NS(parts_of_speech=speech,
        _compose_binary_rules=(rule('modifier'),), _compose_unary_rules=()))
    program = NS(word_ids=torch.tensor(words[:2]), word_rows=torch.tensor(rows[:2]),
        actions=torch.tensor([[0,-1,0],[0,-1,1],[1,0,0]]))
    before = bank.placement.clone()
    bank.observe_close(program)
    assert not torch.equal(bank.placement[rows[:2]], before[rows[:2]])
    torch.testing.assert_close(bank.placement[rows[2:]], before[rows[2:]])


def test_definition_meta_concepts_pull_only_present_members_of_one_order():
    from argument_fixtures import argument_inventory
    from ConceptPlacement import meta_groups
    from placement_fixtures import place_at
    owner, speech, words = argument_inventory(('modifier_argin1',))
    same, higher, other = [owner.new_concept(order=k) for k in (0,1,0)]
    for cid, point in ((same,[0.,1.,0.,0.]),(higher,[0.,0.,1.,0.]),(other,[0.,0.,0.,1.])):
        place_at(owner,cid,point)
    owner.interpret.define(words[0],same)
    owner.interpret.define(words[0],higher)
    owner.interpret.define(words[0],other)
    bank = owner.similarity_codebook.mereology
    context = bank.placement.new_zeros(owner.nVectors)
    context[[owner._csw_row_of(cid) for cid in (words[0],same,higher)]] = 1.
    before = bank.placement.clone()
    groups = meta_groups(owner,context)
    assert all(len(members)==2 for _,members,_ in groups)
    bank.place_groups(groups,b'definition-test')
    assert not torch.equal(bank.placement[owner._csw_row_of(same)],before[owner._csw_row_of(same)])
    for cid in (higher,other):
        torch.testing.assert_close(bank.placement[owner._csw_row_of(cid)],before[owner._csw_row_of(cid)])


def test_intuit_deposits_only_when_selected_and_derives_higher_orders(monkeypatch):
    from test_thought_infer import setup
    from Understanding import ConceptualField
    from SentenceField import publish, knowing
    from IntuitionField import open_field, read
    from QueryWork import QueryWorkBudget
    from Layers import Error
    model, registry, store, refs = setup()
    owner = model._concept_owner()
    object.__setattr__(model, 'conceptualSpaces', (owner,))
    a, b = [owner._csw_row_of(ref[1]) for ref in refs[:2]]
    parent = owner.new_concept(order=1)
    row = owner._csw_row_of(parent)
    for member in (a, b): owner.add_concept_edge(row, member, 1.)
    ids = torch.tensor([*map(lambda ref: ref[1], refs), parent])
    initial = ConceptualField(ids, torch.zeros(4, 1, 1, 2),
        torch.tensor([[[0, 1]]]), torch.tensor([[[0, 1]]]), None)
    def forecast(values):
        # The stack reads each concept's two lanes and gives one signed number.
        assert values.shape == (1, 2, owner._order_caps()[0])
        result = torch.zeros_like(values[:, 0])
        result[0,a], result[0,b] = .8, -.2
        return result
    monkeypatch.setattr(owner.expect_stack, 'forward', forecast)
    def no_key(*args, **kwargs): raise AssertionError('System 1 made a point')
    monkeypatch.setattr('FieldProjection.key', no_key)
    opened = open_field(model, initial)
    torch.testing.assert_close(opened.evidence, initial.evidence)
    publish(model, opened)
    from ConceptSpread import open_prediction
    open_prediction(model, torch.tensor([True]))
    from WalkTrials import ThoughtTrialState
    trial = ThoughtTrialState(model)
    assert read(owner, row=0, work=QueryWorkBudget(64)) is None
    field = knowing(model)
    address = (field.concept_ids == parent).nonzero().item()
    # The parent takes the greater for (.8 of a) and the lesser against
    # (0 of a, below .2 of b).
    torch.testing.assert_close(field.evidence[address,0,0], torch.tensor([.8,0.]))
    trial.restore(effects_only=True)
    torch.testing.assert_close(knowing(model).evidence, field.evidence)
    trial.restore()
    torch.testing.assert_close(knowing(model).evidence, opened.evidence)
    assert not hasattr(model, '_intuition_teacher')
    from test_thought_infer import idea
    content = idea(registry, refs[0])
    requests = (registry.form('intuit', content), registry.form('query', content))
    assert registry.field_candidates(requests) == requests


def test_a_seed_is_a_native_nonnegative_evidence_code():
    from test_familiar_tier import placed_fixture
    from ConceptPlacement import seed
    owner, bank, _, _, _ = placed_fixture()
    cid = owner.new_concept(order=1)
    row = owner._csw_row_of(cid)
    before = bank.mereology.placement[row].clone()
    assert not bool((before < 0).any())
    bank.trust.read_object(cid)
    code = bank.lookup_rows(row)
    assert bool(((code >= 0) & (code <= 1)).all())
    torch.testing.assert_close(bank.mereology.placement[row], seed(cid, before))


def test_explicit_identity_fold_keeps_its_own_seed_and_the_extension_separate():
    from test_reference_close import fixture
    from IdentityRows import mint
    from ConceptPlacement import seed
    resolver, _, atoms = fixture()
    owner = resolver.components.owner
    identity = mint(owner, atoms[0])
    row = owner._csw_row_of(identity)
    torch.testing.assert_close(owner.similarity_codebook.trust.content[row], atoms[0])
    torch.testing.assert_close(torch.nn.functional.normalize(owner.similarity_codebook.W[row],dim=0),
                               seed(identity, atoms[0]))


def test_declared_arguments_and_operators_keep_their_id_seed_without_invented_views():
    from argument_fixtures import argument_inventory
    from PartsOfSpeech import formal_arguments
    from test_parts_of_speech import rule
    from ConceptPlacement import positions, seed
    owner, speech, _ = argument_inventory(formal_arguments((rule('modifier'),)))
    identities = (*speech.concepts.values(), speech.seats['modifier'].concept)
    for cid in identities:
        row = owner._csw_row_of(cid)
        torch.testing.assert_close(positions(owner)[row], seed(cid, positions(owner)[row]))
        assert owner.similarity_codebook.trust.views[row] == 0
