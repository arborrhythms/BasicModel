"""Item 2.3 round 1: for-lane radii, independent against evidence and admission."""
from types import SimpleNamespace
import math
import pytest
import torch

from ClauseRow import Clause, Composition
from IdentityRows import mint, point
from Meaning import ConceptualMeaning
from test_review13_subspaces import fixture as paired_fixture
from ConceptTrust import observe, positive_length


def test_named_word_identity_survives_below_binding_threshold_and_rebuild():
    from definition_fixtures import agreeing_views
    owner, store, obj, atoms = learned(1)
    word = owner.interpret.lookup_word([4], [], form='wug')
    agreeing_views(owner, word)
    description = dict(owner.interpret._pending[word], identity_key='round1e-form')
    row = owner.interpret.define(word, obj, description=description)
    index, alloc = owner.definitions, owner._concept_allocator
    count, next_id = len(store.parts), alloc.next_id
    for rebuild in (False, True):
        if rebuild:
            index.rebuild()
        assert index.word(form='wug') == word
        assert index.word(unit=(4,)) == word
        assert index.word(identity='round1e-form') == word
        assert owner.interpret.word_for_form('wug') == word
        assert word in index.word_ids and index.description(word) == description
        assert index.objects(word) == index.words(obj) == ()
        assert index.row(word, obj) is None and index.deref(word) is None
        assert word not in index.bound_words() and obj not in index.bound_objects()
        assert not owner.interpret.no_object(word)
        assert owner.interpret.remembered_objects(word) == (obj,)
        assert 'definition' in store.part_row(row)['readings']
        assert float(store.parts.evidence(row)[0]) == pytest.approx(-math.expm1(-1/3))
    observe(owner, obj, atoms[0]); observe(owner, obj, atoms[0])
    assert index.objects(word) == (obj,) and index.words(obj) == (word,)
    assert index.row(word, obj) == row and index.deref(word) == obj
    assert word in index.bound_words() and obj in index.bound_objects()
    index.rebuild()
    assert index.objects(word) == (obj,)
    assert len(store.parts) == count and alloc.next_id == next_id


def test_second_naming_below_threshold_keeps_description_and_merges_pair():
    from definition_fixtures import agreeing_views
    owner, store, obj, _ = learned(1)
    word = owner.interpret.lookup_word([4], [], form='wug')
    agreeing_views(owner, word)
    row = owner.interpret.define(word, obj, evidence=(.8,.1))
    alloc = owner._concept_allocator
    description = alloc.word_descriptions[word]
    count = len(store.parts)
    assert owner.definitions.objects(word) == ()
    assert owner.interpret.define(word, obj, evidence=(.4,.7)) == row
    assert alloc.word_descriptions[word] is description
    assert alloc.word_descriptions[word]['forms'] == ('wug',)
    assert len(store.parts) == count and alloc.named_identities[word] == {obj}
    assert (float(store.parts.truth_plus[row]), float(store.parts.truth_minus[row])) == pytest.approx((.8,.7))


def test_lookup_below_threshold_reuses_named_word_without_allocating():
    from definition_fixtures import agreeing_views
    owner, store, obj, _ = learned(1)
    word = owner.interpret.lookup_word([4], [], form='wug')
    agreeing_views(owner, word)
    owner.interpret.define(word, obj)
    before = owner._concept_allocator.next_id, len(store.parts)
    assert owner.definitions.objects(word) == ()
    assert owner.interpret.lookup_word([4], [], form='wug') == word
    assert (owner._concept_allocator.next_id, len(store.parts)) == before


def view(owner, store, identity, content, number):
    observe(owner, identity, content)


def fixture(*, trainable=False, capacity=32):
    owner, bank, _, _ = paired_fixture(meaning_pairs=4, capacity=capacity)
    owner._symbolic_order = 3
    if trainable:
        from ConceptTrust import ConceptTrust
        bank.W = torch.nn.Parameter(bank.W.detach().clone())
        bank.trust = ConceptTrust(owner, bank)
    return owner, owner._fixture_occurrences, torch.eye(11).roll(-3, dims=0)


def learned(count, *, disagree=False, trainable=False):
    owner, store, atoms = fixture(trainable=trainable)
    identity = mint(owner, atoms[0])
    for i in range(1, count):
        view(owner, store, identity, atoms[1] if disagree else atoms[0], i)
    return owner, store, identity, atoms


def relation(owner, store, left, right, *, pair=(.8, .4), trust=None):
    a, b = point(owner, left), point(owner, right)
    term = Composition(torch.maximum(a,b),
        (Composition(a, reference=left), Composition(b, reference=right)), operation='part')
    clause = Clause(ConceptualMeaning.from_description(term.point), point=term.point,
                    composition=term, evidence=pair)
    return store.write_clause(clause, evidence=pair, trust=trust)[1]


def built(owner, value):
    identity = owner.new_concept()
    from order_fixtures import force_order
    force_order(owner, identity, 0)
    row = owner._csw_concept_row(0, identity)
    bank = owner.similarity_codebook
    with torch.no_grad():
        bank.W[row].copy_(value)
        bank.mereology.placement[row].copy_(torch.nn.functional.normalize(value[3:7], dim=0))
        bank.mereology.placement_witnesses[row] = 1
    owner.freeze_concept(identity)
    return identity


def test_views_and_stability_are_the_lane_length():
    owner, _, once, _ = learned(1)
    owner10, _, ten, _ = learned(10)
    unstable, _, mixed, _ = learned(10, disagree=True)
    assert positive_length(owner, once) == pytest.approx(-math.expm1(-1/3))
    assert positive_length(owner10, ten) == pytest.approx(-math.expm1(-10/3))
    assert positive_length(unstable, mixed) == pytest.approx(.1 * -math.expm1(-10/3))


def test_optimizer_step_preserves_for_lengths_and_leaves_against_update():
    from Models import BasicModel
    from Optimizer import SGD
    owner, _, identity, atoms = learned(3, trainable=True)
    frozen = built(owner, atoms[2])
    bank = owner.similarity_codebook
    before = bank.W.detach().clone()
    optimizer = SGD([bank.W], lr=.1)
    (-bank.W * torch.arange(1, 12, device=bank.W.device)).sum().backward()
    optimizer.step()
    rows = [owner._csw_row_of(x) for x in (identity, frozen)]
    torch.testing.assert_close(bank.W[rows,3:7].norm(dim=-1), before[rows,3:7].norm(dim=-1))
    against = bank.W[rows,7:].detach().clone()
    torch.testing.assert_close(against, before[rows,7:] + .1*torch.arange(8,12))
    model = SimpleNamespace(wholeSpaces=[], conceptualSpaces=[owner],
        _clamp_symbolic_codebook=lambda: None, _normalize_perceptual_embedding=lambda: None)
    BasicModel._project_sentence_parameters(model)
    torch.testing.assert_close(bank.W[rows,3:7].norm(dim=-1), before[rows,3:7].norm(dim=-1))
    torch.testing.assert_close(bank.W[rows,7:], against)
    assert not torch.equal(bank.W[rows[0]], before[rows[0]])
    assert not owner.null_concept_row.any()


def test_relation_stores_source_and_reads_current_operand_trust():
    pairs = []
    for count in (1, 10):
        owner, store, identity, atoms = learned(count)
        other = built(owner, atoms[2])
        length = positive_length(owner, identity)
        row = relation(owner, store, identity, other, trust=.7)
        stored = torch.stack((store.parts.truth_plus[row], store.parts.truth_minus[row])).clone()
        torch.testing.assert_close(stored, stored.new_tensor((.56, .28)))
        actual = torch.stack(store.parts.evidence(row)).clone()
        torch.testing.assert_close(actual, stored*length)
        for i in range(count, count+3):
            view(owner, store, identity, atoms[0], i)
        assert bool((torch.stack(store.parts.evidence(row)) > actual).all())
        torch.testing.assert_close(torch.stack((store.parts.truth_plus[row], store.parts.truth_minus[row])), stored)
        pairs.append(actual)
    assert bool((pairs[0] < pairs[1]).all())


def test_three_operands_do_not_multiply_trust():
    owner, store, identity, atoms = learned(10)
    a, b, c = (Composition(.8*atoms[i], reference=built(owner, .8*atoms[i])) for i in (1,2,3))
    def write(term):
        clause = Clause(ConceptualMeaning.from_description(term.point), point=term.point,
            composition=term, evidence=(1., .5))
        return store.write_clause(clause)[1]
    two = Composition(torch.maximum(a.point,b.point), (a,b), operation='part')
    three = Composition(torch.maximum(two.point,c.point), (two,c), operation='part')
    first, second = write(two), write(three)
    assert float(store.parts.evidence(first)[0]) == pytest.approx(.8)
    assert float(store.parts.evidence(second)[0]) == pytest.approx(.8)


def test_definition_threshold_follows_effective_positive_evidence():
    owner, store, _, atoms = learned(0)
    word = owner.interpret.lookup_word([4], [], form='x')
    content = owner.similarity_codebook.trust.content[owner._csw_row_of(word)].clone()
    original, genuine = built(owner, atoms[2]), built(owner, atoms[3])
    view(owner, store, word, content, 0)
    chain = relation(owner, store, original, word)
    named = owner.interpret.define(word, genuine)
    assert named is not None and named != chain
    assert not owner.definitions.objects(word)
    for i in range(1,3):
        view(owner, store, word, content, i)
    assert owner.definitions.objects(word) == (genuine,)
    owner.definitions.rebuild()
    assert owner.definitions.objects(word) == (genuine,)
    assert original not in owner.interpret.remembered_objects(word)
    # A counter-supported named row does not pass the positive-pole threshold.
    other = built(owner, atoms[1])
    owner.interpret.define(word, other, evidence=(0., 1.))
    assert owner.definitions.objects(word) == (genuine,)


def test_checkpoint_preserves_preallocated_lengths_and_view_counters():
    owner, store, identity, atoms = learned(3)
    bank = owner.similarity_codebook
    state = {k:v.clone() for k,v in bank.state_dict().items()}
    parts = {k:v.clone() for k,v in store.parts.state_dict().items()}
    storage = {name:value.data_ptr() for name,value in bank.trust.named_buffers()}
    part_storage = {name:value.data_ptr() for name,value in store.parts.named_buffers()}
    assert bank.trust.views.dtype == torch.long
    assert bank.trust.views[owner._csw_row_of(identity)] == 3
    assert not hasattr(store.parts, 'witness_content') and not hasattr(store.parts, 'witness_valid')
    bank.load_state_dict(state)
    store.parts.load_state_dict(parts)
    assert storage == {name:value.data_ptr() for name,value in bank.trust.named_buffers()}
    assert part_storage == {name:value.data_ptr() for name,value in store.parts.named_buffers()}
    torch.testing.assert_close(bank.trust.lengths, state['trust.lengths'])


def test_mint_has_one_view_and_for_radius_with_zero_against():
    from test_review13_subspaces import fixture as paired_fixture
    owner, bank, _, _ = paired_fixture(meaning_pairs=4)
    owner._symbolic_order = 3
    content = torch.eye(11)[0] + torch.eye(11)[3]
    identity = mint(owner, content)
    from IdentityRows import witnesses
    assert len(witnesses(owner, identity)) == 0
    assert bank.trust.views[owner._csw_row_of(identity)] == 1
    row = owner._csw_row_of(identity)
    value = bank.lookup_rows(row)
    torch.testing.assert_close(value[:3], content[:3])
    assert value[3:7].norm().item() == pytest.approx(-math.expm1(-1/3))
    assert not value[7:11].any()
    assert bank.trust.lengths[row,1] == 0.
    before = value.clone()
    bank.mereology.begin_forward()
    torch.testing.assert_close(bank.lookup_rows(row), before)


def test_kept_word_views_use_written_content_and_for_radius_only():
    from test_review13_subspaces import fixture as paired_fixture
    from FamiliarTier import commit_read_words
    owner, bank, _, _ = paired_fixture(meaning_pairs=4)
    owner._model.wholeSpaces = [SimpleNamespace(subspace=SimpleNamespace(what=None))]
    word = owner.interpret.lookup_word([4], [], form='word')
    row = owner._csw_row_of(word)
    content = bank.lookup_rows(row).clone()
    reading = SimpleNamespace(word_ids=torch.tensor([word]), leaves=content[None], symbol_where=None)
    bank.mereology.place_rows([row], torch.zeros(len(bank.W)), b'word-views')
    lengths = []
    for i in range(3):
        commit_read_words(owner, reading, batch=0, document='views', position=i)
        lengths.append(bank.lookup_rows(row)[3:7].norm().item())
        assert not bank.lookup_rows(row)[7:].any()
        assert bank.trust.lengths[row,1] == 0.
    assert lengths == pytest.approx([-math.expm1(-n/3) for n in (1,2,3)])
    torch.testing.assert_close(bank.lookup_rows(row)[:3], content[:3])
    assert bank.trust.views[row] == 3
    assert bank.trust.agreement[row] == pytest.approx(3.)
    assert len(owner._fixture_occurrences) == len(owner._concept_allocator.parts) == 0
    agreement = bank.trust.agreement.clone()
    bank.W[row,0] = .9
    torch.testing.assert_close(bank.trust.agreement, agreement)


@pytest.mark.parametrize('key,value', [('trustViews',0), ('trustViews',float('nan')), ('definitionTrust',1.1)])
def test_invalid_trust_configuration_fails_at_construction(monkeypatch, key, value):
    from util import TheXMLConfig
    from ConceptTrust import ConceptTrust
    owner, _, _, _ = learned(1)
    original = TheXMLConfig.space
    monkeypatch.setattr(TheXMLConfig, 'space', lambda space, name, *args, **kwargs:
        value if name == key else original(space,name,*args,**kwargs))
    with pytest.raises(ValueError, match=key):
        ConceptTrust(owner, owner.similarity_codebook)


@pytest.mark.parametrize('whitespace', [False, True])
def test_aligned_serial_sentence_has_no_space_or_punctuation_identity(monkeypatch, whitespace):
    from Models import BasicModel
    from test_cs_sparse_weights import _cs
    owner = _cs(nS=512)
    words = ['y', ' ', 'is', ' ', 'x', ' ', 'plus', ' ', 'two', '.']
    isp = SimpleNamespace(_ar_word_part_ids=torch.arange(1,11).reshape(1,10,1),
        _ar_word_part_mask=torch.ones(1,10,1,dtype=torch.bool),
        _word_active_mask=torch.ones(1,10,dtype=torch.bool), _packed_sentence_ids=torch.zeros(1,10,dtype=torch.long))
    ws = SimpleNamespace(whitespace_units=whitespace, property_rows_for_bytes=lambda word: [])
    ps = SimpleNamespace(_forward_input={'word_texts':[words]}, fuse_parts=lambda parts: parts)
    for space in (ps, ws):
        space.nWhat = 8
        space.subspace = SimpleNamespace(what=SimpleNamespace(lookup_rows=lambda rows: torch.zeros(*rows.shape, 8)))
    model = SimpleNamespace(inputSpace=isp, wholeSpace=ws, perceptualSpace=ps,
        conceptualSpaces=[owner], _aligned_serial_word_mode=lambda: True)
    BasicModel._stage_serial_concept_rows(model)
    for word in set(words):
        identity = owner.interpret.word_for_form(word)
        assert (identity is not None) == (word not in (' ', '.'))
    assert all(int(isp._ar_word_concept_ids[0,i]) == -1 for i,w in enumerate(words) if w in (' ', '.'))


def test_views_cost_no_rows_and_relations_do_not_count_as_views():
    owner, store, identity, atoms = learned(3)
    bank = owner.similarity_codebook
    row = owner._csw_row_of(identity)
    assert len(store) == len(store.parts) == 0
    assert bank.trust.views[row] == 3
    before = bank.trust.lengths.clone(), bank.trust.agreement.clone()
    other = built(owner, atoms[1])
    relation(owner, store, identity, other)
    assert bank.trust.views[row] == 3
    torch.testing.assert_close(bank.trust.lengths[row], before[0][row])
    torch.testing.assert_close(bank.trust.agreement[row], before[1][row])
    store.parts.retain([])
    assert bank.trust.views[row] == 3


@pytest.mark.parametrize('paired', [False, True])
def test_untracked_chart_keeps_signed_content_and_whole_norm(paired, monkeypatch):
    from test_cs_sparse_weights import _cs
    owner = paired_fixture(meaning_pairs=0)[0] if paired else _cs(nS=32)
    word = owner.interpret.lookup_word([4], [], form='word')
    bank = owner.similarity_codebook
    value = bank.lookup_rows(owner._csw_row_of(word)).clone()
    before = bank.W.clone()
    negative = -torch.ones(bank.W.shape[-1])
    observe(owner, word, negative)
    bank.trust.remember(word, negative)
    bank.trust.capture(owner._csw_row_of(word))
    bank.trust.project()
    torch.testing.assert_close(bank.W, before)
    assert not bank.trust.views.any()
    assert positive_length(owner, word) == 1.
    assert positive_length(owner, value=negative) == 1.
    from Queries import ThoughtConceptualCapability
    from util import TheXMLConfig
    get = TheXMLConfig.get
    monkeypatch.setattr(TheXMLConfig, 'get', lambda path, *a, **kw:
        None if path == 'architecture.data.dataType' else get(path,*a,**kw))
    probe = value.new_full(value.shape, -.1)
    extent = ThoughtConceptualCapability(owner, lambda a,b: 0.).extent(probe)
    assert extent['support_true'] == pytest.approx(.1*len(probe)**.5)
    assert extent['support_false'] == 0.
    assert positive_length(owner, value=probe) == 1.


def test_unplaced_word_has_radius_without_invented_direction():
    owner, store, _ = fixture()
    word = owner.interpret.lookup_word([4], [], form='word')
    row = owner._csw_row_of(word)
    content = owner.similarity_codebook.trust.content[row].clone()
    for n in range(3):
        view(owner, store, word, content, n)
    assert positive_length(owner, word) == pytest.approx(1-math.exp(-1))
    assert not owner.similarity_codebook.W[row,3:].any()
    assert not owner.similarity_codebook.lookup_rows(row)[3:].any()


def test_mint_without_occurrence_store_keeps_the_landed_contract(monkeypatch):
    owner, _, atoms = fixture()
    monkeypatch.setattr(owner, '_closed_clause_store', lambda: None)
    identity = mint(owner, atoms[0])
    assert owner.similarity_codebook.trust.views[owner._csw_row_of(identity)] == 1
    monkeypatch.setattr(owner, '_preflight_concept_row', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('full')))
    assert mint(owner, atoms[0]) is None


@pytest.mark.parametrize('reference', [(1,-7), (2,5000)])
def test_nonconcept_operand_contributes_one(reference):
    owner, store, _ = fixture()
    row = store.parts.add(reference, (2,5001), truth_plus=.8, truth_minus=.3)
    assert tuple(map(float, store.parts.evidence(row))) == pytest.approx((.8, .3))


def test_ordered_mereology_rises_without_reclosing_and_falls_with_stability():
    from Memory import RELATION
    owner, store, wug, atoms = learned(1)
    bird, winged = built(owner, atoms[1]), built(owner, atoms[2])
    parts = store.parts
    first = parts.add(wug, bird, truth_plus=1.)
    second = parts.add(bird, winged, truth_plus=1.)
    chain = parts.add((RELATION,first), (RELATION,second), truth_plus=1.)
    word = owner.interpret.lookup_word([4], [], form='wug')
    from definition_fixtures import agreeing_views
    agreeing_views(owner, word)
    named = owner.interpret.define(word, wug)
    before = parts.truth_plus.clone(), parts.truth_minus.clone()
    assert float(parts.evidence(first)[0]) == pytest.approx(-math.expm1(-1/3))
    assert float(parts.evidence(second)[0]) == pytest.approx(1.)
    assert float(parts.evidence(chain)[0]) == pytest.approx(-math.expm1(-1/3))
    assert owner.definitions.objects(word) == ()
    observe(owner,wug,atoms[0]); observe(owner,wug,atoms[0])
    assert float(parts.evidence(first)[0]) == pytest.approx(-math.expm1(-1))
    assert float(parts.evidence(chain)[0]) == pytest.approx(-math.expm1(-1))
    assert owner.definitions.objects(word) == (wug,)
    old = parts.effective_plus.clone()
    observe(owner,wug,atoms[3])
    parts.refresh_effective()
    assert parts.effective_plus[first] < old[first]
    assert parts.effective_plus[chain] < old[chain]
    assert parts.effective_plus[named] < old[named]
    assert parts.effective_plus[second] == old[second]
    for row in parts.indices().tolist():
        for kind, child in parts.references(row):
            if kind == RELATION:
                assert max(map(float,parts.evidence(row))) <= max(map(float,parts.evidence(child)))
    torch.testing.assert_close(parts.truth_plus,before[0])
    torch.testing.assert_close(parts.truth_minus,before[1])


def test_negation_merges_counterevidence_and_structural_close_never_lowers():
    owner, store, word, atoms = learned(3)
    obj = built(owner,atoms[1])
    row = relation(owner,store,word,obj,pair=(.8,0.),trust=.8)
    relation(owner,store,word,obj,pair=(0.,.7),trust=.7)
    assert (float(store.parts.truth_plus[row]),float(store.parts.truth_minus[row])) == pytest.approx((.64,.49))
    store.parts.add(word,obj,truth_plus=1.)
    relation(owner,store,word,obj,pair=(0.,0.))
    assert float(store.parts.truth_plus[row]) == 1.


def test_configured_truth_over_unread_words_keeps_its_source_pair():
    owner, store, _ = fixture()
    words = [owner.interpret.lookup_word([i],[],form=str(i)) for i in (4,7)]
    row = relation(owner,store,*words,pair=(1.,0.),trust=.9)
    assert tuple(map(float,store.parts.evidence(row))) == (0.,0.)
    assert float(store.parts.truth_plus[row]) == pytest.approx(.9)
    from definition_fixtures import agreeing_views
    agreeing_views(owner,*words)
    assert float(store.parts.evidence(row)[0]) == pytest.approx(.9*-math.expm1(-1))
    assert float(store.parts.truth_plus[row]) == pytest.approx(.9)


def test_lexical_operator_is_retained_and_composed_at_the_read():
    from dataclasses import replace
    owner,store,operator,atoms = learned(1)
    a,b = (owner.interpret.lookup_word([i],[],form=str(i)) for i in (4,7))
    from definition_fixtures import agreeing_views
    agreeing_views(owner,a,b)
    clause = Clause(ConceptualMeaning.from_description(atoms[1]),relation='operator',
                   refs=(a,operator,b),evidence=(.8,.2))
    row = store.write_clause(clause)[1]
    assert int(store.parts.operator[row]) == operator
    assert tuple(map(float,store.parts.evidence(row))) == pytest.approx((.8*-math.expm1(-1/3),.2*-math.expm1(-1/3)))
    assert (2,operator) in store.parts.protected_references([row])
    state={name:value.clone() for name,value in store.parts.state_dict().items()}
    store.parts.load_state_dict(state)
    assert int(store.parts.operator[row])==operator
    observe(owner,operator,atoms[0])
    assert float(store.parts.evidence(row)[0]) == pytest.approx(.8*-math.expm1(-2/3))
    with pytest.raises(ValueError,match='different lexical operators'):
        store.parts.add(a,b,operator=b)


def test_order_survives_retention_checkpoint_and_rejects_cycles():
    from Memory import LTM1, SYMBOL, RELATION
    parts=LTM1(16)
    a=parts.add(100,101,truth_plus=.8,symbol=8)
    b=parts.add((RELATION,a),(SYMBOL,102),truth_plus=.9,symbol=3)
    c=parts.add(103,104,truth_plus=.5,symbol=11)
    pointer=parts.relation_order.data_ptr()
    parts.retain([a,b])
    assert parts.relation_order[:2].tolist()==[a,b]
    saved={name:value.clone() for name,value in parts.state_dict().items()}
    parts.load_state_dict(saved)
    assert parts.relation_order.data_ptr()==pointer
    assert float(parts.evidence(b)[0])==pytest.approx(.72)
    bad={name:value.clone() for name,value in saved.items()}
    bad['refs'][a,0]=b;bad['ref_kinds'][a,0]=RELATION
    with pytest.raises(ValueError,match='cyclic'):
        parts.load_state_dict(bad)
    parts.refs[a,0]=b;parts.ref_kinds[a,0]=RELATION
    with pytest.raises(ValueError,match='cyclic'):
        parts.refresh_effective()


def test_item107_generic_sentence_never_names_its_chain(monkeypatch):
    from test_clause_acceptance import SentenceFixture
    owner,_,atoms=fixture(capacity=160)
    f=SentenceFixture(monkeypatch,native_cube=False,cs=owner)
    word=owner.interpret.lookup_word([4],[],form='wug')
    original,genuine=built(owner,atoms[1]),built(owner,atoms[2])
    left=Composition(point(owner,original),reference=original)
    right=Composition(owner.similarity_codebook.trust.content[owner._csw_row_of(word)].clone(),reference=word,word_id=word)
    value=torch.maximum(left.point,right.point)
    term=Composition(value,(left,right),operation='union',formal_name='union')
    sentence=Clause(ConceptualMeaning.from_description(value),point=value,composition=term)
    chain=f.store.write_clause(sentence,evidence=(1.,0.))
    assert chain[0]=='sym' and not owner.definitions.objects(word)
    named=owner.interpret.define(word,genuine)
    assert named != chain[1] and not owner.definitions.objects(word)
    from definition_fixtures import agreeing_views
    agreeing_views(owner,word)
    assert owner.definitions.objects(word)==(genuine,)
    reclosed=f.store.write_clause(sentence,evidence=(1.,0.))
    assert isinstance(reclosed,int)  # Known operands now compress to an occurrence.
    assert f.store.row(reclosed)['evidence']==(1.,0.)
    assert f.store.parts.find(original,word)==chain[1]
    owner.definitions.rebuild()
    assert owner.definitions.objects(word)==(genuine,)
    assert original not in owner.interpret.remembered_objects(word)


def test_relation_endpoint_uses_its_larger_effective_pole():
    from Memory import LTM1, RELATION
    parts=LTM1(12)
    child=parts.add(100,101,truth_minus=.7)
    parent=parts.add((RELATION,child),102,truth_plus=.8,truth_minus=.4)
    assert tuple(map(float,parts.evidence(parent)))==pytest.approx((.56,.28))
    parts.add(100,101,truth_plus=.9)
    assert tuple(map(float,parts.evidence(parent)))==pytest.approx((.72,.36))
    assert float(parts.truth_minus[child])==pytest.approx(.7)


def test_unreferenced_empty_leaf_contributes_one_at_the_close():
    owner,store,atoms=fixture()
    known=built(owner,atoms[1])
    left=Composition(torch.zeros_like(atoms[1]))
    right=Composition(point(owner,known),reference=known)
    term=Composition(right.point,(left,right),operation='part')
    clause=Clause(ConceptualMeaning.from_description(term.point),point=term.point,composition=term)
    row=store.write_clause(clause,trust=.9)[1]
    assert tuple(map(float,store.parts.evidence(row)))==pytest.approx((.9,0.))
    with pytest.raises(IndexError,match='unavailable'):
        store.parts.evidence(0)


def test_stored_concept_extent_is_for_radius_and_negate_exchanges_it(monkeypatch):
    from Queries import ThoughtConceptualCapability
    from util import TheXMLConfig
    owner, _, identity, _ = learned(3)
    get, space = TheXMLConfig.get, TheXMLConfig.space
    monkeypatch.setattr(TheXMLConfig, 'get', lambda path, *a, **kw:
        'embedding' if path == 'architecture.data.dataType' else get(path,*a,**kw))
    monkeypatch.setattr(TheXMLConfig, 'space', lambda name, field, *a, **kw:
        3 if (name,field) == ('PartSpace','nDim') else space(name,field,*a,**kw))
    reader = ThoughtConceptualCapability(owner, lambda a,b: 0.)
    value = owner.similarity_codebook.lookup_rows(owner._csw_row_of(identity))
    pair = reader.extent(value)
    radius = -math.expm1(-1)
    assert (pair['support_true'], pair['support_false']) == pytest.approx((radius,0.))
    negated = reader.extent(reader.negate(value))
    assert (negated['support_true'], negated['support_false']) == pytest.approx((0.,radius))


def test_observation_projection_and_lookup_preserve_written_against_evidence():
    owner, _, identity, atoms = learned(1)
    bank = owner.similarity_codebook
    row = owner._csw_row_of(identity)
    against = torch.tensor([.2,.4,.1,0.])
    bank.W[row,7:].copy_(against)
    bank.trust.capture(row)
    captured = bank.trust.lengths[row,1].clone()
    observe(owner,identity,atoms[0])
    assert bank.trust.lengths[row,1] == captured
    bank.trust.project()
    torch.testing.assert_close(bank.W[row,7:],against,rtol=0,atol=0)
    torch.testing.assert_close(bank.lookup_rows(row)[7:],against,rtol=0,atol=0)


def test_empty_content_counts_views_then_adopts_first_nonempty_reference():
    owner, _, atoms = fixture()
    bank = owner.similarity_codebook
    identity = mint(owner,torch.zeros(11))
    row = owner._csw_row_of(identity)
    assert bank.trust.views[row] == bank.trust.agreement[row] == 1
    observe(owner,identity,torch.zeros(11))
    observe(owner,identity,atoms[0])
    assert bank.trust.views[row] == bank.trust.agreement[row] == 3
    torch.testing.assert_close(bank.trust.content[row],atoms[0])
    assert positive_length(owner,identity) == pytest.approx(-math.expm1(-1))
    observe(owner,identity,atoms[1])
    assert bank.trust.views[row] == 4 and bank.trust.agreement[row] == 3
    torch.testing.assert_close(bank.trust.content[row],atoms[0])
    assert positive_length(owner,identity) == pytest.approx(.75*-math.expm1(-4/3))


@pytest.mark.parametrize('signs',[1,3])
def test_null_kind_named_from_held_signs_reaches_the_definition_threshold(signs):
    from FamiliarTier import promote_identity
    from definition_fixtures import agreeing_views
    owner,store,_=fixture()
    word=owner.interpret.lookup_word([4],[],form='Felix')
    agreeing_views(owner,word)
    empty=torch.zeros(11)
    # These are explicitly held observations of one token. This certifies
    # D7's counter through promotion; Q2's sign-close certificates are below.
    for n in range(signs):
        row=store.append_meaning(ConceptualMeaning.from_description(empty),order=0,
            document_key='held-signs',sentence_index=n,evidence=(1.,0.),when=False)
        assert not store.slots[row].any()
        occurrence=store.occurrence_of(row)
        if n == 0:
            key=owner.mtm.open(occurrence,kind=owner.null_concept_id,when=n,value=empty)
        else:
            owner.mtm.bind(key,occurrence,kind=owner.null_concept_id,when=n,value=empty)
    assert owner.mtm.entries[key].kind == owner.null_concept_id
    identity=promote_identity(owner,key,word=word,evidence=(1.,0.))
    bank=owner.similarity_codebook
    seat=owner._csw_row_of(identity)
    assert bank.trust.views[seat] == bank.trust.agreement[seat] == signs
    assert positive_length(owner,identity) == pytest.approx(-math.expm1(-signs/3))
    assert owner.interpret.remembered_objects(word) == (identity,)
    assert owner.definitions.objects(word) == ((identity,) if signs == 3 else ())
    assert not owner.null_concept_lanes.any()


@pytest.mark.parametrize('views', [1, 3])
def test_seatless_word_definition_uses_only_its_objects_radius(views):
    owner, store, obj, _ = learned(views)
    alloc = owner._concept_allocator
    word = alloc.new_concept()
    alloc.symbol_types[word] = {'word'}
    assert owner._csw_row_of(word) is None
    assert positive_length(owner, word) == 1.
    row = owner.interpret.define(word, obj, evidence=(1., .2),
        description=dict(forms=('named',), parts=((901,),), wholes=()))
    assert row is not None
    radius = -math.expm1(-views/3)
    stored = torch.stack((store.parts.truth_plus[row], store.parts.truth_minus[row]))
    torch.testing.assert_close(torch.stack(store.parts.evidence(row)), stored*radius)
    assert owner.definitions.objects(word) == ((obj,) if radius >= .5 else ())
    owner.definitions.rebuild()
    assert owner.definitions.objects(word) == ((obj,) if radius >= .5 else ())
    assert owner._csw_row_of(word) is None
    seated = owner.interpret.lookup_word([4], [], form='unobserved')
    assert owner._csw_row_of(seated) is not None
    assert positive_length(owner, seated) == 0.


def close_named_sign(owner, store, label, number):
    """Keep an empty sign, then consume its lexical naming edge at the close."""
    from FamiliarTier import define_pending, finish_definitions
    from Memory import OCCURRENCE, SYMBOL
    empty = torch.zeros(owner.nDim)
    row = store.append_meaning(ConceptualMeaning.from_description(empty), order=0,
        document_key='pending-signs', sentence_index=number, evidence=(1., 0.), when=False)
    occurrence = store.occurrence_of(row)
    key = owner.mtm.open(occurrence, kind=owner.null_concept_id, when=number, value=empty)
    left, right = (OCCURRENCE, occurrence[-1]), (SYMBOL, label)
    store.parts.add(left, right, truth_plus=1., truth_minus=0.)
    count, next_id = len(store.parts), owner._concept_allocator.next_id
    define_pending(owner, left, right, other_word=label, null_referent=True)
    if owner.mtm.entries[key].pending:
        assert len(store.parts) == count and owner._concept_allocator.next_id == next_id
    finish_definitions(owner)
    owner.mtm.close(store, stream=None, references=(occurrence,))
    return occurrence


@pytest.mark.parametrize('tracked', [True, False])
def test_sign_closes_wait_for_one_remembered_kind_then_join(tracked):
    import copy
    from definition_fixtures import agreeing_views
    from Memory import OCCURRENCE, SYMBOL
    owner, _, _, alloc = paired_fixture(meaning_pairs=4 if tracked else 0)
    owner._symbolic_order = 3
    store = owner._fixture_occurrences
    label = owner.interpret.lookup_word([4], (), form='Felix')
    agreeing_views(owner, label)
    first = close_named_sign(owner, store, label, 0)
    target, = owner.interpret.remembered_objects(label)
    seat = owner._csw_row_of(target)
    named = alloc.parts.find((SYMBOL, target), (SYMBOL, label))
    objects = {i for i, kinds in alloc.symbol_types.items() if 'object' in kinds}
    assert objects == {target}
    assert owner.mtm.entries[owner.mtm.token_of(first)].symbol == target
    assert owner.definitions.objects(label) == (() if tracked else (target,))
    stored = torch.stack((alloc.parts.truth_plus[named], alloc.parts.truth_minus[named])).clone()
    torch.testing.assert_close(torch.stack(alloc.parts.evidence(named)), stored * (-math.expm1(-1/3) if tracked else 1.))
    second = close_named_sign(owner, store, label, 1)
    key = owner.mtm.token_of(second)
    if tracked:
        entry = owner.mtm.entries[key]
        assert entry.pending and entry.kind == owner.null_concept_id and entry.symbol is None
        assert alloc.parts.find((OCCURRENCE, second[-1]), (SYMBOL, target)) is None
        assert owner.similarity_codebook.trust.views[seat] == 2
        assert owner.definitions.objects(label) == ()
        # Pending null-kind state and the observed-address ledger survive a checkpoint.
        owner.mtm.restore(copy.deepcopy(owner.mtm.state()))
        from FamiliarTier import define_pending
        left, right, _ = owner.mtm.entries[key].refinement['pending_sign_definition']
        rows, next_id = len(alloc.parts), alloc.next_id
        define_pending(owner, left, right, other_word=label, null_referent=True)
        assert owner.similarity_codebook.trust.views[seat] == 2
        assert len(alloc.parts) == rows and alloc.next_id == next_id
        third = close_named_sign(owner, store, label, 2)
        assert owner.similarity_codebook.trust.views[seat] == 3
        assert positive_length(owner, target) == pytest.approx(-math.expm1(-1.))
        for address in (second, third):
            entry = owner.mtm.entries[owner.mtm.token_of(address)]
            assert not entry.pending and entry.symbol == entry.kind == target
            assert alloc.parts.find((OCCURRENCE, address[-1]), (SYMBOL, target)) is not None
    else:
        assert owner.mtm.entries[key].symbol == target
    assert owner.definitions.objects(label) == (target,)
    assert {i for i, kinds in alloc.symbol_types.items() if 'object' in kinds} == objects
    assert owner._csw_row_of(target) == seat
    torch.testing.assert_close(torch.stack((alloc.parts.truth_plus[named], alloc.parts.truth_minus[named])), stored)


def test_pending_sign_expires_without_an_identity_or_member_row():
    from definition_fixtures import agreeing_views
    from Memory import OCCURRENCE, SYMBOL
    owner, store, _ = fixture()
    alloc = owner._concept_allocator
    label = owner.interpret.lookup_word([4], (), form='Felix')
    agreeing_views(owner, label)
    close_named_sign(owner, store, label, 0)
    target, = owner.interpret.remembered_objects(label)
    second = close_named_sign(owner, store, label, 1)
    key = owner.mtm.token_of(second)
    entry = owner.mtm.entries[key]
    assert entry.pending and entry.kind == owner.null_concept_id and entry.symbol is None
    next_id, rows = alloc.next_id, len(alloc.parts)
    for _ in range(owner.mtm.horizon + 1):
        owner.mtm.close(store, stream=None)
    assert key not in owner.mtm.entries and owner.mtm.token_of(second) is None
    assert alloc.next_id == next_id and len(alloc.parts) == rows
    assert alloc.parts.find((OCCURRENCE, second[-1]), (SYMBOL, target)) is None
    third = close_named_sign(owner, store, label, 2)
    assert owner.definitions.objects(label) == (target,)
    assert owner.mtm.entries[owner.mtm.token_of(third)].symbol == target
    assert alloc.parts.find((OCCURRENCE, second[-1]), (SYMBOL, target)) is None


def test_sign_sentence_observes_a_remembered_endpoint_before_joining(monkeypatch):
    from test_clause_acceptance import SentenceFixture
    from test_item25_completion import _sign_equation
    from definition_fixtures import agreeing_views
    owner, _, _, alloc = paired_fixture(meaning_pairs=4, capacity=160)
    owner._symbolic_order = 3
    f = SentenceFixture(monkeypatch, cs=owner, native_cube=False)
    label = owner.interpret.lookup_word([4], (), form='Felix')
    agreeing_views(owner, label)
    _sign_equation(f, 'This', 'Felix', 0)
    target, = owner.interpret.remembered_objects(label)
    _, second = _sign_equation(f, 'That', 'Felix', 1)
    key = owner.mtm.token_of(second)
    entry = owner.mtm.entries[key]
    assert entry.pending and entry.symbol is None and entry.kind == owner.null_concept_id
    left, right, _ = entry.refinement['pending_sign_definition']
    # The source sentence already wrote this edge to the remembered endpoint.
    assert right == (alloc.parts.SYMBOL, target)
    source = alloc.parts.find(left, right)
    pair = (float(alloc.parts.truth_plus[source]), float(alloc.parts.truth_minus[source]))
    assert owner.similarity_codebook.trust.views[owner._csw_row_of(target)] == 2
    _, third = _sign_equation(f, 'There', 'Felix', 2)
    assert owner.definitions.objects(label) == (target,)
    assert owner.similarity_codebook.trust.views[owner._csw_row_of(target)] == 3
    for address in (second, third):
        assert owner.mtm.entries[owner.mtm.token_of(address)].symbol == target
    assert alloc.parts.find(left, right) == source
    assert (float(alloc.parts.truth_plus[source]), float(alloc.parts.truth_minus[source])) == pair
    assert {i for i, kinds in alloc.symbol_types.items() if 'object' in kinds} == {target}
