"""Item 2.5 §38.2. One history returns to the weights each epoch."""
import copy
import json
import os
from pathlib import Path

import pytest
from part_fixtures import stored_pair
import torch

from FamiliarTier import promote_identity
from Memory import OCCURRENCE, RELATION, SYMBOL
from test_presentation_memory import begin, fixture, model_for, write


def receipt(name, data):
    target = os.environ.get('ITEM25_RECEIPT')
    if target:
        path = Path(target)
        path.mkdir(parents=True, exist_ok=True)
        (path / (name+'.json')).write_text(json.dumps(
            dict(torch_version=str(torch.__version__), **data), indent=2)+'\n')


def same_state(before, after):
    if torch.is_tensor(before):
        torch.testing.assert_close(before,after,rtol=0,atol=0)
    elif isinstance(before, dict):
        assert before.keys() == after.keys()
        for key in before:
            same_state(before[key],after[key])
    elif isinstance(before,(list,tuple)):
        assert len(before) == len(after)
        for left,right in zip(before,after,strict=True):
            same_state(left,right)
    else:
        assert before == after


def test_epoch_restores_every_relation_column_index_and_given_row():
    resolver, store, points = fixture()
    owner = resolver.components.owner
    bank = store.parts
    model = model_for(owner, store)
    a, b = owner.new_concept(), owner.new_concept()
    given = write(store, points[0], 'given', store.ORIGIN_USER)
    preexisting = bank.add(a, b, truth_plus=.3, truth_minus=.1, surprise=.75)
    bank.add((OCCURRENCE, given[-1]), (SYMBOL, a), truth_plus=.2)
    def indexes():
        bank._index()
        return copy.deepcopy((bank._pairs, bank._parts, bank._wholes, bank._references))
    alloc = owner._concept_allocator
    alloc.remember_singleton((b,0),(SYMBOL,a),None,points[0])
    alloc.singleton_rows[b,0] = ((SYMBOL,a),(SYMBOL,b))
    bank.refresh_effective()  # Snapshot the live derived authority, not its lazy zeros.
    initial = {k:v.clone() for k,v in bank.state_dict().items()}
    expected_indexes = indexes()
    pointers = {k:v.data_ptr() for k,v in bank.named_buffers()}
    begin(model)
    for k,v in bank.state_dict().items():
        assert torch.equal(v, initial[k])
    assert indexes() == expected_indexes and store.location(given) is not None
    bank.add(a, b, truth_plus=.85, truth_minus=.99, surprise=.01)
    store.truth_plus[store.location(given)] = .5
    store.access_count[store.location(given)] += 4
    from test_reference_close import known_concept
    c = known_concept(owner, points[2])
    alloc.remember_singleton((b,1),(SYMBOL,c),None,points[1])
    alloc.singleton_rows[b,1] = ((SYMBOL,c),(SYMBOL,b))
    outer = bank.add((RELATION, preexisting), (SYMBOL, c), truth_plus=.9)
    occurrence = write(store, points[1], 'epoch')
    bank.add((OCCURRENCE, occurrence[-1]), (RELATION, outer), truth_plus=.2)
    word = owner.interpret.lookup_word([17], [], form='label')
    owner.interpret.define(word, c)
    assert owner.definitions.objects(word) == (c,)
    begin(model)
    assert indexes() == expected_indexes
    assert set(alloc.singleton_members) == {(b,0)} and set(alloc.singleton_rows) == {(b,0)}
    torch.testing.assert_close(alloc.singleton_members[b,0][2], points[0])
    assert {k:v.data_ptr() for k,v in bank.named_buffers()} == pointers
    for k,v in bank.state_dict().items():
        assert torch.equal(v, initial[k]), k
    assert store.location(occurrence) is None
    idx = store.location(given)
    assert int(store.access_count[idx]) == 1 and float(store.truth_plus[idx]) == 1.
    assert not owner.definitions.objects(word)
    assert owner.interpret.word_for_form('label') == word
    assert owner._concept_allocator.named_identities[word] == {c}
    receipt('epoch-relation-columns',dict(columns=list(initial),indexes=['pairs','parts','wholes','references'],
                                          given=given,restored=True))


@pytest.mark.parametrize('member_order,evidence', [(1,(1.,1.)),(0,(1.,1.)),(1,(0.,1.))])
def test_relative_member_write_expires_with_its_epoch(member_order, evidence):
    from ClauseRow import Clause, Composition, predicate_identity
    from Meaning import ConceptualMeaning
    resolver, store, points = fixture()
    owner, model = resolver.components.owner, model_for(resolver.components.owner, store)
    alloc = owner._concept_allocator
    begin(model)
    symbols = []
    for i, point in enumerate(points[:3]):
        occ = write(store, point, ('admission',i), order=0)
        key = owner.mtm.open(occ, kind=None, value=point, when=i)
        from test_reference_close import known_concept
        symbols.append(known_concept(owner, point))
    a,b,target = symbols
    if member_order == 0:
        a,b = [owner.interpret.lookup_word([n], [], form=bytes([n])) for n in (97,98)]
    inside = alloc.parts.add(a,b)
    assert owner._concept_source_order(inside) == member_order and alloc.order_of(target) == 1
    operands = (Composition(points[0], reference=inside, order=member_order),
                Composition(points[2], reference=target, order=1))
    term = Composition(points[2], operands, operation='part', order=1)
    clause = Clause(ConceptualMeaning.from_description(points[2]), relation='part',
        refs=(inside, predicate_identity('part'), target), composition=term, order=1)
    outer = store.write_clause(clause, evidence=evidence)
    assert outer == ('sym',alloc.parts.find((RELATION,inside),(SYMBOL,target)))
    assert stored_pair(store,outer) == evidence
    assert store.row(outer)['evidence'] == (0., 0.)
    begin(model)
    assert not alloc.parts.contains(inside) and not alloc.parts.contains(outer[1])
    assert alloc.settle(target) == 1


def test_ambiguous_named_promotion_fails_before_minting():
    resolver,store,points = fixture()
    owner = resolver.components.owner
    alloc = owner._concept_allocator
    word = owner.interpret.lookup_word([23], [], form='label')
    ids = []
    for i,point in enumerate(points[:2]):
        occ = write(store, point, ('separate',i), order=0)
        key = owner.mtm.open(occ, kind=None, value=point, when=i)
        obj = promote_identity(owner,key)
        owner.interpret.define(word,obj)
        ids.append(obj)
    occ = write(store, points[2], 'named', order=0)
    token = owner.mtm.open(occ, kind=None, value=points[2], when=2, reading=word)
    counter, rows = alloc.next_id, len(alloc.parts)
    with pytest.raises(ValueError, match='ambiguous identity promotion') as exc:
        promote_identity(owner, token, word=word)
    assert f'words {[word]}' in str(exc.value) and 'label' in str(exc.value)
    assert f'denotations {sorted(ids)}' in str(exc.value)
    assert (alloc.next_id, len(alloc.parts)) == (counter, rows)
    assert owner.mtm.entries[token].symbol is None


def test_ten_epochs_ground_the_same_names_in_one_fixed_document():
    from DocumentCover import open_cover
    from Meaning import ConceptualMeaning
    resolver, store, points = fixture(percept=True)
    owner = resolver.components.owner
    alloc, model = owner._concept_allocator, model_for(owner, store)
    before = set(alloc.placement)
    names = set()
    measurements = []
    begin(model)
    for epoch in range(10):
        cover = open_cover(owner, store, stream=0, document='one-document')
        cover_symbol = store.raw_meaning_of(store.location(cover)).role_refs[0][1]
        word = owner.interpret.lookup_word([10], [], form='label')
        meaning = ConceptualMeaning.from_description(points[0])
        row = store.append_meaning(meaning, kind='observation', evidence=(1.,0.), order=0,
            document_key='one-document', sentence_index=0)
        occurrence = store.occurrence_of(row)
        token = owner.mtm.open(occurrence, kind=None, value=points[0], when=epoch, reading=word)
        identity = promote_identity(owner, token, word=word)
        assert store.parts.find((OCCURRENCE, occurrence[-1]), (SYMBOL, identity)) is not None
        assert owner.interpret.denotations(word) == (identity,)
        # This occurrence grounds the word's native reading, the object that
        # names it and the cover under which this one-document history occurred.
        lexical = store.parts.add((OCCURRENCE, occurrence[-1]), (SYMBOL,word), truth_plus=1.)
        grounded = {word:lexical, identity:occurrence, cover_symbol:cover}
        concepts = set(alloc.placement) - before - store.parts.epoch_handles
        assert set(grounded) == concepts
        if epoch:
            assert concepts == names
        new = sorted(concepts - names)
        names = concepts
        held = [len(store),len(store.parts)]
        begin(model)
        assert all(('occ', occurrence[-1]) not in alloc.refs(cid,'part') for cid in names)
        assert not owner.definitions.objects(word) and alloc.named_identities[word] == {identity}
        assert owner.interpret.word_for_form('label') == word
        measurements.append(dict(epoch=epoch+1, minted=new, grounded=sorted(grounded),
                                 rows_in_epoch=held, rows_after_boundary=[len(store),len(store.parts)]))
    assert [len(r['minted']) for r in measurements] == [3]+[0]*9
    assert len({tuple(r['rows_after_boundary']) for r in measurements}) == 1
    assert len({tuple(r['rows_in_epoch']) for r in measurements}) == 1
    receipt('epoch-grounding-ten',dict(epochs=measurements))


def test_native_train_keeps_last_grounding_and_recognizes_it_across_ten_epochs(tmp_path, monkeypatch, eager_reading):
    import xml.etree.ElementTree as ET
    import Forgetting
    from data import TheData
    from Models import BaseModel
    from reading_fixtures import force_absolute_reading
    from test_corpus import manifest

    path = tmp_path / 'one-document.xml'
    tree = ET.parse(Path(__file__).resolve().parents[1] / 'data/MM_corpus_orders.xml')
    tree.find('./architecture/training/packSentences').text = 'true'
    tree.find('./architecture/training/numWorkers').text = '0'
    tree.write(path)
    TheData.load('curriculum', dat={'corpusManifest': str(manifest(tmp_path, [
        dict(text='alpha.', split='train', stage=0, document='only', sentence=0)]))})
    monkeypatch.delenv('BASIC_RUN_TEST', raising=False)
    model = BaseModel.from_config(str(path), data=TheData)[0]
    owner, store = model._concept_owner(), model.symbolSpace.ltm_store
    alloc = owner._concept_allocator
    force_absolute_reading(model)
    original_boundary, run = Forgetting.model_boundary, model.runEpoch
    starts, seen, reports = [], set(), []
    baseline = set()
    def observed_boundary(target, **options):
        if target is not model or not options.get('epoch_boundary'):
            return original_boundary(target,**options)
        optimizer_state = copy.deepcopy(model._optimizer.state_dict())
        parameters = {k:p.detach().clone() for k,p in model.named_parameters()}
        first = not starts
        if first:
            history=copy.deepcopy((store.state_dict(),store.semantic_extras(),
                                   store.parts.state_dict(),owner.mtm.state()))
        result = original_boundary(target, **options)
        if first:
            same_state(history,(store.state_dict(),store.semantic_extras(),
                                store.parts.state_dict(),owner.mtm.state()))
            baseline.update(alloc.placement)
        same_state(optimizer_state,model._optimizer.state_dict())
        same_state(parameters,dict(model.named_parameters()))
        starts.append((len(store),len(store.parts)))
        return result

    def grounded(ltm, symbols):
        found = {cid: [] for cid in symbols}
        for i in range(len(ltm)):
            meaning = ltm.raw_meaning_of(i)
            for ref in meaning.role_refs:
                if ref and ref[0] in ('name','sym') and ref[1] in found:
                    found[ref[1]].append(ltm.occurrence_of(i))
        # A relative sentence is stored in LTM1. Its two tagged endpoints
        # ground their symbols at that relation's own recorded occurrence.
        for row in ltm.parts.indices().tolist():
            for tag,cid in ltm.parts.references(row):
                if tag == SYMBOL and cid in found:
                    found[cid].append(ltm.occurrence_of(('sym',row)))
        # An uninterpreted word is grounded by its basic-percept occurrence,
        # even when the grammar keeps no conceptual reference to that word.
        # Follow the stored member reference through the loaded store's owner.
        space = ltm._concept_owner()
        words = {percept: word for word, percept in space.basic_percepts.words.items()
                 if word in found}
        for entry in space.mtm.entries.values():
            if entry.refinement.get('member_kind') != 'percept':
                continue
            for address, concept in entry.refinement.get('members', ()):
                if concept[:1] == ('bp',) and concept[1] in words:
                    assert ltm.location(address) is not None
                    found[words[concept[1]]].append(address)
        assert all(found.values()), found
        return found

    def observed_epoch(*args, **kw):
        result = run(*args, **kw)
        if kw.get('optimizer') is not None:
            symbols = set(alloc.placement) - baseline - alloc.parts.epoch_handles
            records = grounded(store,symbols)
            assert len(store.parts) > 0 and len(store) > 0
            reports.append(dict(epoch=len(reports)+1, minted=sorted(symbols-seen),
                grounding=records, rows=[len(store),len(store.parts)]))
            seen.update(symbols)
        return result

    restored = None
    try:
        monkeypatch.setattr(Forgetting,'model_boundary',observed_boundary)
        monkeypatch.setattr(model,'runEpoch',observed_epoch)
        model.runTrial(numEpochs=2, batchSize=1, lr=.001)
        assert len(reports)==len(starts)==2
        assert len(store) >= reports[-1]['rows'][0] and len(store.parts) >= reports[-1]['rows'][1]
        assert all(store.location(ref) is not None for refs in reports[-1]['grounding'].values() for ref in refs)
        checkpoint = tmp_path / 'finished.ckpt'
        model.save_weights(str(checkpoint))
        restored = BaseModel.from_config(str(path), data=TheData)[0]
        assert restored.load_weights(str(checkpoint), require_match=True)
        loaded = restored.symbolSpace.ltm_store
        assert grounded(loaded,seen) == grounded(store,seen)
        pool = restored._concept_owner().mtm
        assert pool.entries.keys() == owner.mtm.entries.keys()
        for key,entry in owner.mtm.entries.items():
            assert pool.entries[key].occurrences == entry.occurrences
            torch.testing.assert_close(pool.value_of(key), owner.mtm.value_of(key))

        for name,value in store.parts.state_dict().items():
            assert torch.equal(value,loaded.parts.state_dict()[name])
        for _ in range(8):
            model.runEpoch(optimizer=model._optimizer, batchSize=1, split='train')
        assert len(starts)==len(reports)==10 and len(set(starts))==1
        assert reports[0]['minted'] and all(not row['minted'] for row in reports[1:])
        receipt('epoch-native-training',dict(boundary_rows=starts,epochs=reports,
                second_epoch_checkpoint_preserves_rows=True, first_epoch_identical=True,
                parameters_and_optimizer_preserved_at_every_boundary=True))
    finally:
        for instance in (model,restored):
            if instance is not None:
                instance.End()
                instance.symbolSpace.soft_reset()


@pytest.mark.parametrize('older_unstamped_class', [False, True])
def test_later_candidate_reads_a_promoted_classes_order_after_epoch_withdrawal(older_unstamped_class):
    from FamiliarTier import observe_candidate
    from test_attention_promotion import _close
    from test_familiar_tier import placed_fixture
    cs, _, alloc, store, (a,b,c) = placed_fixture()
    cs._promotion_enabled = True
    model = model_for(cs,store)
    begin(model)

    def promote(order, support):
        for _ in range(cs.concept_mint_count):
            observe_candidate(cs,order,support)
            _close(cs)
        symbol, = cs.mtm.promote_candidates(cs)
        return symbol

    lower = promote(1,{a:2.,b:1.})
    assert alloc.order_of(lower) == 1
    lower_row = cs._csw_row_of(lower)
    assert alloc.reference_orders[lower] == 1
    if older_unstamped_class:
        del alloc.reference_orders[lower]
    begin(model)
    assert not store.parts.parts((SYMBOL,lower))
    upper = promote(1,{lower_row:2.,c:1.})
    assert alloc.reference_orders[lower] == 1 and alloc.reference_orders[upper] == 2
    assert store.parts.find((SYMBOL,lower),(SYMBOL,upper)) is not None
    begin(model)
    assert alloc.order_of(lower) == 1 and alloc.order_of(upper) == 2
    assert not store.parts.parts((SYMBOL,lower))
    assert observe_candidate(cs,1,{a:2.,b:1.}) is None
    assert {r for r in store.parts.parts((SYMBOL,lower))} == {
        (SYMBOL,cs.concept_id_at_row(a)),(SYMBOL,cs.concept_id_at_row(b))}
    assert not cs.mtm.promote_candidates(cs)
    assert promote(1,{lower_row:2.,c:1.}) == upper
    assert store.parts.find((SYMBOL,lower),(SYMBOL,upper)) is not None
