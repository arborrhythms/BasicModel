"""Native order, grounding and learned weights in one retained history."""
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
        occ = write(store, point, ('admission',i))
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
    assert tuple(float(store.row(outer)[lane]) for lane in ('truth_plus', 'truth_minus')) == evidence
    begin(model)
    if member_order == 1 and evidence[0] > 0:
        with pytest.raises(ValueError, match='recorded concept order disagrees'):
            alloc.settle(target)
    else:
        assert alloc.settle(target) == 1


def test_ambiguous_named_promotion_fails_before_minting():
    resolver,store,points = fixture()
    owner = resolver.components.owner
    alloc = owner._concept_allocator
    word = owner.interpret.lookup_word([23], [], form='label')
    ids = []
    for i,point in enumerate(points[:2]):
        occ = write(store, point, ('separate',i))
        key = owner.mtm.open(occ, kind=None, value=point, when=i)
        obj = promote_identity(owner,key)
        owner.interpret.define(word,obj)
        ids.append(obj)
    occ = write(store, points[2], 'named')
    token = owner.mtm.open(occ, kind=None, value=points[2], when=2, reading=word)
    counter, rows = alloc.next_id, len(alloc.parts)
    with pytest.raises(ValueError, match='ambiguous identity promotion') as exc:
        promote_identity(owner, token, word=word)
    assert f'words {[word]}' in str(exc.value) and 'label' in str(exc.value)
    assert f'denotations {sorted(ids)}' in str(exc.value)
    assert (alloc.next_id, len(alloc.parts)) == (counter, rows)
    assert owner.mtm.entries[token].symbol is None




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
            history=copy.deepcopy((store.state_dict(),
                                   store.parts.state_dict(),owner.mtm.state()))
        result = original_boundary(target, **options)
        if first:
            same_state(history,(store.state_dict(),
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
            symbols = set(alloc.placement) - baseline - set(alloc.parts.indices().tolist())
            records = grounded(store,symbols)
            assert len(store) > 0
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
        assert len(starts)==len(reports)==10
        assert reports[0]['minted']
        # Later activity promotions are new concepts (§17.14), not duplicate words.
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
    upper = promote(1,{lower_row:2.,c:1.})
    assert alloc.order_of(lower) == 1 and alloc.reference_orders[upper] == 2
    assert store.parts.find((SYMBOL,lower),(SYMBOL,upper)) is not None
    begin(model)
    assert alloc.order_of(lower) == 1 and alloc.order_of(upper) == 2
    assert observe_candidate(cs,1,{a:2.,b:1.}) is None
    assert {r for r in store.parts.parts((SYMBOL,lower))} == {
        (SYMBOL,cs.concept_id_at_row(a)),(SYMBOL,cs.concept_id_at_row(b))}
    assert not cs.mtm.promote_candidates(cs)
    assert promote(1,{lower_row:2.,c:1.}) == upper
    assert store.parts.find((SYMBOL,lower),(SYMBOL,upper)) is not None
