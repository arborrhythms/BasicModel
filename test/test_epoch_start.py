"""Thinking §40. Preserve given knowledge and start a new history at a training boundary."""
from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
from part_fixtures import stored_pair
import torch

from FamiliarTier import promote_identity
from Memory import OCCURRENCE, RELATION, SYMBOL
from Meaning import ConceptualMeaning
from test_presentation_memory import begin, fixture, model_for, write




def _supply_relation(store, point, left, right, *, trust=.9):
    from ClauseRow import Clause, Composition, predicate_identity
    term = Composition(point, (Composition(point, reference=left),
                               Composition(point, reference=right)), operation='implies')
    clause = Clause(ConceptualMeaning.from_description(point), relation='implies',
                    refs=(left, predicate_identity('implies'), right), composition=term)
    with store.clause_assertions(trust=trust, ):
        return store.write_clause(clause)[1]






























def test_provisioned_symbol_relation_keeps_its_own_evidence_across_epochs():
    from test_clause_storage import clause_store, part_clause
    store, _ = clause_store()
    model = model_for(store._concept_owner(), store)
    begin(model)
    with store.clause_assertions(trust=.7):
        row = store.write_clause(part_clause())
    assert row[0] == 'sym'
    begin(model)
    assert store.row(row)['evidence'] == pytest.approx((.7, 0.))
    with store.clause_assertions(trust=-.2):
        assert store.write_clause(part_clause()) == row
    begin(model)
    assert store.row(row)['evidence'] == pytest.approx((.7, .2))







def test_other_order_active_definition_does_not_hide_remembered_identity():
    from order_fixtures import force_order
    resolver,store,points=fixture()
    owner=resolver.components.owner
    model=model_for(owner,store)
    begin(model)
    word=owner.interpret.lookup_word([23],[],form='label')
    first=write(store,points[0],'first')
    token=owner.mtm.open(first,kind=None,value=points[0],when=0,reading=word)
    known=promote_identity(owner,token,word=word)
    begin(model)
    other=owner.new_concept()
    force_order(owner,other,2)
    owner._csw_concept_row(2,other)
    assert owner.interpret.define(word,other) is not None
    now=write(store,points[1],'again')
    token=owner.mtm.open(now,kind=None,value=points[1],when=1,reading=word)
    assert promote_identity(owner,token,word=word)==known
    assert owner.interpret.denotations(word)==tuple(sorted((known,other)))






def test_unstarted_checkpoint_preserves_setup_until_its_first_epoch_boundary(tmp_path):
    import copy
    from test_epoch_history import same_state
    from test_ltm_consolidation import _make_model, _SERIAL_CONFIG
    model = _make_model(_SERIAL_CONFIG)
    restored = None
    try:
        owner, store = model._concept_owner(), model.symbolSpace.ltm_store
        point = owner.similarity_codebook.W[0].detach().clone()
        symbol = owner.new_concept()
        setup = write(store, point, 'intentional-setup')
        relation = store.parts.add((OCCURRENCE, setup[-1]), (SYMBOL, symbol), truth_plus=.7, truth_minus=.2)
        assert tuple(map(float, store.parts.evidence(relation))) == pytest.approx((.7, .2))
        owner.mtm.open(setup, kind=None, value=point, when=0)
        history = copy.deepcopy((store.state_dict(), store.parts.state_dict(), owner.mtm.state()))
        checkpoint = tmp_path / 'unstarted.ckpt'
        model.save_weights(str(checkpoint))
        model.End(); model.symbolSpace.soft_reset()
        from Layers import GrammarLayer
        GrammarLayer.set_chart_authority(None)
        model = None
        restored = _make_model(_SERIAL_CONFIG)
        assert restored.load_weights(str(checkpoint), require_match=True)
        owner, store = restored._concept_owner(), restored.symbolSpace.ltm_store
        same_state(history, (store.state_dict(), store.parts.state_dict(), owner.mtm.state()))
        begin(restored)
        same_state(history, (store.state_dict(), store.parts.state_dict(), owner.mtm.state()))
        experienced = write(store, point, 'first-epoch')
        during_epoch = store.parts.add((OCCURRENCE, experienced[-1]), (SYMBOL, symbol), truth_plus=1.)
        owner.mtm.open(experienced, kind=None, value=point, when=1)
        begin(restored)
        assert store.location(experienced) is not None and store.parts.contains(during_epoch)
        assert store.location(setup) is not None and store.parts.contains(relation)
    finally:
        for current in (model, restored):
            if current is not None:
                current.End(); current.symbolSpace.soft_reset()
