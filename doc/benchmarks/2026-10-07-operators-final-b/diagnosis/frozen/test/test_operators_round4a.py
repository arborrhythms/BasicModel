"""Round 4a: fixed sentence identity, open-world bilattice, kept prediction."""
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def test_sentence_identity_is_recomputable_and_does_not_consume_global_rng():
    from MeaningCodes import identity_code, sentence_key
    key = sentence_key([b'hello', b'world'])
    before = torch.get_rng_state().clone()
    numpy_before = np.random.get_state()
    code = identity_code(key, 64, 3)
    assert code.sum() == 3 and set(code.tolist()) == {0., 1.}
    assert torch.equal(before, torch.get_rng_state())
    assert all(np.array_equal(a, b) for a, b in zip(numpy_before, np.random.get_state()))
    command = ('from MeaningCodes import *; import json; '
               'print(json.dumps(identity_code(sentence_key([b"hello", b"world"]),64,3).tolist()))')
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]/'bin'))
    other = json.loads(subprocess.check_output([sys.executable, '-c', command], env=env))
    assert other == code.tolist()


def test_content_address_is_independent_of_store_and_persists_with_occurrences():
    from Layers import TernaryTruthStore
    from Meaning import ConceptualMeaning
    stores = [TernaryTruthStore(8, capacity=4) for _ in range(2)]
    keys = []
    for store in stores:
        from Occurrence import sentence_key
        key = sentence_key([b'hello', b'world'])
        row = store.append_meaning(ConceptualMeaning.from_description(torch.ones(8)),
                                   document_key='fixture', sentence_index=1, content_key=key)
        keys.append(key)
        assert store.content_key(row) == keys[-1]
        assert store.rows_for_content(keys[-1]) == (row,)
        store.slots[row].zero_()
        assert store.content_key(row) == keys[-1]
    assert stores[0].occurrence_of(0) == stores[1].occurrence_of(0)
    assert keys[0] == keys[1]
    restored = TernaryTruthStore(8, capacity=4)
    restored.load_state_dict(stores[0].state_dict())
    assert restored.content_key(0) == keys[0]


def test_one_presentation_bootstraps_detached_context_and_exact_gate_certificate():
    configuration = 'XOR_grammar'
    from test_mm_xor import _fresh_model
    from MereologicalCodes import occurrence_memberships
    from MeaningCodes import identity_code, certificate
    model, _, _ = _fresh_model(f'data/{configuration}.xml')
    try:
        with torch.no_grad():
            model.forward(model.inputSpace.prepInput(['hello world','hello there','loving world','loving there']))
        owner = model._concept_owner()
        derived = owner.similarity_codebook.mereology
        assert derived.context_width == 128 and derived.meaning_pairs == 64 and derived.meaning_ones == 3
        assert not derived._context  # snapshot predates this first presentation
        store = owner._closed_clause_store()
        rows = [r for r in range(len(store)) if int(store.rel_type[r]) == store.REL_NONE]
        assert len(rows) == 4
        codes = {r: identity_code(store.content_key(r),64,3) for r in rows}
        context = derived.occurrence_terms()
        postings = occurrence_memberships(owner, store)
        extents, values = [], []
        for word, (n, value) in sorted(context.items()):
            containing = sorted(set(rows) & postings[word])
            weights = 1/(1+(float(store._next_ts)-1-store.timestamp[containing]).clamp_min(0))
            expected = (torch.stack([codes[r] for r in containing])*weights[:,None]).sum(0)/weights.sum()
            torch.testing.assert_close(value[:64],expected)
            assert not value[64:].any() and not value.requires_grad and n == len(containing)
            extents.append({rows.index(r) for r in containing}); values.append(value.numpy())
        report = certificate(np.stack(values),torch.stack(list(codes.values())).numpy(),extents)
        assert report['exact'] and report['distinct_meanings'] == report['words'] == 4
        derived.begin_forward()
        assert len(derived._context) == 4 and not list(derived.parameters())
    finally:
        model.End(); model.symbolSpace.soft_reset()


@pytest.mark.parametrize('name', ['conjunction', 'disjunction', 'intersection', 'union', 'sum'])
def test_composition_preserves_form_kernel_and_uses_bilattice_without_normalization(name):
    from Language import GRAMMAR_LAYER_CLASSES
    from MeaningCodes import compose
    layer = GRAMMAR_LAYER_CLASSES[name]()
    layer.meaning_layout = (3, 7)
    left = torch.tensor([.2,.4,.8, .3,.9,.2,.8])
    right = torch.tensor([.6,.1,.5, .7,.4,.6,.1])
    expected_form = layer.compose(left[:3],right[:3])
    output = layer.compose(left,right)
    torch.testing.assert_close(output[:3],expected_form,rtol=0,atol=0)
    torch.testing.assert_close(output[3:],compose(name,left[3:],right[3:]),rtol=0,atol=0)
    doubled = layer.compose(torch.cat((left[:3],left[3:]*2)),torch.cat((right[:3],right[3:]*2)))
    torch.testing.assert_close(doubled[:3],output[:3],rtol=0,atol=0)
    torch.testing.assert_close(doubled[3:],output[3:]*2,rtol=0,atol=0)


def test_not_and_negative_interpretation_exchange_both_poles():
    from Interpret import activate_code
    from Language import NotLayer
    layer=NotLayer(); layer.meaning_layout=(2,6)
    value=torch.tensor([[.2,.6,.3,.9,.4,.1]])
    flipped=torch.tensor([[.2,.6,.4,.1,.3,.9]])
    torch.testing.assert_close(layer.forward(value),flipped)
    torch.testing.assert_close(layer.reverse(flipped),value)
    torch.testing.assert_close(activate_code(value,torch.tensor([-.5]),2),flipped*.5)
    torch.testing.assert_close(activate_code(value,torch.tensor([.5]),2),value*.5)


def test_small_width_false_membership_certificate_matches_literal_all_pairs():
    from MeaningCodes import certificate, compose, membership
    # Row 2 is spuriously covered by the union of rows 0 and 1, with no
    # collision between individual sentence codes. Word 0 occurs in 0,1.
    codes=torch.tensor([[1.,0.,1.],[0.,1.,1.],[1.,1.,0.]])
    extents=[{0,1},{2},set()]
    means=torch.tensor([[.5,.5,1.,0.,0.,0.],[1.,1.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.]])
    report=certificate(means.numpy(),codes.numpy(),extents)
    errors=dict(conjunction=0,disjunction=0,not_against=0,and_not_against=0,not_for=0,and_not_for=0)
    for a,b in itertools.product(range(3),repeat=2):
        neg=compose('not',means[a]); mixed=compose('conjunction',means[a],compose('not',means[b]))
        for row,code in enumerate(codes):
            for op,truth in [('conjunction',row in extents[a] and row in extents[b]),('disjunction',row in extents[a] or row in extents[b])]:
                errors[op]+=int(bool(membership(compose(op,means[a],means[b])[:3],code)) != truth)
            for name,pole,truth in [('not_against',neg[3:],row in extents[a]),('and_not_against',mixed[3:],row in extents[b]),('not_for',neg[:3],False),('and_not_for',mixed[:3],False)]:
                errors[name]+=int(bool(membership(pole,code)) != truth)
    assert report['errors']==errors and not report['exact']
    assert errors['disjunction']>0 and errors['not_against']>0


def test_predictor_trains_once_on_kept_rows_only_with_detached_targets():
    from Layers import Error
    from SentenceCredit import reader_weights, expectation_costs
    from ObjectiveOwnership import backward_owned
    parameter=torch.nn.Parameter(torch.tensor(1.))
    targets=[torch.tensor([2.,3.,4.],requires_grad=True),torch.tensor([8.,9.,10.],requires_grad=True)]
    registries=[]
    for target in targets:
        registry=Error(row_mask=torch.tensor([True,True,False]))
        registry.error('expectation.roles',(parameter-target.detach()).square(),1.,category='expectation')
        registries.append(registry)
    parts=torch.zeros(3,2,3);parts[1,0,0]=1.
    weights=reader_weights(parts,torch.tensor([True,True,False]),None)
    costs=expectation_costs(registries,weights)
    backward_owned(costs,{'expectation':(parameter,)})
    torch.testing.assert_close(parameter.grad,torch.tensor(-9.)) # targets 2 and 9, never 8 or 3
    assert all(t.grad is None for t in targets)
    optimizer=torch.optim.Adam([parameter],lr=.01);optimizer.step()
    assert optimizer.state[parameter]['step']==1


@pytest.mark.parametrize('configuration', ['XOR_grammar', 'MM_grammar'])
def test_static_certificate_on_each_configured_gate_vocabulary(configuration):
    from MeaningCodes import sentence_key, identity_code, certificate
    from util import XMLConfig
    cfg=XMLConfig(f'data/{configuration}.xml','data/model.xml')
    pairs=(int(cfg.data['ConceptualSpace']['nDim'])-int(cfg.data['PartSpace']['nDim']))//2
    ones=3
    corpus=[s.split() for s in [b'hello world',b'hello there',b'loving world',b'loving there']]
    codes=torch.stack([identity_code(sentence_key(words),pairs,ones) for words in corpus])
    words=sorted({w for row in corpus for w in row})
    extents=[{row for row,terms in enumerate(corpus) if w in terms} for w in words]
    means=torch.stack([torch.cat((codes[sorted(rows)].mean(0),torch.zeros(pairs))) for rows in extents])
    result=certificate(means.numpy(),codes.numpy(),extents)
    assert result['exact'] and result['distinct_meanings']==4


def test_nondeclared_meaning_write_preserves_block_at_structural_face():
    from Language import TenseLayer
    from types import SimpleNamespace
    layer=TenseLayer();layer.meaning_layout=(8,12)
    value=torch.tensor([[.1,.2,.3,.4,.5,.6,.7,.8,.9,.2,.8,.1]])
    output=layer.compose_from_grammar_context((value,),context=SimpleNamespace())
    torch.testing.assert_close(output[:,8:],value[:,8:],rtol=0,atol=0)
    torch.testing.assert_close(output[:,:8],layer.compose(value[:,:8]),rtol=0,atol=0)


def test_live_trial_predictor_has_one_kept_target_step(monkeypatch):
    from test_mm_xor import _fresh_model
    model,_,_=_fresh_model('data/XOR_grammar.xml')
    parameter=torch.nn.Parameter(torch.tensor(1.))
    model.register_parameter('round4a_prediction_probe',parameter)
    groups=model.objective_parameter_groups
    def owned(optimizer):
        result=groups(optimizer)
        result['expectation']=(*result['expectation'],parameter)
        return result
    monkeypatch.setattr(model,'objective_parameter_groups',owned)
    score=model._sentence_path_cost
    targets=[]
    def cost(*args,**kwargs):
        result=score(*args,**kwargs)
        target=parameter.new_tensor([2.,3.,4.,5.] if not targets else [8.,9.,10.,11.])
        targets.append(target)
        model._sentence_cost_registry.error('expectation.kept_probe',
            (parameter-target).square(),1.,category='expectation')
        return result
    monkeypatch.setattr(model,'_sentence_path_cost',cost)
    try:
        optimizer=model.getOptimizer(lr=.01)
        model.runEpoch(optimizer=optimizer,batchSize=4,split='train')
        assert len(targets)==2
        weights=model._sentence_reader_weights
        kept=targets[0]*weights[:,0]+targets[1]*weights[:,1]
        expected=2*(1-kept).mean()
        # Adam's first moment encodes the exact single owned gradient.
        beta=next(g for g in optimizer.param_groups if any(p is parameter for p in g['params']))['betas'][0]
        torch.testing.assert_close(optimizer.state[parameter]['exp_avg'],expected*(1-beta))
        assert optimizer.state[parameter]['step']==1
    finally:
        model.End();model.symbolSpace.soft_reset()


def test_occurrence_membership_recency_def_exclusion_and_snapshot_are_preserved():
    from Layers import TernaryTruthStore
    from Meaning import ConceptualMeaning
    from MereologicalCodes import MereologicalCodes
    from MeaningCodes import identity_code
    store=TernaryTruthStore(20,capacity=8);store.image_form_width=4
    owner=SimpleNamespace(_closed_clause_store=lambda:store,
        _csw_row_of=lambda cid:{11:0,12:1}.get(cid),_definition_index=lambda:None)
    derived=MereologicalCodes(owner,torch.zeros(2,20),percept_width=4,percept_event_width=4)
    for i,kind in enumerate([store.REL_NONE,store.REL_NONE,store.REL_DEF,store.REL_OPERATOR]):
        from Occurrence import sentence_key
        row=store.append_meaning(ConceptualMeaning.from_description(torch.full((20,),float(i))),rel_type=kind,
            content_key=sentence_key([str(i).encode()]), document_key='fixture', sentence_index=i)
        store._append_leaf_terms(row,((0,),(),()),(True,True,True))
    # The second row is witnessed by a reference alone; membership is still one row.
    store._leaf_postings[(0,0)].remove(1);store.refs[1,0]=11
    terms=derived.occurrence_terms();assert set(terms)=={0} and terms[0][0]==2
    codes=torch.stack([identity_code(store.content_key(r),8,derived.meaning_ones) for r in [0,1]])
    assert not torch.equal(codes[0],codes[1])
    weights=1/(1+(float(store._next_ts)-1-store.timestamp[:2]).clamp_min(0))
    expected=(codes*weights[:,None]).sum(0)/weights.sum()
    torch.testing.assert_close(terms[0][1],torch.cat((expected,torch.zeros(8))))
    store.slots[:,0,4:]=99. # composed meanings cannot feed their own bootstrap
    torch.testing.assert_close(derived.occurrence_terms()[0][1],terms[0][1])
