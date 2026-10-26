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
    from Layers import LTM0
    from Meaning import ConceptualMeaning
    stores = [LTM0(8, capacity=4) for _ in range(2)]
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
    restored = LTM0(8, capacity=4)
    restored.load_state_dict(stores[0].state_dict())
    assert restored.content_key(0) == keys[0]


def test_one_presentation_updates_detached_context_at_the_close():
    from test_mm_xor import _fresh_model
    model, _, _ = _fresh_model('data/XOR_grammar.xml')
    try:
        with torch.no_grad():
            model.forward(model.inputSpace.prepInput(['hello world','hello there','loving world','loving there']))
        derived = model._concept_owner().similarity_codebook.mereology
        assert derived.context_width == 128 and derived.meaning_pairs == 64 and derived.meaning_ones == 3
        context = derived.context_audit()
        assert len(context) == 4
        for record in context:
            assert record['placement_witnesses'] == 2 and record['direction_norm'] == pytest.approx(1.)
            assert not derived.placement.requires_grad
        derived.begin_forward()
        assert len(derived.context_audit()) == 4 and not list(derived.parameters())
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
    expected_form = (left[:3] + right[:3]) / 2
    output = layer.compose(left,right)
    torch.testing.assert_close(output[:3],expected_form,rtol=0,atol=0)
    torch.testing.assert_close(output[3:],compose(name,left[3:],right[3:]),rtol=0,atol=0)
    attenuated = layer.compose(torch.cat((left[:3],left[3:]*.5)),torch.cat((right[:3],right[3:]*.5)))
    torch.testing.assert_close(attenuated[:3],output[:3],rtol=0,atol=0)
    torch.testing.assert_close(attenuated[3:],output[3:]*.5,rtol=0,atol=0)


def test_not_exchanges_poles_and_interpretation_keeps_the_pair():
    from Interpret import activate_code
    from Language import NotLayer
    layer=NotLayer(); layer.meaning_layout=(2,6)
    value=torch.tensor([[.2,.6,.3,.9,.4,.1]])
    flipped=torch.tensor([[.2,.6,.4,.1,.3,.9]])
    torch.testing.assert_close(layer.forward(value),flipped)
    torch.testing.assert_close(layer.reverse(flipped),value)
    expected = torch.tensor([[.1,.3,0.,0.,.15,.45]])
    torch.testing.assert_close(activate_code(value,torch.tensor([.5]),2,
        evidence=torch.tensor([[0.,.5]])),expected)
    expected = torch.tensor([[.1,.3,.15,.45,.3,.9]])
    torch.testing.assert_close(activate_code(value,torch.tensor([.5]),2,
        evidence=torch.tensor([[.5,1.]])),expected)


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
