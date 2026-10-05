"""§15 ports of the interval, reader and inverse ownership contracts."""
from types import SimpleNamespace
import torch
import pytest
from torch import nn
from test_review13_subspaces import fixture, word


def towers():
    from Spaces import Codebook
    cs, cb, ps, alloc = fixture()
    ws = Codebook()
    ws.W = nn.Parameter(torch.tensor([[.7,.8,.9], [.6,.7,.8]]))
    cs._model.wholeSpace = SimpleNamespace(subspace=SimpleNamespace(what=ws))
    return cs, cb, ps, ws, alloc


def test_net_evidence_interval_and_detached_native_sources():
    cs, cb, ps, ws, alloc = towers()
    row = word(cs, [(4,.75),(4,-.25),(7,.25)])
    cs.add_concept_feature(row, 'ws', 0, .5)
    cs.add_concept_feature(row, 'ws', 0, -.25)
    lower = torch.maximum(ps.W[4], ps.W[7])
    upper = 1-.25*(1-ws.W[0])
    expected = lower  # wholes bound the form; they do not enter it
    result = cb.lookup_rows(row)
    torch.testing.assert_close(result[:3], expected)
    assert not result.requires_grad
    assert not result[3:].any()
    assert ps.W.requires_grad and ws.W.requires_grad and alloc.layer().features.values.requires_grad


def test_against_alone_is_not_negative_form_and_no_wholes_means_lower():
    cs, cb, ps, ws, alloc = towers()
    row = word(cs, [(4,.5),(7,-.5)])
    torch.testing.assert_close(cb.lookup_rows(row)[:3], ps.W[4])


def test_room_projection_reports_before_after_and_moves_extremal_rows():
    cs, cb, ps, ws, alloc = towers()
    row = word(cs, [(7,1.)])
    cs.add_concept_feature(row, 'ws', 1, 1.)
    before_ps, before_ws = ps.W.detach().clone(), ws.W.detach().clone()
    report = cb.mereology.project_room(margin=.1)
    assert report['before']['count'] == 1
    assert abs(report['before']['largest']-.3) < 1e-6
    assert report['after']['largest'] < 1e-6
    torch.testing.assert_close(ps.W, before_ps, rtol=0, atol=0)
    torch.testing.assert_close(ws.W[1,0], before_ws[1,0]+.3)
    torch.testing.assert_close(ps.W[4], before_ps[4])
    assert ps.W.grad is None and ws.W.grad is None


def pair(parent, bank):
    from Language import LanguageSpace
    class Product:
        @staticmethod
        def compose(x,y):
            return x*y
    B = len(parent)
    return LanguageSpace._bounded_binary_reconstruction(Product(), parent,
        torch.zeros_like(parent), torch.zeros(B,dtype=torch.bool),
        torch.zeros(B,dtype=torch.bool), bank,
        torch.ones(bank.shape[:2],dtype=torch.bool), bank.shape[1])


def test_pair_is_hard_detached_and_relative_under_product_scaling():
    bank = torch.tensor([[[.2,.4],[.7,.3]]], requires_grad=True)
    parent = torch.tensor([[.11,.13]], requires_grad=True)
    left,right,ok = pair(parent,bank)
    assert ok.all() and not left.requires_grad and not right.requires_grad
    scaled = (parent.detach()*.01).requires_grad_()
    l,r,_ = pair(scaled,bank.detach()*.1)
    torch.testing.assert_close(l, left*.1)
    torch.testing.assert_close(r, right*.1)


def test_byte_scorer_bank_is_constant_leaf_is_live():
    from Models import BasicModel
    bank = torch.tensor([[[.2,.6],[.7,.3]]], requires_grad=True)
    leaf = torch.tensor([[.3,.5]], requires_grad=True)
    owner = SimpleNamespace(_BYTE_ASSIGNMENT_TAU=.1, _readback_percept_width=2)
    cost = BasicModel._byte_word_cost(owner, leaf, torch.tensor(0), bank,
        torch.tensor([[[65,0],[66,0]]]), torch.ones(1,2,2,dtype=torch.bool),
        torch.tensor([[[65,0]]]), torch.ones(1,1,2,dtype=torch.bool),True)
    a,b = torch.autograd.grad(cost.sum(),(leaf,bank),allow_unused=True)
    assert a.norm()>0 and b is None


def test_sentence_path_has_no_native_prototype_or_evidence_gradient(monkeypatch):
    import Models
    from test_mm_xor import _fresh_model
    model, _, data = _fresh_model('data/XOR_grammar.xml')
    original = Models.BaseModel._backward_training_loss
    observations = []
    def backward(self,total,*args,**kwargs):
        if getattr(self, '_sentence_backward', False):
            parameters = [self.perceptualSpace.subspace.what.W]
            parameters += [cs._concept_allocator.layer().features.values for cs in self.conceptualSpaces]
            parameters = list({id(p):p for p in parameters if torch.is_tensor(p) and p.requires_grad}.values())
            pullback = getattr(self, '_sentence_pullback', None)
            grads = (pullback.gradients(total,parameters) if pullback is not None else
                     torch.autograd.grad(total, parameters, retain_graph=True, allow_unused=True))
            observations.append([0. if g is None else float(g.detach().abs().max()) for g in grads])
        return original(self,total,*args,**kwargs)
    monkeypatch.setattr(Models.BaseModel,'_backward_training_loss',backward)
    optimizer = model.getOptimizer(lr=.01)
    try:
        raw,target = next(iter(data.data_loader(split='train',num_streams=4)))
        batch = model.inputSpace.prepInput(raw), model.outputSpace.prepOutput(target)
        model.runBatch(train=True,optimizer=optimizer,batchSize=4,split='train',batch_override=batch)
        assert observations and all(not any(row) for row in observations), observations
    finally:
        model.End(); model.symbolSpace.soft_reset(); torch._dynamo.reset()


def test_raw_root_reader_is_affine_and_cuts_the_root_gradient():
    from SentenceUnderstanding import PrimedSymbols, SentenceUnderstanding, SentenceRecordReader
    root = torch.tensor([[3.,4.]],requires_grad=True)
    end = torch.cat((root[:,None],root.new_zeros(1,2,2)),1)
    bank = PrimedSymbols(torch.tensor([[0]]),root[:,None],torch.ones(1,1),
        torch.ones(1,1,dtype=torch.bool),torch.tensor([[[65]]]),torch.ones(1,1,1,dtype=torch.bool))
    record = SentenceUnderstanding(root,end,torch.ones(1,dtype=torch.long),end.flatten(1)[:,None],
        torch.ones(1,1,dtype=torch.long),root[:,None],torch.zeros(1,1,dtype=torch.long),
        torch.ones(1,1,dtype=torch.bool),bank,torch.tensor(0))
    features = record.reader_features()
    torch.testing.assert_close(features[:,:2], torch.tensor([[3.,4.]]))
    torch.testing.assert_close(features[:,2:4],features[:,:2])
    reader = SentenceRecordReader(2,1)
    assert len(list(reader.parameters())) == 1
    reader(record).sum().backward()
    assert root.grad is None and reader.weight.grad is not None


def test_zero_parent_relative_search_remains_finite():
    parent = torch.zeros(1,2,requires_grad=True)
    bank = torch.tensor([[[.2,.4],[.7,.3]]],requires_grad=True)
    left,right,_ = pair(parent,bank)
    assert torch.isfinite(left).all() and torch.isfinite(right).all()
    assert not left.requires_grad and not right.requires_grad


def test_review16_measurement_wiring_on_one_ordinary_batch(tmp_path):
    import json, os, sys
    from pathlib import Path
    from bounded_tests import run_guarded, GIB
    root=Path(__file__).resolve().parents[1]
    receipt=root/'doc/benchmarks/2026-10-03-operators-attention'
    env=os.environ.copy()
    env.update(PYTEST_PLUGINS='review16_gate_observer', MODEL_COMPILE='none',
        OWNERSHIP_OBSERVER_OUTPUT=str(tmp_path/'ownership'),
        ITEM7_XOR_MEASUREMENTS=str(tmp_path/'observations.jsonl'), ITEM7_XOR_GATE='5',
        REVIEW16_REPORTS=str(tmp_path/'reports.jsonl'),
        PYTHONPATH=os.pathsep.join(map(str,(receipt,root/'bin',root/'test'))))
    result=run_guarded([sys.executable,'-m','pytest','-q',str(receipt/'review16_observer_probe.py')],
        cwd=root,env=env,log_path=tmp_path/'observer.log',memory_bytes=8*GIB,timeout=1800)
    assert result['exit_code']==0,(tmp_path/'observer.log').read_text()


def test_repeated_part_address_does_not_multiply_its_11b_evidence():
    cs,cb,ps,_,_=towers()
    row=word(cs,[((4,4,7),.5)])
    torch.testing.assert_close(cb.lookup_rows(row)[:3], ps.W[[4,7]].amax(0))


def test_zero_parent_relative_search_with_large_valid_codes_stays_finite():
    parent=torch.zeros(1,2,requires_grad=True)
    left,right,_=pair(parent,torch.tensor([[[.7,.8],[.8,.9]]]))
    assert torch.isfinite(left).all() and torch.isfinite(right).all()
    assert not left.requires_grad and not right.requires_grad


def test_room_report_includes_fixed_missing_whole_boundary():
    cs,cb,ps,_,_=towers()
    row=word(cs,[(4,1.)])
    report=cb.mereology.project_room(.3)
    assert report['before']['count']==1
    assert abs(report['before']['largest']-.1)<1e-6
    assert abs(report['after']['largest']-.1)<1e-6
    torch.testing.assert_close(ps.W[4,2],torch.tensor(.8))


@pytest.mark.parametrize('advantage', [-.5, 0., .5])
def test_compose_score_function_moves_toward_the_cheaper_departure(advantage):
    from types import SimpleNamespace
    from Models import BasicModel
    from Layers import Error
    from ObjectiveOwnership import registry_costs, backward_owned
    logits = torch.nn.Parameter(torch.tensor([[.2, -.1, .4]]))
    unrelated = torch.nn.Parameter(torch.tensor([.5]))
    actions = torch.tensor([[1, 0]])
    logp = logits.log_softmax(-1).gather(1, actions)
    path = [None, [None] * 28]
    path[1][18], path[1][27] = actions, logp
    registry = Error(row_mask=torch.tensor([True]))
    model = SimpleNamespace(_compose_forced_slots=torch.tensor([[True, False]]),
                            _sentence_cost_registry=registry)
    optimizer = torch.optim.SGD([logits, unrelated], lr=.1)
    before = logits.detach().clone()
    compared = torch.tensor([[1., 1. + advantage]], requires_grad=True)
    loss = BasicModel._compose_score_function_loss(model, path, compared)
    costs = registry_costs(registry)
    backward_owned(costs, {'reconstruction': (logits, unrelated)})
    if advantage:
        probability = before.softmax(-1)
        expected = -probability * probability[:, 1:2]
        expected[:, 1] += probability[:, 1]
        torch.testing.assert_close(logits.grad, advantage * expected)
        doubled = BasicModel._compose_score_function_loss(
            model, path, torch.tensor([[1., 1. + 2 * advantage]]))
        twice = torch.autograd.grad(doubled, logits)[0]
        torch.testing.assert_close(twice, 2 * logits.grad)
    optimizer.step()
    if advantage:
        assert set(costs) == {'reconstruction'}
        assert (logits.softmax(-1)[0, 1] - before.softmax(-1)[0, 1]) * advantage < 0
        assert not torch.equal(logits, before)
    else:
        assert not costs and logits.grad is None
        assert not loss.requires_grad
        torch.testing.assert_close(logits, before, rtol=0, atol=0)
    assert unrelated.grad is None and unrelated.item() == .5
    assert compared.grad is None
