"""One sentence surrogate, detached bands, and live compose credit."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import torch
import pytest


def receipt(tmp_path, name, value):
    destination = Path(os.environ.get('ITEM25_RECEIPT', tmp_path))
    destination.mkdir(parents=True, exist_ok=True)
    record = value if isinstance(value, dict) else {'records': value}
    (destination / (name + '.json')).write_text(
        json.dumps({**record, 'torch_version': str(torch.__version__)}, indent=2) + '\n')


def checkpoint_without_random_state(path):
    """Reuse one ambient model's weights without restoring any random stream."""
    saved=torch.load(path,weights_only=False)
    training=saved['training_state']
    for key in ('torch_rng_state','python_rng_state','numpy_rng_state',
                'cuda_rng_state_all','mps_rng_state'):
        training.pop(key,None)
    torch.save(saved,path)
    return saved


def test_strength_has_declared_interval_and_zero_does_not_draw():
    from SentenceTraining import strength
    model = SimpleNamespace(reconstruction_jitter=0.)
    before = torch.get_rng_state().clone()
    assert strength(model, torch.zeros(())).item() == 1.
    assert torch.equal(before, torch.get_rng_state())
    model.reconstruction_jitter = .5
    values = torch.stack([strength(model, torch.zeros(())) for _ in range(256)])
    assert bool(((values >= .5) & (values <= 1.)).all())
    assert bool((values < .6).any()) and bool((values > .9).any())
    for bad in (-.01, 1.01, float('nan')):
        model.reconstruction_jitter = bad
        with pytest.raises(ValueError, match='reconstructionJitter'):
            strength(model, torch.zeros(()))


def test_compose_credit_reaches_live_semantics_and_its_three_writers_are_attributable(tmp_path):
    from Language import OperationSelectionLayer
    from test_compose_operations import Add, Negate
    from SentenceCredit import score_function
    from SentenceTraining import surrogate
    reports = []
    for separate in (True, False):
        theta = torch.nn.Parameter(torch.tensor([[[.2], [.3], [.7]]]))
        reader = torch.nn.Parameter(torch.tensor(.8))
        chooser = OperationSelectionLayer(d_model=1, ops=[Add()], unary_ops=[Negate()])
        chooser.separate_chooser = separate
        with torch.no_grad():
            chooser.reduce_anchor.fill_(.4)
            chooser.apply_anchor.fill_(.2)
            chooser.stop_anchor.fill_(.1)
        _, value, route = chooser(theta, slots=1)
        probability = route['probabilities'][:, 0, None]
        credit, _ = score_function(probability, torch.ones_like(probability),
            torch.ones_like(probability, dtype=torch.bool), probability.new_tensor([[.8,.3]]))
        costs = dict(reconstruction=(value-.4).square().sum(),
                     output=(reader*value-.1).square().sum(), chooser=credit.sum())
        gradients = {name:torch.autograd.grad(cost, theta, retain_graph=True, allow_unused=True)[0]
                     for name,cost in costs.items()}
        assert gradients['reconstruction'].abs().sum() > 0
        assert gradients['output'].abs().sum() > 0
        if separate:
            assert gradients['chooser'] is None
        else:
            assert gradients['chooser'].abs().sum() > 0
            expected = .6*gradients['reconstruction'] + gradients['output'] + gradients['chooser']
            surrogate(costs, theta.new_tensor(.6)).backward()
            torch.testing.assert_close(theta.grad, expected)
            assert reader.grad is not None and chooser.reduce_anchor.grad is not None
        reports.append(dict(separate_chooser=separate,
            gradients={name:None if value is None else value.tolist() for name,value in gradients.items()}))
    receipt(tmp_path, 'semantic-writers', reports)


def test_bands_and_intuition_receive_exactly_their_own_terms_in_one_surrogate(tmp_path):
    from CandidateAttention import CandidateAttention
    from Expectation import ActivationPredictor, recombine_percept
    from Intuition import Idea, Term, seed, selected
    from ConceptSpread import Prediction, loss as predictor_loss
    from QueryWork import QueryWorkBudget
    from SentenceTraining import surrogate
    from test_thought_infer import setup, idea
    model, registry, store, refs = setup()
    owner = model._concept_owner()
    semantic = torch.nn.Parameter(torch.tensor([[.2,.4]]))
    target = torch.full((1,2,2), .8, requires_grad=True)
    predictor = ActivationPredictor(2)
    keys = semantic[:,None].expand(-1,2,-1)
    context = semantic.repeat(1,2)
    knowing = semantic[:,None].expand(-1,2,-1)
    predicted = predictor(keys,context,knowing).sigmoid()
    frozen = Prediction(keys,context,knowing,predicted.detach(),torch.arange(2),
        torch.tensor([True]),target,torch.ones(1,2,dtype=torch.bool))
    concept_model = SimpleNamespace(_concept_prediction=frozen,
        _concept_owner=lambda:SimpleNamespace(concept_predictor=predictor))
    attention = CandidateAttention(2,2,hidden_width=4)
    attended = attention(semantic,keys).sigmoid()
    consumed = recombine_percept(semantic, predicted[:,0], attended[:,0,None]).content
    root = Idea(observed=torch.full((owner.outputShape[-1],), .4))
    term = seed(owner,Term(('sym',909),root,root,(.8,.2),idea(registry,refs[1]),
        (root,),(1.,)*4),QueryWorkBudget(256))
    intuition = selected(owner,term)[0].roles
    costs = dict(reconstruction=(consumed-.7).square().sum(),
        output=(consumed+intuition.mean()-.1).square().sum(),
        expectation=predictor_loss(concept_model).sum(),
        attention=(attended-.9).square().sum())
    bands = {'expectation':tuple(predictor.parameters()), 'attention':tuple(attention.parameters()),
             'output':tuple(owner.intuition.parameters())}
    expected = {}
    for name, parameters in bands.items():
        expected[name] = torch.autograd.grad(costs[name],parameters,retain_graph=True,allow_unused=True)
        for other, cost in costs.items():
            if other != name:
                assert all(g is None for g in torch.autograd.grad(cost,parameters,retain_graph=True,allow_unused=True))
    surrogate(costs,semantic.new_tensor(.6)).backward()
    for name, parameters in bands.items():
        for parameter, gradient in zip(parameters,expected[name]):
            if gradient is None:
                assert parameter.grad is None
            elif gradient.is_sparse:
                torch.testing.assert_close(parameter.grad.coalesce().indices(),gradient.coalesce().indices())
                torch.testing.assert_close(parameter.grad.coalesce().values(),gradient.coalesce().values())
            else:
                torch.testing.assert_close(parameter.grad,gradient)
    assert target.grad is None and semantic.grad is not None
    receipt(tmp_path,'band-gradient-equality',dict(expectation=True,attention=True,
        intuition_output_only=True,target_gradient=False))


def native_config(tmp_path, separate=False, jitter=0.):
    import xml.etree.ElementTree as ET
    from util import init_config
    import Language
    tree=ET.parse('data/MM_corpus_orders.xml')
    architecture=tree.getroot().find('architecture')
    for name,value in (('separateChooser',str(separate).lower()),('reconstructionJitter',str(jitter))):
        element=architecture.find(name)
        if element is None:element=ET.SubElement(architecture,name)
        element.text=value
    path=tmp_path/'one-backward.xml';tree.write(path)
    init_config(str(path),defaults_path=str(Path('data/model.xml').resolve()))
    Language.TheGrammar._configured=False
    return path


def test_native_sentence_counts_one_backward_and_step_and_preserves_band_gradients(tmp_path,monkeypatch,eager_reading):
    from Models import BaseModel
    from data import TheData
    from MathChainTraining import present
    from Corpus import admit_training_forms
    from corpus_lessons import stage_documents
    TheData.processLM({split:dict(text=['.'] if split=='train' else [],label=[])
                      for split in ('train','validation','test')})
    model=BaseModel.from_config(str(native_config(tmp_path)),data=TheData)[0]
    try:
        stage=dict(documents=[dict(document=0,records=[dict(key='one-backward',
            sentences=['This is Alpha.','What is this?'],question=1,answer='Alpha')])])
        texts,targets=stage_documents(TheData,stage)
        admit_training_forms(model,texts,targets)
        optimizer=model.getOptimizer(lr=.001)
        owner=model.percept_expectation
        assert not hasattr(owner,'optimizer')
        assert {id(p) for p in owner.parameters()} <= {id(p) for g in optimizer.param_groups for p in g['params']}
        concept=model._concept_owner()
        def storage():
            return {label+'.'+name:dict(pointer=value.data_ptr(),shape=list(value.shape),
                bytes=value.numel()*value.element_size())
                for label,module in (('percept',owner),('concept',concept.concept_predictor),('intuition',concept.intuition))
                for name,value in (*module.named_parameters(),*module.named_buffers())}
        allocated=storage()
        from corpus_capacity import optimizer_parameter_bytes
        state_before=optimizer_parameter_bytes(optimizer,owner.parameters())
        backward=torch.Tensor.backward;step=optimizer.step
        calls=[];steps=[];audits=[]
        def counted_backward(value,*args,**kwargs):
            calls.append(int(model._open_sentence_slot))
            return backward(value,*args,**kwargs)
        def counted_step(*args,**kwargs):
            steps.append(int(model._open_sentence_slot))
            return step(*args,**kwargs)
        monkeypatch.setattr(torch.Tensor,'backward',counted_backward)
        monkeypatch.setattr(optimizer,'step',counted_step)
        train=model._backward_training_loss
        def checked(*args,**kwargs):
            assert model._sentence_backward
            costs=model._single_sentence_objectives
            groups=model.objective_parameter_groups(optimizer)
            from GradientProjection import sides, project, metric
            objectives = sides(costs, model._sentence_reconstruction_strength)
            expected={}
            for name in ('expectation','attention'):
                parameters=groups[name]
                gradients = [tuple(None for _ in parameters) if cost is None else
                    torch.autograd.grad(cost, parameters, retain_graph=True, allow_unused=True)
                    for cost in objectives.values()]
                expected[name] = tuple(a if r is None else r if a is None else
                    r + project(r, a, metric(optimizer, p, r))[0]
                    for p, r, a in zip(parameters, *gradients))
            result=train(*args,**kwargs)
            for name,gradients in expected.items():
                for parameter,gradient in zip(groups[name],gradients):
                    if gradient is None:
                        assert parameter.grad is None or not parameter.grad.count_nonzero()
                    elif gradient.is_sparse:
                        torch.testing.assert_close(parameter.grad.coalesce().values(),gradient.coalesce().values())
                        torch.testing.assert_close(parameter.grad.coalesce().indices(),gradient.coalesce().indices())
                    else:torch.testing.assert_close(parameter.grad,gradient,atol=1e-6,rtol=1e-5)
            audits.append({name:sum(g is not None for g in values) for name,values in expected.items()})
            return result
        monkeypatch.setattr(model,'_backward_training_loss',checked)
        result=present(model,split='train',optimizer=optimizer,batch_size=1)
        assert result['sentences']==2 and len(calls)==0 and len(steps)==len(audits)==2
        assert model._single_backward_calls == 1  # counter resets for each presented batch
        assert storage()==allocated
        receipt(tmp_path,'native-single-backward',dict(sentences=2,backward_calls=calls,
            optimizer_steps=steps,band_gradient_equality=audits,
            percept_arrivals=int(owner.updates),percept_private_optimizer=False,
            fixed_storage=allocated,storage_unchanged=True,
            percept_shared_optimizer_bytes_before=state_before,
            percept_shared_optimizer_bytes_after=optimizer_parameter_bytes(optimizer,owner.parameters())))
    finally:
        model.End();model.symbolSpace.soft_reset();torch._dynamo.reset()


def test_thought_and_generate_features_stay_detached_when_compose_is_live():
    from Language import OperationSelectionLayer, GeneratePolicy
    from Meaning import ConceptualMeaning
    semantic=torch.nn.Parameter(torch.tensor([.2,.3,.7,.6]))
    layer=OperationSelectionLayer(d_model=4,chooser='mlp')
    layer.separate_chooser=False
    active=ConceptualMeaning.from_description(semantic)
    request=ConceptualMeaning.from_description(semantic*.7)
    logits=layer.thought_logits(active,(request,None))
    (-logits.log_softmax(-1)[0,0]).backward()
    assert semantic.grad is None
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in layer.parameters())
    policy=GeneratePolicy(4,2)
    features=torch.ones(1,2,8,requires_grad=True)
    policy(semantic[None],features).square().sum().backward()
    assert semantic.grad is None and features.grad is None
    assert policy.weight.grad.abs().sum()>0


@pytest.mark.parametrize('width',[0,1])
def test_numeric_sentence_target_matches_data_authority_before_the_update(width):
    from Models import BasicModel
    from Layers import Error
    from What import What, WhatAnswer
    from test_supplied_answer_training import _sentence
    model,state,parameters=_sentence(True)
    model.separate_chooser=False
    model._sentence_supplied_answers=torch.zeros(2,width)
    model._sentence_answer_questions=tuple(What.supervised(i) for i in range(2))
    model.inputSpace.data.what=lambda question:WhatAnswer(question,float(question.where+1))
    warnings=[]
    model._warn_zeroed_channel=lambda *args:warnings.append(args)
    registry=Error(row_mask=torch.ones(2,dtype=torch.bool))
    cost=BasicModel._sentence_numeric_error(model,state,0,torch.ones(2,dtype=torch.bool),
        dict(record=None),registry=registry)
    torch.testing.assert_close(model._sentence_answer_raw_cost,torch.tensor([25.,64.]))
    assert cost is not None and cost.requires_grad
    gradients=torch.autograd.grad(registry.total().sum(),parameters,allow_unused=True)
    assert all(gradient is not None and gradient.abs().sum()>0 for gradient in gradients)
    assert bool(warnings)==(width==0)


def test_gradient_receipt_uses_sentence_variance_and_epoch_vectors_without_updating():
    from SentenceTraining import begin_sentence,measure,finish_sentence,gradient_receipt
    parameter=torch.nn.Parameter(torch.tensor(.5))
    optimizer=torch.optim.SGD([parameter],lr=.1)
    model=SimpleNamespace(_measure_sentence_gradients=True,conceptualSpaces=(),
        objective_parameter_groups=lambda optimizer:dict(reconstruction=(parameter,),
            output=(),expectation=(),attention=()))
    parameter.grad=torch.tensor(.25)
    for coefficient in (4.,-2.):
        begin_sentence(model)
        measure(model,dict(reconstruction=parameter.square(),output=-parameter,
            chooser=coefficient*parameter),optimizer)
        finish_sentence(model)
        assert parameter.item()==.5 and parameter.grad.item()==.25
    report=gradient_receipt(model,clear=True)
    group=report['chooser_variance']['owner.reconstruction']
    assert group==dict(samples=2,mean_square_norm=10.,mean_gradient_square_norm=1.,variance=9.)
    agreement=report['dissonance']['owner.reconstruction']
    assert agreement['reconstruction_output_cosine']==pytest.approx(-1.)
    assert agreement['reconstruction_chooser_cosine']==pytest.approx(1.)
    assert model._gradient_diagnostic_seconds>=0
    assert not model._epoch_objective_gradients and not model._chooser_gradient_moments
