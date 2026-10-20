"""A constituent annotation teaches a pair, not a mandated adjective operator."""
from types import SimpleNamespace as NS

import torch

from GrammarLessons import (constituent_examples, compose_loss, student_target_actions,
                            student_target_probability)


def test_modification_target_accepts_either_operator_and_keeps_reference_choice_free(monkeypatch):
    import GrammarLessons
    rules=tuple(NS(method_name=name) for name in ('lower','intersection','product'))
    layer=NS(r_reduce=3,chooser=NS(WHAT_CONTEXT_DIM=0))
    language=NS(_compose_binary_rules=rules,_compose_unary_rules=(),_tree_layer=lambda _:layer)
    probabilities=torch.tensor([.1,.1,.2,.3,.1,.2],requires_grad=True)
    state=dict(spans=((0,),(1,)),kwargs=dict(reference_data=dict(binary_ops=(0,1,1,2,2))))
    program=NS(leaves=torch.eye(2),student_states=[state],actions=torch.tensor([2]))
    tree=['modification',0,1]
    example=constituent_examples(language,program,tree)[0]
    assert example['target']==(1,2)
    assert student_target_actions(layer,example)==[1,2,3,4]
    torch.testing.assert_close(student_target_probability(layer,example,probabilities),torch.tensor(.7))
    monkeypatch.setattr(GrammarLessons,'score_student_state',lambda *args:probabilities)
    loss=compose_loss(language,[program],[dict(tree=tree,waits=True)])
    loss.backward()
    assert probabilities.grad[0]==probabilities.grad[-1]==0
    assert (probabilities.grad[1:5]<0).all()
    assert program.actions.tolist()==[2]  # No lesson-side program replacement.
    # The enclosing lower still targets its own operation; the bracket only
    # admits modification when the student's two operands match that pair.
    program.leaves=torch.eye(3)
    program.student_states=[dict(state,spans=((0,),(1,))),dict(state,spans=((1,),(2,))),
                            dict(state,spans=((0,),(1,2)))]
    assert [ex['target'] for ex in constituent_examples(language,program,['lower',0,['modification',1,2]])]==[-1,(1,2),0]
    from np_corpus import corpus
    assert all('intersection' not in str(row[2]['tree']) for row in corpus())


def test_identity_carrying_unary_faces_reverse_exactly_and_retire_free_marker_state():
    from Language import LowerLayer, LowerUnaryLayer, DefiniteUnaryLayer, PrepositionLayer, PrepositionUnaryLayer
    from GrammarCheckpoint import align
    # Dyadic inputs make bit-exactness inspectable without picking a seed.
    value=torch.arange(32.).reshape(4,8)/32
    for face,base in ((LowerUnaryLayer,LowerLayer(8,8)),(DefiniteUnaryLayer,LowerLayer(8,8)),
                      (PrepositionUnaryLayer,PrepositionLayer(8,8))):
        unary=face(base)
        assert not list(unary.parameters()) and 'marker' not in unary.state_dict()
        assert torch.equal(unary.marker,torch.zeros(8))
        parent=unary.compose(value)
        assert torch.equal(parent,value) and torch.equal(unary.reverse(parent),value)
        compiled=torch.compile(lambda x:unary.reverse(unary.compose(x)),backend='eager',fullgraph=True)
        assert torch.equal(compiled(value),value)
        host=torch.nn.Module();host.add_module('face',unary);host.add_module('alias',unary)
        old={'face.marker':torch.ones(8),'alias.marker':torch.full((8,),2.)}
        restored,drift,transforms=align(host,old,None)
        assert restored=={} and set(drift)==set(old) and all(value is None for value in transforms.values())


def test_np_gates_behaviors_and_reports_the_kept_operator():
    from np_experiment import summarize_stage, MENU
    assert MENU==('STOP','lower','definite','intersection','product')
    description=dict(family='description',brackets=True,nps=[],duplicate_writes=0,occurrence_collisions=0,
        unexpected_occurrences=1,unlowered_clean=False,question_expected=True,answer_exact=True,
        observed=dict(program=['lower']))
    result=summarize_stage([description],'description')
    assert result['passed'] and set(result['gates'])=={'brackets','answers'}
    assert result['reports']['kept_programs']=={'lower':1}
    assert result['reports']['unlowered']['correct']==0
    assert not summarize_stage([dict(description,answer_exact=False)],'description')['passed']
    missing=dict(description);del missing['answer_exact']
    assert not summarize_stage([missing],'description')['passed']


def test_np_section5a_keeps_behavior_thresholds_and_only_reports_empty_meanings():
    from copy import deepcopy
    from np_experiment import summarize_stage, GATES, REPORTS
    assert GATES==dict(exact_brackets=.9,one_order1_occurrence=.95,
                      duplicates=0,collisions=0,exact_answers=1.)
    assert {'unexpected_occurrences','learned_operation','lowering_stamps',
            'kept_operators','meaning_norms'} <= REPORTS.keys()
    rows=[dict(id=str(i),family='np',brackets=i<18,
        nps=[dict(one_occurrence=i<19,reduction=False)],duplicate_writes=0,
        occurrence_collisions=0,unexpected_occurrences=1,unlowered_clean=None,
        question_expected=True,answer_exact=True,observed=dict(program=['product'],
            closed_meaning_norm=0.,steps=[dict(method='product',meaning_norms=[0.,0.,0.])],
            lowerings=[dict(before=0,after=1)])) for i in range(20)]
    result=summarize_stage(rows,'adjective')
    assert result['passed']
    assert result['gates']['brackets']['correct']==18
    assert result['gates']['occurrences']['correct']==19
    assert 'unexpected' not in result['gates']['occurrences']
    assert result['reports']['unexpected_occurrences']==20
    assert result['reports']['kept_operators']=={'product':20}
    assert result['reports']['reductions']['correct']==0
    assert result['reports']['meaning_norms'][0]['operations']==[
        dict(method='product',meaning_norms=[0.,0.,0.])]
    assert result['reports']['lowering_stamps'][0]['lowerings']==[dict(before=0,after=1)]
    for field,value in [('brackets',False),('answer_exact',False),
                        ('duplicate_writes',1),('occurrence_collisions',1)]:
        failed=deepcopy(rows);failed[0][field]=value
        assert not summarize_stage(failed,'adjective')['passed'],field
    failed=deepcopy(rows);failed[0]['nps'][0]['one_occurrence']=False
    assert not summarize_stage(failed,'adjective')['passed']


def test_np_menu_admits_both_modification_operators_at_the_native_mask():
    from Language import OperationSelectionLayer
    from PartsOfSpeech import formal_arguments
    from test_parts_of_speech import Pair, Unary, rule
    from np_experiment import MENU
    binary=[rule(name) for name in ('lower','definite','intersection','product','verb')]
    unary=[rule('surface_unary',1)]
    arguments=formal_arguments(binary+unary)
    layer=OperationSelectionLayer(d_model=4,ops=[Pair() for _ in binary],unary_ops=[Unary()],reduce_pressure=0.)
    layer.configure_arguments(arguments,binary,unary)
    layer.set_stage_menu(MENU)
    logits=layer(torch.ones(1,2,4),slots=2,occupation=torch.ones(1,2,len(arguments)))[2]['logits']
    assert torch.isfinite(logits).tolist()==[[True,True,True,True,False,False,False,True]]
