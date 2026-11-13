"""Frozen study identities and honest paired comparisons; no training or generation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
STUDIES=ROOT/'data/eval/studies'


def protocol():
    return json.loads((STUDIES/'protocol.json').read_text())


def test_studies_are_frozen_disjoint_and_have_exactly_256_causal_tasks():
    from eval_studies import load_tasks
    manifest=json.loads((STUDIES/'manifest.json').read_text())
    assert hashlib.sha256((STUDIES/'recipe.json').read_bytes()).hexdigest()==manifest['recipe_sha256']
    train=load_tasks(STUDIES/'manifest.json','training')
    causal=load_tasks(STUDIES/'manifest.json','causal-heldout')
    heldout=load_tasks(STUDIES/'manifest.json','learning-heldout')
    assert len(causal)==256 and len({row['key'] for row in causal})==256
    assert set(row['key'] for row in causal)<=set(row['key'] for row in heldout)
    first=lambda rows:{' '.join(row['sentences'][0].lower().split()) for row in rows}
    assert first(train).isdisjoint(first(heldout))
    assert len(set(manifest['renaming'].values()))==len(manifest['renaming'])
    assert manifest['learning_seeds']==protocol()['seeds']==[0,1,2]


def test_controls_preserve_task_identity_and_transform_answers_with_vocabulary():
    from eval_studies import load_tasks,documents,evaluation_documents
    from collections import Counter
    import re
    rows=load_tasks(STUDIES/'manifest.json','causal-heldout')
    mapping=json.loads((STUDIES/'manifest.json').read_text())['renaming']
    for row,doc in zip(rows,documents(rows,'renamed_vocabulary')):
        assert doc['document']==row['key']
        expected=None if row['answer'] is None else re.sub(r'\w+',lambda m:mapping[m.group(0)],row['answer'])
        assert doc['records'][0]['answer']==expected
        for original,shuffled in zip(row['sentences'],row['controls']['shuffled_order']):
            assert Counter(re.findall(r'\w+',original))==Counter(re.findall(r'\w+',shuffled))
    with pytest.raises(ValueError,match='unsupported native'):
        documents(rows,'direct_answer')
    for control in ('shuffled_order','shuffled_determiners'):
        assert evaluation_documents(rows,control)==documents(rows,'ordinary')
    assert evaluation_documents(rows,'renamed_vocabulary')==documents(rows,'renamed_vocabulary')


def test_unavailable_undeclared_and_unmatched_comparisons_cannot_pass():
    from eval_studies import compare_gate,causal_readiness
    spec=protocol()
    treatment=dict(status='complete',task_keys=['a'],exposure=4,initialization_sha256='a',metrics=dict(answer_accuracy=.75))
    control=dict(treatment,initialization_sha256='b',metrics=dict(answer_accuracy=.25))
    assert compare_gate(spec,'anaphora',treatment,control)['status']=='undeclared'
    assert compare_gate(spec,'anaphora',treatment,dict(control,exposure=3))['status']=='unmatched'
    assert compare_gate(spec,'anaphora',treatment,dict(control,status='unavailable'))['status']=='unavailable'
    with pytest.raises(ValueError,match='task identities'):
        compare_gate(spec,'anaphora',treatment,dict(control,task_keys=['b']))
    spec['thresholds']={'anaphora':dict(direction='higher',minimum_difference=.6)}
    assert not compare_gate(spec,'anaphora',treatment,control)['passed']
    arms={name:dict(treatment) for name in spec['causal_arms']}
    assert causal_readiness(spec,arms,{name:{} for name in arms})['status']=='unqualified'


def test_nested_catalogs_require_a_real_opaque_implementation():
    from types import SimpleNamespace as NS
    from eval_studies import validate_catalogs
    catalogs=protocol()['routing_catalogs']
    implementations={name:NS(routing_kind='structural') for row in catalogs for name in row['structural']}
    assert validate_catalogs(catalogs,implementations)==dict(status='unavailable',missing_implementations=['opaque_binary'])
    implementations['opaque_binary']=NS(routing_kind='opaque')
    assert validate_catalogs(catalogs,implementations)['status']=='ready'
    changed=deepcopy(catalogs);changed[1]['structural']=changed[0]['structural']
    with pytest.raises(ValueError,match='strict nested'):
        validate_catalogs(changed,implementations)


def test_work_observer_counts_discarded_meters_without_changing_decisions():
    from QueryWork import QueryWorkBudget
    from eval_studies import observe_query_work
    original=QueryWorkBudget.consume
    with observe_query_work() as reading:
        first,discarded=QueryWorkBudget(3),QueryWorkBudget(2)
        assert first.consume('read',2)
        assert not first.consume('read',2)
        assert discarded.consume('candidate',1)
        assert first.spent==2 and discarded.spent==1
    assert QueryWorkBudget.consume is original
    assert reading['charged_units']==3 and reading['exhausted']==1

def test_referent_score_requires_fresh_identity_distinct_mentions_and_reuse():
    from eval_studies import score_referents, referent_expectations, load_tasks
    from types import SimpleNamespace as NS
    rows=load_tasks(STUDIES/'manifest.json','learning-heldout',family='mint_bind')
    model=NS(_concept_owner=lambda:NS(interpret=NS(word_for_form=lambda form:7)))
    expected=referent_expectations(model,rows)
    assert len(expected)==64
    expected=expected[:1]
    first=dict(leaves=[0,1],occurrence=31,reference=('ref',10),new_reference=True)
    second=dict(leaves=[2,3],occurrence=32,reference=('ref',10),new_reference=False)
    event=dict(sentence=0,word_ids=[3,7,4,7],audit=[first,second])
    assert score_referents(expected,{0:[event]})['value']==1
    for change in (dict(reference=('ref',20)),dict(occurrence=31)):
        changed=dict(event,audit=[first,dict(second,**change)])
        assert score_referents(expected,{0:[changed]})['value']==0
    assert score_referents(expected,{0:[dict(event,audit=[dict(first,new_reference=False),second])]})['value']==0
    assert score_referents(expected,{})['status']=='unavailable'
    assert score_referents(expected,{0:[dict(event,audit=[first,first,second])]})['value'] is None


def test_referent_observer_copies_pinned_words_and_ignores_stale_audits():
    import torch
    from types import SimpleNamespace as NS
    from eval_studies import observe_referents
    program=NS(word_ids=torch.tensor([3,7,4,7]));reference=('ref',1)
    pool=NS(entries={});pool.token_of=lambda ref:ref if ref in pool.entries else None
    owner=NS(mtm=pool)
    class Model:
        def _concept_owner(self):return owner
        def _commit_sentence(self,*args):
            program.word_ids=None
            pool.entries[reference]=object()
            self._reference_close_audit={(0,0):(dict(leaves=(0,1),occurrence=11,reference=reference,operation='mint'),)}
            return 'unchanged'
    model=Model()
    with observe_referents(model) as pending:
        result=model._commit_sentence(None,0,torch.tensor([True]),[dict(entries=[program])],None,torch.tensor([0]))
        assert result=='unchanged'
    assert pending[0]['word_ids']==[3,7,4,7]
    assert pending[0]['audit'][0]['new_reference']
    assert '_commit_sentence' not in model.__dict__
    program.word_ids=torch.tensor([3,7,4,7])
    original=lambda *args:'unchanged'
    model._commit_sentence=original
    with observe_referents(model) as pending:
        model._commit_sentence(None,0,torch.tensor([True]),[dict(entries=[program])],None,torch.tensor([0]))
    assert pending[0]['audit']==[]
    assert model._commit_sentence is original
