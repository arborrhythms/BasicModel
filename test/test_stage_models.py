"""A trained stage is restored and measured; its test never supplies training."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('phase', [
    pytest.param('stage_words', marks=pytest.mark.trained('stage_words')),
    pytest.param('stage_sentences', marks=pytest.mark.trained('stage_sentences')),
    pytest.param('stage_math', marks=pytest.mark.trained('stage_math')),
])
@pytest.mark.artifact_eval
def test_saved_stage_reproduces_criteria_and_operator_column(phase, phase_registry):
    from eval_operators import digest, operator_table
    from eval_lessons import require_criteria
    artifact = phase_registry.require(phase, strict=True)
    receipt = json.loads(phase_registry.resolve(artifact.receipt).read_text())
    assert artifact.config is not None
    config = phase_registry.resolve(artifact.config)
    assert digest(config) == receipt['config_sha256']
    assert digest(ROOT/'data/training-corpus.json') == receipt['training_manifest_sha256']
    models = {'M0': receipt['baseline'], phase:dict(config=str(config),
        checkpoint=str(phase_registry.resolve(artifact.path)))}
    table = operator_table(models, ROOT/'data/eval/stage-models.json')
    reading = table['readings'][phase]
    assert reading['status'] == 'complete', reading
    require_criteria([reading['criteria'][name]['criterion'] for name in receipt['required_lessons']])
    assert table['rows'] == receipt['operator_rows']
    # Earlier-stage retention is a reading; there is no undeclared retention bar.


def test_undeclared_unavailable_and_empty_criteria_do_not_pass():
    from eval_lessons import criterion_reading, require_criteria
    readings=dict(trust=dict(admitted_words=4,stable_word_fraction=.75,named_words=0,
        denoting_word_fraction=0,operations={'lift':dict(measurable=True,definiteness=[.8,.9])}),
        answers=[dict(text_correct=True),dict(text_correct=False)])
    result=criterion_reading(dict(measure='definition_trust_fraction',threshold=None),readings)
    assert result['value']==.75 and result['status']=='undeclared' and not result['passed']
    result=criterion_reading(dict(measure='denotation_fraction',threshold=.5),readings)
    assert result['value'] is None and result['status']=='unavailable'
    result=criterion_reading(dict(measure='occupant_definiteness_and_answers',threshold=.6),readings,operations=['lift'])
    assert result['value']==.5 and not result['passed']
    result=criterion_reading(dict(measure='occupant_definiteness',threshold=.8),readings,operations=['lift'])
    assert result['passed']
    require_criteria([result])
    for values in ([], [dict(result,passed=False)]):
        with pytest.raises(ValueError,match='all declared'):
            require_criteria(values)


def test_operator_delta_reports_zero_baseline_and_rejects_changed_ownership():
    import torch
    from eval_operators import parameter_change
    result=parameter_change({'weight':torch.tensor([3.,4.])},{'weight':torch.tensor([0.,0.])})
    assert result['delta_norm']==5. and result['relative_delta'] is None
    result=parameter_change({'weight':torch.tensor([6.,8.])},{'weight':torch.tensor([3.,4.])})
    assert result['relative_delta']==1.
    assert parameter_change({}, {})['parameters']==0
    with pytest.raises(ValueError,match='ownership'):
        parameter_change({}, {'weight':torch.ones(1)})


def test_operator_use_counts_kept_actions_and_rejects_missing_leaves():
    import torch
    from types import SimpleNamespace as NS
    from eval_operators import count_program
    language=NS(_compose_binary_rules=[NS(method_name='lower',surface_name='definite')],
                _compose_unary_rules=[])
    program=NS(leaves=torch.ones(2,4),actions=torch.tensor([[0,-1,0],[0,-1,1],[1,0,-1],[-1,-1,-1]]))
    assert count_program(program,language)=={'definite':1}
    program.actions[:2,2]=torch.tensor([1,0])
    assert count_program(program,language)=={'definite':1}
    program.actions[:2,2]=0
    with pytest.raises(ValueError,match='every leaf'):
        count_program(program,language)


def test_operator_observer_counts_only_kept_active_program_and_restores_method():
    import torch
    from types import SimpleNamespace as NS
    from eval_operators import observe_kept_programs
    program=NS(leaves=torch.ones(2,4),actions=torch.tensor([[0,-1,0],[0,-1,1],[1,0,-1]]))
    language=NS(_compose_binary_rules=[NS(method_name='lower',surface_name='definite')],
                _compose_unary_rules=[])
    token=object()
    class Model:
        languageSpace=language
        calls=0
        def _commit_sentence(self,*args):
            self.calls+=1
            # The native journal is no longer readable after commit.
            program.leaves=None
            return token
    model=Model()
    observations=[dict(entries=[None,None]),dict(entries=[program,None])]
    with observe_kept_programs(model) as reading:
        assert model._commit_sentence(None,0,torch.tensor([True,False]),observations,
            None,torch.tensor([1,0])) is token
    assert reading==dict(counts={'definite':1},sentences=1)
    assert model.calls==1 and '_commit_sentence' not in model.__dict__


def test_rehearsal_selects_declared_document_prefix_without_resizing():
    from stage_models import select_documents, math_stages
    documents=[dict(document='z',records=[1,2]),dict(document='a',records=[3])]
    assert select_documents(documents,1)==[documents[1]]
    assert len(list(math_stages(ROOT,2)))==len(json.loads((ROOT/'data/math/manifest.json').read_text())['grades'])
    for stage in math_stages(ROOT,2):
        assert len(stage['documents'])<=2
        assert bool(stage['documents']) != stage['evaluation_only']
        assert all(document['records'] for document in stage['documents'])
