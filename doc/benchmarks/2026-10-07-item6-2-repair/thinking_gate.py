"""One unseeded thinking measurement; retain every node and every outcome."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'test'),str(HERE)]
import bounded_tests as bounded


def main():
    from verification import validate
    source=bounded.source_snapshot(ROOT)
    validate(source)
    helpers=json.loads((HERE/'measured-source/measurement-helpers.json').read_text())
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha for name,sha in helpers.items())
    old=json.loads(gzip.decompress((HERE.parent/'2026-09-24-item11c/explicit-result.json.gz').read_bytes()))
    retired={row['nodeid'] for row in json.loads((HERE/'test-retirements.json').read_text())}
    assert retired <= set(old['selected'])
    selectors=[*(node for node in old['selected'] if node not in retired),
        'test/test_reasoning_cde_model.py::TestReasoningCDEModel::test_gates_on',
        'test/test_reasoning_cde_model.py::TestReasoningCDEModel::test_training_step_uses_the_normal_policy_configuration',
        'test/test_unified_thought_controller.py::test_runbatch_credits_each_controller_row_from_its_own_answer',
        'test/test_item6_2_thinking.py', 'test/test_item6_2_repair.py']
    plan=dict(selectors=selectors,source=source,seed=None,retries=0,
        scope='Ported 11c seventeen explicit mechanism nodes; two retired removed contracts saved in test-retirements.json; MM_query_reasoning configured one-batch optimizer smoke; native chaining, answer and full expectation credit certificates.',
        retirements=sorted(retired), seed_policy='No fixture seed calls; the ported 11c helper is unseeded. Any explicit seed attempt fails the measurement.',
        claim='Mechanism and optimizer checks, not learned parsing or unforced chain success rates.')
    path=HERE/'thinking-gate-plan.json'
    assert not path.exists(), 'the thinking measurement has already been declared'
    path.write_text(json.dumps(plan,indent=2)+'\n')
    os.environ.pop('BASIC_SEED',None)
    os.environ.update(MODEL_COMPILE='none',BASICMODEL_DEVICE='cpu',RUN_SLOW='1',
        BASIC_AUTOLOAD='false',BASIC_AUTOSAVE='false',PYTHONDONTWRITEBYTECODE='1',
        PYTEST_PLUGINS='thinking_gate_observer',
        THINKING_GATE_OUTPUT=str(HERE/'thinking-observations'),
        PYTHONPATH=os.pathsep.join((str(HERE),str(ROOT/'bin'),str(ROOT/'test'))))
    code,report_path=bounded.main([*selectors,'--workers','2','--memory-gib','16',
        '--batch-size','128','--run-dir',str(HERE/'thinking-gate')])
    result=json.loads(report_path.with_name('result.json').read_text())
    print(json.dumps(dict(exit_code=code,reason=result['reason'],selected=len(result['selected']),completed=len(result['completed']))))
    return code

if __name__=='__main__':
    raise SystemExit(main())
