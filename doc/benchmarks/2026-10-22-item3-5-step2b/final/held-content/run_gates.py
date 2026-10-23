"""One frozen-source pass of the storage round's requested scoped gates."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
BASE = HERE.parents[2]
source = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()

def unchanged():
    assert not subprocess.check_output(['git','diff',source,'--name-only','--','bin','test','data'],
                                      cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','ls-files','--others','--exclude-standard','--','bin','test','data'],
                                      cwd=ROOT,text=True).strip()
    audit = json.loads((HERE/'source-audit.json').read_text())
    for name, digest in audit['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name

selections = {'regressions': ('medium_term_memory', 'reference_arrivals', 'reference_arrival_review', 'reference_choices', 'part_closing', 'reference_occurrence_views', 'reference_review10', 'reference_participants'), 'native': ('math_chain_section14', 'reference_close', 'reference_mentions', 'bind_thought', 'identity_rows', 'identity_cube', 'part_store', 'operators_final_b'), 'modules': ('step2b_storage', 'clause_storage', 'clause_acceptance', 'sentence_end_state', 'sentence_identity', 'selected_nested_meaning', 'sentence_retention', 'relation_closing', 'definition_taxonomy', 'truth_text', 'forgetting', 'concept_forgetting', 'accessible_mind', 'part_read', 'nested_retention', 'operators_round3a', 'operators_round4a0', 'operators_final', 'review14_contracts', 'learning_readiness', 'structural_checkpoint', 'ternary_truth_store', 'two_truth_stores', 'assertion_provenance', 'existence_provenance', 'nested_retention_owners', 'expectation_retention', 'occurrence_coordinates', 'live_query_occurrences', 'truth_grounded_reasoning', 'reasoning', 'reasoning_cde_model', 'mereological', 'reference_reconstruction', 'identity_corpus', 'query_executors', 'query_vp_boundaries', 'clause_closing', 'clause_scope', 'clause_grammar', 'sentence_journal', 'sentence_references', 'part_expectation', 'reference_index', 'independent_components', 'ltm0_location', 'compiled_expectation_boundary', 'existence_metadata', 'expectation_defaults', 'identification_evidence', 'item4_native', 'item55_mechanisms', 'joint_objectives', 'lies_can_be_learned', 'live_references', 'negative_expectation', 'predicate_identity', 'relation_references', 'relative_sentence_codebook_insertion', 'replacement_kernel_contracts', 'selected_relation_meaning', 'sentence_expectation', 'taxonomy_policy', 'tetralemma_leaf', 'truth_criterion_gates_learning', 'two_codebook_meta_taxonomy')}
selections = {k:[f'test/test_{n}.py' for n in names] for k,names in selections.items()}
flat = [s for selectors in selections.values() for s in selectors]
assert len(set(flat)) == len(flat)
assert all((ROOT/selector).is_file() for selector in flat)
changed = subprocess.check_output(['git','diff','bb0045d7a','--name-only','--','test'],
                                 cwd=ROOT,text=True).splitlines()
assert all(p in flat for p in changed if p.startswith('test/test_') and p.endswith('.py'))
unchanged()
assert not (HERE/'frozen-source.json').exists()
(HERE/'frozen-source.json').write_text(json.dumps(dict(source=source,selectors=selections,
    requested_scope='step 2b, spec 36 and 39–63; held-content guard; native includes the full eighth file',
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
runs=[]
for name,selectors in selections.items():
    unchanged()
    env=dict(os.environ,BASICMODEL_DEVICE='cpu',RUN_SLOW='1',
        DEVELOPER_DIR='/Library/Developer/CommandLineTools',
        PYTHONPATH=os.pathsep.join(map(str,(HERE,BASE/'2026-10-21-item3-5-reference-reconstruction',
            BASE/'2026-10-20-item3-5-stm1',BASE/'2026-10-20-item3-5-stm1/review3',
            BASE/'2026-10-19-item4-step0-round2',ROOT/'bin',ROOT/'test'))))
    for key in ('ARRIVAL_REPLAY','ARRIVAL_CLOSING_BASELINE','CLOSING_REPLAY'):
        env.pop(key,None)
    if name=='native':
        env.update(PYTEST_PLUGINS='closing_probe,arrival_capture,arrival_keep_probe,early_failure_probe,reference_probe,bootstrap_probe',
            CLOSING_TRACE=str(HERE/(name+'-trace.jsonl')),SCOPED_REPORTS=str(HERE/(name+'-reports')),
            ITEM35_BIND_RECEIPT=str(HERE/(name+'-generated')),
            BOOTSTRAP_CAPTURE=str(HERE/(name+'-bootstrap-state')),
            ARRIVAL_KEEP_DIR=str(HERE/(name+'-keep')),ARRIVAL_KEEP_SOURCE=source,
            REFERENCE_EVIDENCE=str(HERE/(name+'-reference-evidence')),
            ARRIVAL_CAPTURE=str(HERE/(name+'-state')))
    else:
        env.update(PYTEST_PLUGINS='storage_capture',CONTRACT_CAPTURE=str(HERE/(name+'-state')))
    command=[str(ROOT/'.venv/bin/python'),'test/test_report.py',*selectors,
        '--workers','1','--memory-gib','8','--timeout','900','--suite-timeout','5400',
        '--batch-size','16','--max-files','1','--run-dir',str(HERE/name)]
    assert not (HERE/name).exists()
    record=dict(source=source,name=name,command=command,
                started=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with (HERE/(name+'.log')).open('x') as log:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    record.update(exit=result.returncode,finished=datetime.datetime.now(datetime.timezone.utc).isoformat())
    runs.append(record)
    (HERE/'gate-runs.json').write_text(json.dumps(runs,indent=2)+'\n')
    unchanged()
    print(name,result.returncode,flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
