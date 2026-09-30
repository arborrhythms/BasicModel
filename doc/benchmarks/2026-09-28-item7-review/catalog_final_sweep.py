"""Describe every final failure without changing, waiving or rerunning it."""
from collections import Counter
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CURRENT = HERE / 'closing-sweep'


def reports(result):
    values = {}
    for worker in result['workers']:
        for report in worker.get('reports', []):
            if report['phase'] == 'call' or report['outcome'] != 'passed':
                if values.get(report['nodeid'], {}).get('outcome') != 'failed':
                    values[report['nodeid']] = report
    return values


def main():
    summary = json.loads((CURRENT / 'summary.json').read_text())
    assert summary['coverage_complete_unique']
    previous = reports(json.loads((HERE / 'rename/full/result.json').read_text()))
    current = reports(json.loads((CURRENT / 'full/result.json').read_text()))
    whole_files = {
        'test_mental_model.py', 'test_perceptual_loopback.py',
        'test_init_scale.py', 'test_shamatha_inline_grammar.py',
        'test_subspace_context.py', 'test_bivector_retirement_state.py',
        'test_sparsity_regularizer.py', 'test_phase2_end_to_end.py',
        'test_word_space_attach_knowledge.py',
    }
    other_groups = {
        'test_item9b_schedule.py': 'Compilation',
        'test_gradient_factorization.py': 'Gradient assertions',
        'test_reasoning_cde_model.py': 'Batch and clause admission',
        'test_space_equiv_selfcheck.py': 'Batch and clause admission',
        'test_output_synthesis.py': 'Answer and MM output',
        'test_mm_xor.py': 'Answer and MM output',
        'test_runtime_split_ingestion.py': 'Provisioning counts and reset',
        'test_expectation_review.py': 'Provisioning counts and reset',
        'test_compose_pair_driver.py': 'Action, depth and history assertions',
        'test_subspace_what_stm_contract.py': 'Action, depth and history assertions',
        'test_relevance_bases.py': 'Action, depth and history assertions',
    }
    failures = []
    for failure in summary['failures']:
        nodeid = failure['nodeid']
        filename = nodeid.split('::')[0].split('/')[-1]
        if filename in whole_files:
            group = 'Retired WholeSpace API expectations'
        elif filename == 'test_compiled_word_chunk.py':
            group = ('Gradient assertions' if nodeid.endswith(
                'test_tiny_canonical_detached_reverse_stops_at_root') else 'Compilation')
        else:
            group = other_groups[filename]
        prior = previous.get(nodeid)
        failures.append(dict(group=group, prior_outcome=(prior or {}).get('outcome'),
            prior_report=prior, current_report=failure))
    assert len(failures) == len({f['current_report']['nodeid'] for f in failures}) == 30
    item7 = {n: r['outcome'] for n, r in current.items() if n.startswith('test/test_item7_')}
    result = dict(
        status='Red; stopped for review, not accepted or committed',
        source_digest=summary['source_digest'], source_files=summary['source_files'],
        complete_unique_coverage=summary['coverage_complete_unique'],
        comparison='Same selector names against the mechanical-rename sweep. A renamed selector is unmatched, not normalized to a different test.',
        failure_groups=dict(Counter(f['group'] for f in failures)),
        prior_outcomes_of_current_failures=dict(Counter(
            f['prior_outcome'] or 'unmatched selector' for f in failures)),
        changed_outcome_counts={f'{a} -> {b}': c for (a, b), c in Counter(
            (x['before'], x['after']) for x in summary['changed_outcomes']).items()},
        item7_outcomes=dict(Counter(item7.values())), item7_cases=item7,
        new_selectors=summary['new_cases'], removed_selectors=summary['removed_cases'],
        selector_note='The 98 added and 86 removed selector names include renames and replacements; the 43 explicitly retired WholeSpace cases have their separate disposition.',
        retired_case_disposition='../sweep-fixtures/retired-cases.json',
        failures=failures,
    )
    (CURRENT / 'failure-ledger.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('failure_groups',
        'prior_outcomes_of_current_failures', 'changed_outcome_counts', 'item7_outcomes')}, indent=2))


if __name__ == '__main__':
    main()
