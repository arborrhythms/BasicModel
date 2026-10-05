"""Classify saved failures without running or altering a test."""
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def classification(node, message):
    file = node.split('::')[0]
    if 'test_normal_text_reconstruction_updates_the_grammar_chooser' in node:
        return 'regression', ('Held: the unchanged fixed-seed fixture requires chooser movement, '
            'but both reconstruction comparisons tie exactly. The strict-improvement rule '
            'permits no preference update. No seed, assertion or fixture has been changed.')
    if 'concept inventory exhausted before percept admission' in message or 'test_22_new_word' in node:
        return 'regression', ('Removed automatic conceptual-row allocation for letter percepts '
            'in the accepted §14 addendum; symbol/concept capacities remain 6/8. Assertions unchanged.')
    ports = {
        'test_gradient_factorization.py': 'The derived dictionary is a refreshed buffer, not a free Parameter; native perception parameters retain reconstruction ownership.',
        'test_interleave_schedule.py': 'The cache buffer is shared, while getW derives fresh values; a fresh read need not have the old data_ptr or old values.',
        'test_wholespace_property_migration.py': 'The shared derived cache occurs zero times among optimizer parameters.',
        'test_readback_log_probability.py': 'Byte-scoring bank codes are detached; the recovered leaf keeps gradient.',
        'test_grammar_reconstruction_gate.py': 'The pair inverse returns hard detached bank members, replacing the soft symmetric inverse.',
        'test_tied_operator_reconstruction.py': 'Hard pair search replaces the soft inverse; retained wrapper branches may expose a zero pullback, never competitor-code credit.',
        'test_review13_subspaces.py': 'Full-presence part max replaces evidence-scaled parts and the interval midpoint.',
        'test_review14_addendum.py': 'Full-presence part max replaces the interval midpoint; no conceptual rows are allocated for letters.',
        'test_reduction_pressure.py': 'Two sentence closings can also create nested clause rows; count the two top-level writes while retaining depth and unary checks.',
        'test_sentence_end_state.py': 'The fixture includes and checks the new selected-log-probability journal column.',
        'test_thought_answer_adapters.py': 'The numerical identity stub accepts the declared-role occupancy keyword; assertions unchanged.',
    }
    name = Path(file).name
    if name in ports:
        return 'port', ports[name]
    if name == 'test_review14_contracts.py':
        if 'measurement_wiring' in node:
            return 'regression', ('The current audit runs one ordinary eager batch, observes the '
                'same optimizer steps, and supports the actual AnchorDot chooser as well as MLP. '
                'Its preference owner includes the operation anchors.')
        return 'port', '§15 full-presence form, whole-only room, raw affine reader and hard inverse replace the §14 expectations.'
    regressions = {
        'test_compose_review.py': 'Reshape the supplied exploration draw to the logits batch dimensions; assertions unchanged.',
        'test_inverse_unary_exclusion.py': 'Construct the random draw with integer dimensions, avoiding the patched torch.full tuple-size error; assertions unchanged.',
        'test_grammar_separator.py': 'Use the existing identity to resolve a missing physical grammar row for both forward leaves and the decoder bank; retain the two-word assertion. The separator fixture supplies matching forward and inverse operators.',
        'test_mind_generativity.py': 'Admit the fixture forms through native definitions before reading their derived codes; unused inventory rows correctly remain zero. Seed and assertions unchanged.',
        'test_reasoning.py': 'The existing cancellation consistency probe uses the catalogued sum (mean), preserving its prior arithmetic after disjunction was renamed.',
        'test_reconstruction_roundtrip.py': 'Reduce singleton axes from the innermost outward, matching the unchanged reduction-order contract without altering tolerance.',
        'test_output_walk.py': 'Keep the materialization compatibility result for generation targets as None; declared occupancy is a separate internal read. Assertions unchanged.',
    }
    if name in regressions:
        return 'regression', regressions[name]
    if name in ('test_prepared_answer_boundary.py', 'test_trial_policy_ownership.py',
                'test_generation_catalog.py', 'test_output_path_supervised.py',
                'test_arithmetic_isolation.py'):
        return 'regression', ('Declared role masks preserve zero-valued resolved answers; native '
            'answer synthesis uses its operator inverse without lexical-bank eligibility or hard-pair '
            'substitution. Output-gradient assertions are unchanged. The explicit expansion fixture '
            'uses a captured live word form rather than a root a fresh negation may erase.')
    raise ValueError((node, message[-400:]))


def main():
    failures = []
    old = json.loads((HERE/'review14-sweep-summary.json').read_text())
    for r in old['failure_reports']:
        failures.append(dict(run='review14-sweep', **r))
    for path in sorted((HERE/'probes').glob('review15*/worker-*.json')):
        for r in json.loads(path.read_text()).get('reports', []):
            if r['outcome'] == 'failed':
                failures.append(dict(run=str(path.parent.relative_to(HERE)), **r))
    for r in json.loads((HERE/'review15-sweep-summary.json').read_text())['failures']:
        failures.append(dict(run='review15-sweep', **r))
    result = []
    for row in failures:
        kind, treatment = classification(row['nodeid'], row.get('message', ''))
        result.append(dict(row, classification=kind, treatment=treatment))
    unique = {row['nodeid']: row['classification'] for row in result}
    output = dict(note='Historical failures stand. This classifies their treatment; current outcomes are in the final focused report.',
                  distinct_failed_contracts=dict(collections.Counter(unique.values())),
                  reports=result)
    with (HERE/'review15-failure-classification.json').open('x') as f:
        json.dump(output, f, indent=2)
        f.write('\n')
    print(json.dumps(output['distinct_failed_contracts']))


if __name__ == '__main__':
    main()
