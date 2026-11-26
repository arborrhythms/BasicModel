"""Boundary-only routing measurements from committed programs, not candidates.

These diagnostics never enter grammar features, training credit or answer
construction. A descriptive zero is not evidence of learned utility.
"""

from LearningEvaluation import fineweb_readiness


def compare_coverage(stages, training_states):
    """Validate a same-corpus, growing-catalog comparison before claiming decline.

    This tests only routing decline, never the separate causal quality gate.
    Readiness reuses item 9's million-sentence prerequisite with no override.
    """
    if len(stages) < 2 or len(stages) != len(training_states):
        raise ValueError('coverage comparison needs aligned stages and checkpoints')
    identities = [r['sentence_id'] for r in stages[0]['rows']]
    if not identities:
        raise ValueError('coverage comparison requires held-out sentences')
    previous, opaque_catalog = None, None
    for stage in stages:
        if (stage['corpus'] != stages[0]['corpus'] or
                [r['sentence_id'] for r in stage['rows']] != identities):
            raise ValueError('coverage comparisons require the same ordered corpus')
        structured = {(str(kind), r['name']) for kind, rules in stage['catalog'].items()
                      for r in rules if r['structural']}
        opaque = {(str(kind), r['name']) for kind, rules in stage['catalog'].items()
                  for r in rules if not r['structural']}
        if previous is not None and not previous < structured:
            raise ValueError('structural coverage must grow by strict catalog inclusion')
        if opaque_catalog is not None and opaque_catalog != opaque:
            raise ValueError('the opaque candidate inventory must stay fixed')
        previous, opaque_catalog = structured, opaque
    ready = [fineweb_readiness(state) for state in training_states]
    if not all(r.eligible for r in ready):
        return dict(status='skipped', reasons=[r.reason for r in ready if not r.eligible],
                    learned_utility='unproven')
    if not opaque_catalog:
        return dict(status='unavailable', reason='no opaque candidate in the catalog',
                    learned_utility='unproven')
    shares = [sum(r['opaque_operations'] > 0 for r in s['rows']) / len(identities)
              for s in stages]
    return dict(status='passed' if shares[-1] < shares[0] else 'failed',
                sentence_opaque_shares=shares, learned_utility='unproven')
