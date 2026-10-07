"""Boundary-only routing measurements from committed programs, not candidates.

These diagnostics never enter grammar features, training credit or answer
construction. A descriptive zero is not evidence of learned utility.
"""
from collections import Counter
import hashlib
import json

from GrammarPreference import operator_is_structural
from LearningEvaluation import fineweb_readiness


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class RoutingCoverage:
    def __init__(self, language, *, corpus, stage):
        if not corpus or not stage:
            raise ValueError('routing evidence requires a corpus identity and stage')
        self.corpus, self.stage = corpus, stage
        self.catalog = {}
        for kind, arity, label in ((1, 2, 'binary'), (2, 1, 'unary')):
            rules = getattr(language, f'_compose_{label}_rules')
            layer = language._tree_layer(arity)
            ops = () if layer is None else (layer.ops if arity == 2 else layer.unary_ops)
            if len(rules) != len(ops):
                raise ValueError('routing measurement requires the owned grammar catalog')
            self.catalog[kind] = [dict(name=rule.method_name,
                structural=operator_is_structural(op)) for rule, op in zip(rules, ops)]
        self.rows = []
        self._ids = set()

    def record(self, program, *, sentence_id):
        if not isinstance(sentence_id, str) or not sentence_id or sentence_id in self._ids:
            raise ValueError('routing sentence IDs must be unique nonempty strings')
        if program is None or len(program.leaves) == 0:
            raise ValueError('missing/empty sentence program is not a zero opaque count')
        actions = program.actions.detach().cpu()
        if actions.ndim != 2 or actions.shape[1] != 3:
            raise ValueError('invalid committed grammar actions')
        counts, opaque, shifted, padded = Counter(), 0, [], False
        for kind, local, word in actions.tolist():
            if kind < 0:
                padded = True
                continue
            if padded:
                raise ValueError('non-padding action follows terminal padding')
            if kind == 0:
                shifted.append(word)
                continue
            if kind not in self.catalog or not 0 <= local < len(self.catalog[kind]):
                raise ValueError('unknown committed rule cannot disappear from coverage')
            rule = self.catalog[kind][local]
            counts[rule['name']] += 1
            opaque += not rule['structural']
        if shifted != list(range(len(program.leaves))):
            raise ValueError('routing program does not own every sentence leaf exactly once')
        self._ids.add(sentence_id)
        self.rows.append(dict(sentence_id=sentence_id, operations=sum(counts.values()),
            opaque_operations=opaque, rules=dict(counts)))

    def report(self):
        n = len(self.rows)
        operations = sum(r['operations'] for r in self.rows)
        opaque = sum(r['opaque_operations'] for r in self.rows)
        opaque_sentences = sum(r['opaque_operations'] > 0 for r in self.rows)
        return dict(corpus=self.corpus, stage=self.stage, catalog=self.catalog,
            catalog_sha256=digest(self.catalog), sentences=n, operations=operations,
            opaque_operations=opaque, opaque_sentences=opaque_sentences,
            no_operation_sentences=sum(r['operations'] == 0 for r in self.rows),
            sentence_opaque_share=opaque_sentences / n if n else None,
            operation_opaque_share=opaque / operations if operations else None,
            rows=list(self.rows), learned_utility='unproven')


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
