"""The reviewed known ladder; expectations never enter the model's read."""
import json
from pathlib import Path

from Assessment import read_sentence, format_record
from game_corpus import render


def specification():
    return json.loads((Path(__file__).resolve().parents[1] / 'data/assessment/known_probe.json').read_text())


def _references(meaning):
    if not meaning:
        return set()
    return {tuple(ref) for ref in meaning['references'] if ref is not None} | set().union(
        *(_references(child) for child in meaning['children']))


def _checks(record, rung):
    failures = []
    if record['missing']:
        failures.append('missing leaves')
    if record['forest'] != [rung['tree']]:
        failures.append('different tree')
    def repeated(tree):
        return 'operator' in tree and (tree['count'] > 1 or any(map(repeated, tree['children'])))
    if any(map(repeated, record['trees'])):
        failures.append('repeated unary operator')
    if not all(word['known'] for word in record['words']):
        failures.append('unknown word')
    checks, words = rung['checks'], record['words']
    if 'minted' in checks and not all(word['attention'] == 'minted' for word in words):
        failures.append('word not minted')
    if 'trusted_definition' in checks and not all(word['trusted'] for word in words):
        failures.append('definition not trusted')
    if ('one_object' in checks or 'same_object' in checks) and not all(len(word['objects']) == 1 for word in words):
        failures.append('word does not denote one object')
    if 'said_back' in checks and record['answer'] != record['text']:
        failures.append('not said back')
    if 'earlier_referent' in checks:
        earlier = {tuple(ref) for ref in record['context_subjects']}
        if not earlier.intersection(_references(record['meaning'])):
            failures.append('no earlier referent')
    return failures


def known_probe(model, vocabulary, *, source, scalar=None, trace=False):
    """Assess available rungs in order, without training or changing a result.

    The caller declares its taught kinds; this reader never guesses a part of
    speech from a spelling. Unavailable and failed prerequisites block later
    rungs. These reviewed trees still report without gating a test or lesson.
    """
    spec = specification()
    table = []
    for rung in spec['rungs']:
        result = dict(rung=rung['rung'], status='not applicable', records=[])
        missing = [kind for kind in rung['kinds'] if not vocabulary.get(kind)]
        if rung.get('unavailable'):
            result['reason'] = rung['unavailable']
        elif missing:
            result['reason'] = 'fixture has no declared '+', '.join(missing)
        elif rung['rung'] == 2 and len(set(vocabulary['words'])) < 20:
            result['reason'] = f"fixture declares {len(set(vocabulary['words']))} distinct words, fewer than twenty"
        elif any(prior['status'] != 'passed' for prior in table):
            result.update(status='blocked', reason='lower rung has not passed')
        else:
            assignments = [dict(vocabulary, word=word) for word in dict.fromkeys(vocabulary['words'])] if rung['rung'] == 2 else [vocabulary]
            failures, objects = [], []
            for assignment in assignments:
                slots = {key: dict(form=value) for key, value in assignment.items() if isinstance(value, str)}
                text = render(rung['frame'], slots)
                context = [render(frame, slots) for frame in rung.get('context', [])]
                readings = []
                for _ in range(spec['repetitions'] if rung['rung'] <= 2 else 1):
                    record = read_sentence(model, text, scalar=scalar, context=context,
                        answer='said_back' in rung['checks'], trace=trace)
                    readings.append(record)
                    failures.extend(_checks(record, rung))
                if 'same_object' in rung['checks']:
                    identities = [record['words'][0]['objects'] if record['words'] else [] for record in readings]
                    if any(ids != identities[0] for ids in identities[1:]):
                        failures.append('object changed between readings')
                    objects.extend(identities[0])
                result['records'].extend(readings)
            if 'distinct_objects' in rung['checks'] and len(set(objects)) != len(assignments):
                failures.append('objects are not distinct')
            result.update(status='failed' if failures else 'passed', reasons=list(dict.fromkeys(failures)))
        table.append(result)
    return dict(source=source, status=spec['status'], vocabulary=vocabulary, rungs=table)


def unavailable(source, reason):
    return dict(source=source, status=specification()['status'], rungs=[
        dict(rung=n, status='not applicable', reason=reason, records=[]) for n in range(1, 10)])


def readiness(report, rung):
    result = report['rungs'][rung-1]
    return 'ready' if result['status'] == 'passed' else 'not ready: '+result['status']


def format_table(report):
    lines = [report['source'], 'rung | result | reason']
    for row in report['rungs']:
        lines.append(f"{row['rung']} | {row['status']} | "+row.get('reason', ', '.join(row.get('reasons', []))))
        if row['status'] == 'failed':
            lines.extend(format_record(record) for record in row['records'])
    return '\n'.join(lines)
