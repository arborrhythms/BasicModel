"""Correct registered back-reference labels from saved sufficient statistics.

No model, forward pass, optimizer step or new measurement. Every recovered
nonzero norm/cosine must exactly match a saved bucket with the same support.
"""
import copy
import itertools
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def is_vocabulary(name):
    return '._vocabulary.' in name


def is_head(name):
    return name.startswith('inputSpace.outputSpace.') and not is_vocabulary(name)


def is_intra(name):
    return name.startswith('conceptualSpaces.') and '.layers.0.' in name


for arm in ('step5a', 'cut'):
    folder = HERE / ('XOR_grammar-' + arm)
    original = json.loads((folder / 'parameter-groups.json').read_text())
    groups = copy.deepcopy(original)
    groups['perception'] = [p for p in original['perception'] if not is_head(p['name'])]
    groups['reading_map'] = [p for p in original['reading_map'] if not is_vocabulary(p['name'])]
    groups['expectation_predictor'] += [p for p in original['other'] if is_intra(p['name'])]
    groups['other'] = [p for p in original['other'] if not is_intra(p['name'])]
    (folder / 'parameter-groups-by-role.json').write_text(json.dumps(groups, indent=2) + '\n')
    path = folder / 'gradients.json'
    if not path.exists():
        continue
    snapshots = json.loads(path.read_text())
    corrected = copy.deepcopy(snapshots)
    for raw, result in zip(snapshots, corrected):
        buckets = raw['groups']
        objectives = list(raw['costs'])
        for name, members in groups.items():
            members = {p['name'] for p in members}
            support = {o: members & set().union(*(set(b['nonzero_parameters'][o])
                       for b in buckets.values())) for o in objectives}
            norms = {}
            for o, parameters in support.items():
                if not parameters:
                    norms[o] = 0.
                    continue
                matches = [b['norms'][o] for b in buckets.values()
                           if set(b['nonzero_parameters'][o]) == parameters]
                assert matches and len(set(matches)) == 1, (name, o, parameters)
                norms[o] = matches[0]
            cosines = {}
            for a, b in itertools.combinations(objectives, 2):
                key = a + '__' + b
                if not norms[a] or not norms[b]:
                    cosines[key] = None
                elif not support[a] & support[b]:
                    cosines[key] = 0.
                else:
                    matches = [v['cosines'][key] for v in buckets.values()
                               if set(v['nonzero_parameters'][a]) == support[a]
                               and set(v['nonzero_parameters'][b]) == support[b]]
                    assert matches and len(set(matches)) == 1, (name, key)
                    cosines[key] = matches[0]
            result['groups'][name] = dict(norms=norms, cosines=cosines,
                nonzero_parameters={o: sorted(p) for o, p in support.items()})
        result['role_correction'] = ('InputSpace registers OutputSpace as a back-reference; '
            'its head is the reading map and its shared vocabulary is perception. '
            'ConceptualSpace.layers.0 is intraSentenceLayer. No new measurement; '
            'all nonzero norms/cosines recovered exactly from identical saved supports.')
    (folder / 'gradients-by-role.json').write_text(json.dumps(corrected, indent=2) + '\n')
print('Role labels corrected from saved statistics; no model constructed.')
