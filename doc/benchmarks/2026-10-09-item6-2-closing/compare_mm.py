"""Compare diagnostic replays to each retained original MM miss."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def main():
    before = read(HERE/'standing-summary-before-bisection.json')
    cases = []
    for row in before['outcomes']:
        if row['kind'] != 'mm' or row['bar']:
            continue
        name = row['name']
        original, = [item for item in map(json.loads,
            (HERE/'measurements'/name/'observations.jsonl').read_text().splitlines()) if item['kind'] == 'mm']
        closing = read(HERE/'mm-bisection'/f'{name}-closing.json')
        landing = read(HERE/'mm-bisection'/f'{name}-landing.json')
        original_trajectory = [dict(epoch=item['epoch'], mse=item['mse'], predictions=item['predictions'])
                               for item in closing['trajectory']]
        original_matches = (original['trajectory'] == original_trajectory and
                            original['best'] == closing['best'] and original['calls'] == closing['steps'])
        fields = ('construction_rng', 'initial_parameters', 'trajectory', 'best', 'steps',
                  'final_parameters', 'final_rng')
        compared = {field: closing[field] == landing[field] for field in fields}
        cases.append(dict(name=name, original_best=original['best'], original_steps=original['calls'],
            raw_bar=False, replay_matches_original=original_matches, equality=compared,
            landing_identical=original_matches and all(compared.values()),
            closing=f'{name}-closing.json', landing=f'{name}-landing.json'))
    result = dict(kind='Diagnostic bisection; no declared attempt retried or replaced',
        landing_commit='e43638a747e373b3b10343c642f76cd1d0757ec1',
        closing_source_sha256=before['source_sha256'], seed=None, cases=cases,
        all_misses_identical=all(row['landing_identical'] for row in cases),
        basis='Operators plan section 20: original misses retain their raw outcomes. '
              'Equality of the complete numerical and RNG paths rules out a closing-source regression.')
    path = HERE/'mm-bisection/result.json'
    with path.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
