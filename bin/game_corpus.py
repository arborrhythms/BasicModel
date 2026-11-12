"""Template-tier transformation games: expand a recipe's frames into pairs.

A recipe (data/games/recipes/<game>.json) holds, per variant, the slot
classes, the source frames and the target frame, and per class the filler
entries with every form a frame names. This module substitutes slots and
splits rows; it holds no word of any language. Item 0.5 plan, section 6.
"""
import hashlib
import itertools
import json
import random
import re
from pathlib import Path

SPLITS = ('train', 'validation', 'test', 'transfer')
_SLOT = re.compile(r'\{(\w+)\.(\w+)\}')


def content_split(text):
    """sha256 of the normalized text modulo ten: 0-7 train, 8 validation, 9 test."""
    normalized = ' '.join(text.lower().split())
    h = int(hashlib.sha256(normalized.encode()).hexdigest(), 16) % 10
    return 'train' if h < 8 else 'validation' if h == 8 else 'test'


def render(frame, assignment):
    def form(match):
        slot, field = match.groups()
        entry = assignment[slot]
        if field not in entry:
            raise ValueError(f'filler {entry} lacks the form {field!r} that a frame needs')
        return entry[field]
    return _SLOT.sub(form, frame)


def _assignments(recipe, variant):
    frame = recipe['variants'][variant]
    slots = tuple(frame['slots'])
    pools = []
    for slot in slots:
        cls = frame['slots'][slot]
        pool = [(entry, False) for entry in recipe['fillers'][cls]]
        pool += [(entry, True) for entry in recipe.get('transfer', {}).get(cls, ())]
        if not pool:
            raise ValueError(f'slot {slot} names the empty class {cls}')
        pools.append(pool)
    distinct = [tuple(group) for group in frame.get('distinct', ())]
    for choice in itertools.product(*pools):
        assignment = {slot: entry for slot, (entry, _) in zip(slots, choice)}
        if any(len({json.dumps(assignment[s], sort_keys=True) for s in group}) < len(group)
               for group in distinct):
            continue
        yield assignment, any(flag for _, flag in choice)


def expand(recipe, *, seed):
    """All rows of every variant; ``seed`` samples a variant's ``limit``."""
    rng = random.Random(seed)
    rows = []
    for variant in sorted(recipe['variants']):
        frame = recipe['variants'][variant]
        found = []
        for assignment, transfer in _assignments(recipe, variant):
            sources = [render(text, assignment) for text in frame['sources']]
            target = render(frame['target'], assignment)
            split = 'transfer' if transfer else content_split(sources[0])
            found.append((sources, target, split))
        found.sort()
        limit = frame.get('limit')
        if limit is not None and len(found) > int(limit):
            found = sorted(rng.sample(found, int(limit)))
        for index, (sources, target, split) in enumerate(found):
            rows.append(dict(game=recipe['game'], variant=variant, tier='template', split=split,
                             key=f"{recipe['game']}-{variant}-{index:06d}", sources=sources, target=target))
    return rows


def verify(rows, recipe):
    """Accept a row whose pair and split the recipe produces; reject the rest."""
    expected = {}
    for variant in recipe['variants']:
        frame = recipe['variants'][variant]
        for assignment, transfer in _assignments(recipe, variant):
            sources = tuple(render(text, assignment) for text in frame['sources'])
            expected[(variant, sources, render(frame['target'], assignment))] = (
                'transfer' if transfer else content_split(sources[0]))
    accepted, rejected = 0, []
    for row in rows:
        split = expected.get((row['variant'], tuple(row['sources']), row['target']))
        if split is not None and split == row['split'] and row['game'] == recipe['game']:
            accepted += 1
        else:
            rejected.append(row['key'])
    return dict(accepted=accepted, rejected=rejected)


def freeze(recipes_dir, out_dir, *, seed):
    """Write data/games/<game>/<variant>.jsonl and manifest.json; tests never call this."""
    recipes_dir, out_dir = Path(recipes_dir), Path(out_dir)
    files = []
    for recipe_path in sorted(recipes_dir.glob('*.json')):
        recipe_text = recipe_path.read_text()
        recipe = json.loads(recipe_text)
        rows = expand(recipe, seed=seed)
        check = verify(rows, recipe)
        if check['rejected']:
            raise ValueError(f'{recipe_path.name}: rows the recipe does not produce: {check["rejected"][:5]}')
        for variant in sorted(recipe['variants']):
            chosen = [row for row in rows if row['variant'] == variant]
            path = out_dir / recipe['game'] / f'{variant}.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            text = ''.join(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n' for row in chosen)
            path.write_text(text)
            counts = {}
            for row in chosen:
                counts[row['split']] = counts.get(row['split'], 0) + 1
            files.append(dict(game=recipe['game'], variant=variant, tier='template',
                              path=str(path.relative_to(out_dir)),
                              sha256=hashlib.sha256(text.encode()).hexdigest(), rows=counts,
                              verify=dict(accepted=len(chosen), rejected=0),
                              recipe=str(recipe_path.relative_to(out_dir)),
                              recipe_sha256=hashlib.sha256(recipe_text.encode()).hexdigest()))
    manifest = dict(schema=1, generator='bin/game_corpus.py', seed=int(seed),
                    command=f'python bin/game_corpus.py freeze --seed {int(seed)} '
                            f'--recipes data/games/recipes --out data/games',
                    split_rule=('sha256 of the first source\'s normalized text modulo ten: 0 to 7 train, '
                                '8 validation, 9 test; a row with a transfer filler is transfer'),
                    files=files)
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
    return manifest


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Freeze the template-tier games.')
    parser.add_argument('command', choices=['freeze'])
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--recipes', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    for entry in freeze(args.recipes, args.out, seed=args.seed)['files']:
        print(entry['path'], entry['rows'])
