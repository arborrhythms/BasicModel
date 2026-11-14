"""Stage data of the ten parts-of-speech lessons (item 0.5 plan, section 9).

Each recipe (data/lessons/recipes/<nn>-<phase>.json) holds record frames with
slots, an optional question index and answer frame, filler classes, and a
held-out list. Expansion reuses the games' slot substitution, so no word of
any language is held here. A record with a held-out filler goes to the
stage's ``heldout`` documents, at most a quarter of a variant's training
records; every other record to ``documents``.
"""
import hashlib
import json
import random
from pathlib import Path

from game_corpus import _assignments, copy_recipes, render

PHASES = ('words', 'interpret', 'clause', 'ask', 'verb', 'modifiers',
          'determiners', 'adverb', 'prepositions', 'relations')


def expand(recipe, *, seed):
    rng = random.Random(seed)
    shim = dict(recipe, game=recipe['phase'], transfer=recipe.get('heldout', {}))
    train, heldout = [], []
    for variant in sorted(recipe['variants']):
        frame = recipe['variants'][variant]
        found = []
        for assignment, held in _assignments(shim, variant):
            record = dict(sentences=[render(text, assignment) for text in frame['sentences']])
            if 'question' in frame:
                record['question'] = int(frame['question'])
                record['answer'] = render(frame['answer'], assignment)
            found.append((json.dumps(record, sort_keys=True), held))
        kept = sorted(text for text, held in found if not held)
        held = sorted(text for text, held in found if held)
        limit = frame.get('limit')
        if limit is not None and len(kept) > int(limit):
            kept = sorted(rng.sample(kept, int(limit)))
        # Held-out records are at most a quarter of the variant's training
        # records, drawn with the recorded seed; at least one is kept.
        cap = max(1, len(kept) // 4)
        if len(held) > cap:
            held = sorted(rng.sample(held, cap))
        for part, texts in (('train', kept), ('heldout', held)):
            for index, text in enumerate(texts):
                record = dict(json.loads(text), key=f"{recipe['phase']}-{variant}-{part}-{index:04d}")
                (train if part == 'train' else heldout).append(record)
    def documents(records):
        return [dict(key=record['key'], document=i, records=[record]) for i, record in enumerate(records)]
    return dict(name=recipe['phase'], lesson=int(recipe['lesson']), criterion=recipe['criterion'],
                measure=list(recipe.get('measure', ())), documents=documents(train),
                heldout=documents(heldout))


def vocabulary(stages):
    import re
    words = set()
    for stage in stages:
        for part in ('documents', 'heldout'):
            for document in stage[part]:
                for record in document['records']:
                    for sentence in record['sentences']:
                        words.update(re.findall(r"[A-Za-z]+", sentence))
    return sorted(words)


def freeze(recipes_dir, out_dir, *, seed):
    """Write parts-of-speech.json and manifest.json; tests never call this."""
    recipes_dir, out_dir = Path(recipes_dir), Path(out_dir)
    recipe_paths = copy_recipes(recipes_dir, out_dir)
    recipes = [json.loads(path.read_text()) for path in recipe_paths]
    if [recipe['phase'] for recipe in recipes] != list(PHASES):
        raise ValueError('the recipes must be the ten phases in lesson order')
    stages = [expand(recipe, seed=seed) for recipe in recipes]
    data = dict(version=2, source='item 0.5, plan section 9', vocabulary=vocabulary(stages),
                stages=stages)
    text = json.dumps(data, indent=1, sort_keys=True) + '\n'
    path = out_dir / 'parts-of-speech.json'
    path.write_text(text)
    manifest = dict(schema=1, generator='bin/lesson_stages.py', seed=int(seed),
                    command=f'python bin/lesson_stages.py freeze --seed {int(seed)} '
                            f'--recipes data/lessons/recipes --out data/lessons',
                    file=path.name, sha256=hashlib.sha256(text.encode()).hexdigest(),
                    recipes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in recipe_paths},
                    stages=[dict(name=s['name'], lesson=s['lesson'],
                                 records=sum(len(d['records']) for d in s['documents']),
                                 heldout=sum(len(d['records']) for d in s['heldout'])) for s in stages])
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
    return manifest


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Freeze the ten lesson stages.')
    parser.add_argument('command', choices=['freeze'])
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--recipes', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    for stage in freeze(args.recipes, args.out, seed=args.seed)['stages']:
        print(stage)
