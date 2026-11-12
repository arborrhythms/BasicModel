"""The frozen template-tier games are data: hashes, splits and pairs hold.

Reads data/games/manifest.json, the rows and the recipes it names (item 0.5
plan, section 6). The corpus is never written here; ``verify`` checks each
frozen row against its recipe.
"""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GAMES = ROOT / 'data' / 'games'


def manifest():
    return json.loads((GAMES / 'manifest.json').read_text())


def rows(entry):
    return [json.loads(line) for line in (GAMES / entry['path']).read_text().splitlines() if line]


def recipe(entry):
    return json.loads((GAMES / entry['recipe']).read_text())


def lesson_words():
    words = set()
    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == 'sentences':
                    for sentence in item:
                        words.update(re.findall(r'[A-Za-z]+', sentence))
                elif key == 'answer' and isinstance(item, str):
                    words.add(item)
                else:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    walk(json.loads((ROOT / 'data' / 'corpus-lessons.json').read_text()))
    return words


def test_files_match_their_manifest_hashes_counts_and_recipes():
    data = manifest()
    assert data['schema'] == 1 and isinstance(data['seed'], int)
    assert {entry['game'] for entry in data['files']} == {'not', 'conjunction', 'unary_chain'}
    for entry in data['files']:
        text = (GAMES / entry['path']).read_text()
        assert hashlib.sha256(text.encode()).hexdigest() == entry['sha256'], entry['path']
        recipe_text = (GAMES / entry['recipe']).read_text()
        assert hashlib.sha256(recipe_text.encode()).hexdigest() == entry['recipe_sha256']
        counts = {}
        for row in rows(entry):
            counts[row['split']] = counts.get(row['split'], 0) + 1
        assert counts == entry['rows']


def test_every_frozen_row_is_a_pair_its_recipe_produces():
    from game_corpus import verify
    for entry in manifest()['files']:
        check = verify(rows(entry), recipe(entry))
        assert check['rejected'] == [] and check['accepted'] == sum(entry['rows'].values())


def test_a_source_keeps_its_split_and_transfer_words_never_enter_a_game():
    from game_corpus import content_split
    seen = {}
    for entry in manifest()['files']:
        spec = recipe(entry)
        transfer = {form for entries in spec.get('transfer', {}).values() for filler in entries
                    for form in filler.values()}
        for row in rows(entry):
            words = set(re.findall(r'[A-Za-z]+', ' '.join(row['sources']) + ' ' + row['target']))
            if row['split'] == 'transfer':
                assert words & transfer, row['key']
                continue
            assert not words & transfer, row['key']
            assert row['split'] == content_split(row['sources'][0])
            first = ' '.join(row['sources'][0].lower().split())
            assert seen.setdefault(first, row['split']) == row['split']


def test_game_fillers_are_new_words_and_transfer_fillers_are_lesson_words():
    known = lesson_words()
    for entry in manifest()['files']:
        spec = recipe(entry)
        for entries in spec['fillers'].values():
            for filler in entries:
                assert not set(filler.values()) & known, filler
        for entries in spec.get('transfer', {}).values():
            for filler in entries:
                assert set(filler.values()) & known, filler


def test_the_expander_knows_no_language():
    from game_corpus import expand, verify
    toy = {'game': 'not', 'fillers': {'a': [{'f': 'ka'}, {'f': 'po'}], 'b': [{'g': 'zu'}]},
           'transfer': {'a': [{'f': 'mi'}]},
           'variants': {'v': {'slots': {'x': 'a', 'y': 'b'}, 'sources': ['{x.f} {y.g}.'],
                              'target': '{x.f} lo {y.g}.'}}}
    out = expand(toy, seed=0)
    assert sorted((row['sources'][0], row['target']) for row in out) == [
        ('ka zu.', 'ka lo zu.'), ('mi zu.', 'mi lo zu.'), ('po zu.', 'po lo zu.')]
    assert [row['split'] for row in out if row['sources'][0] == 'mi zu.'] == ['transfer']
    assert verify(out, toy)['rejected'] == []


def test_written_pairs_preserve_authorship_and_an_audit_sample():
    found = {}
    for entry in manifest()['files']:
        if entry['tier'] != 'written':
            continue
        spec = recipe(entry)
        provenance = entry['provenance']
        assert provenance['model'] and provenance['date']
        assert provenance['prompt_sha256'] == hashlib.sha256(spec['provenance']['prompt'].encode()).hexdigest()
        assert set(provenance['audit_keys']) <= {row['key'] for row in spec['authored_rows']}
        assert provenance['audit_status'] == 'pending'
        assert rows(entry) == [row for row in spec['authored_rows'] if row['variant'] == entry['variant']]
        assert entry['verify'] == dict(accepted=len(rows(entry)), rejected=0)
        found[entry['game']] = found.get(entry['game'], 0) + len(rows(entry))
    assert found == {'not': 12, 'conjunction': 12, 'unary_chain': 12}
