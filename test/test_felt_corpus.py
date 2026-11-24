"""The frozen felt-world corpus is data: hashes, scenes, meanings and English hold.

Reads data/felt/manifest.json, the rows and the recipes it names (item 1.6
plan, section 14.6, checks 1 to 11). The corpus is never generated here and no
module of the model is imported. Everything this test knows of English (which
verb forms are singular, which words take the other article, the marks) is
read from data/felt/recipes/cast.json.
"""
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FELT = ROOT / 'data' / 'felt'
SIDE = 8
VALUES = range(1, 10)
SLOT = re.compile(r'\{(\w+)\}')
TOKEN = re.compile(r'[a-z]+|[^a-z\s]')


@lru_cache(maxsize=None)
def manifest():
    return json.loads((FELT / 'manifest.json').read_text())


@lru_cache(maxsize=None)
def cast():
    return json.loads((FELT / manifest()['cast']).read_text())


def entries():
    return manifest()['files']


@lru_cache(maxsize=None)
def _rows(path):
    return tuple(json.loads(line) for line in (FELT / path).read_text().splitlines() if line)


def rows(entry):
    return _rows(entry['path'])


@lru_cache(maxsize=None)
def _recipe(path):
    return json.loads((FELT / path).read_text())


def recipe(entry):
    return _recipe(entry['recipe'])


def frames(spec):
    out = {g['id']: g for g in spec['statements']}
    out.update({q['id']: q for q in spec.get('questions', ())})
    return out


def sentence(spec, record):
    frame = frames(spec)[record['frame']]
    return frame['sentences'][record.get('part', 0)] if 'sentences' in frame else frame


def things(scene):
    out = {}
    for x, y, z, steps in scene:
        out.setdefault((steps + 3) // 25, []).append((x, y, z, steps))
    return out


def reserved_step(d):
    return 25 * d + 1 if d % 2 else 25 * d - 1


def is_reserved_place(d, x, y):
    return (x + y + d) % 8 == 0


def text_digits(spec, frame, slots, value):
    names = SLOT.findall(frame['text'])
    named = [spec['forms'][n][0] for n in names] if frame['text'] else \
        [a for a in frame['args'] if isinstance(a, str)]
    out = {slots[n] for n in named if spec['slots'].get(n) == 'digit'}
    answer = frame.get('answer')
    if answer and answer['type'] == 'wh' and isinstance(value, int):
        out.add(value)
    return out


# 1 -------------------------------------------------------------------------

def test_every_file_has_the_hash_and_counts_of_the_manifest():
    data = manifest()
    assert data['schema'] == 1 and data['seed'] == 20261118
    assert hashlib.sha256((FELT / data['cast']).read_bytes()).hexdigest() == data['cast_sha256']
    assert sorted(e['category'] for e in entries()) == list(range(1, 28))
    for entry in entries():
        text = (FELT / entry['path']).read_text()
        assert hashlib.sha256(text.encode()).hexdigest() == entry['sha256'], entry['path']
        recipe_text = (FELT / entry['recipe']).read_text()
        assert hashlib.sha256(recipe_text.encode()).hexdigest() == entry['recipe_sha256']
        assert Counter(row['split'] for row in rows(entry)) == entry['rows'], entry['path']
        assert len({row['key'] for row in rows(entry)}) == len(rows(entry))
        assert len({json.dumps(row['entries'], sort_keys=True) for row in rows(entry)}) == len(rows(entry))
    for d in VALUES:
        held = data['reserved'][str(d)]
        assert held['step'] == reserved_step(d)
        assert sorted(map(tuple, held['places'])) == sorted(
            (x, y) for x in range(SIDE) for y in range(SIDE) if is_reserved_place(d, x, y))


# 2 -------------------------------------------------------------------------

def connected(places):
    places = set(places)
    seen, todo = set(), [next(iter(places))]
    while todo:
        x, y = todo.pop()
        if (x, y) in seen:
            continue
        seen.add((x, y))
        todo.extend(p for p in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)) if p in places)
    return seen == places


def test_every_scene_is_a_valid_image_of_level_sets():
    for entry in entries():
        for row in rows(entry):
            general_reserved = row['category'] == 2 and row['split'] == 'generalization'
            for item in row['entries']:
                scene = item['scene']
                if scene is None:
                    continue
                assert isinstance(scene, list)
                assert len({(x, y, z) for x, y, z, _ in scene}) == len(scene), row['key']
                for x, y, z, steps in scene:
                    assert 0 <= x < SIDE and 0 <= y < SIDE and z == 0, row['key']
                    d = (steps + 3) // 25
                    assert d in VALUES and abs(steps - 25 * d) <= 3, row['key']
                for d, places in things(scene).items():
                    assert connected([(x, y) for x, y, _, _ in places]), row['key']
                    for x, y, _, steps in places:
                        held = steps == reserved_step(d) and is_reserved_place(d, x, y)
                        if general_reserved:
                            assert held, row['key']
                        else:
                            assert steps != reserved_step(d) and not is_reserved_place(d, x, y), row['key']


# 3 -------------------------------------------------------------------------

def test_every_row_is_one_its_recipe_produces_and_its_meanings_hold():
    from felt_world_corpus import verify
    for entry in entries():
        check = verify(list(rows(entry)), recipe(entry), cast())
        assert check['rejected'] == [] and check['accepted'] == len(rows(entry)), entry['path']
        for row in rows(entry):
            statements = [m for m in row['meaning'] if 'value' not in m]
            if row['split'] == 'witness':
                assert [m['truth'] for m in statements].count(False) == 1, row['key']
            else:
                assert all(m['truth'] is True for m in statements), row['key']


def test_verify_rejects_a_row_changed_in_its_answer_its_text_or_its_image():
    from felt_world_corpus import verify
    tampered = 0
    for entry in entries():
        spec = recipe(entry)
        for row in rows(entry)[:30]:
            if row['split'] == 'inference':
                continue
            if row['answer']:
                bad = copy.deepcopy(row)
                bad['answer'] = 'x' + bad['answer']
                assert verify([bad], spec, cast())['rejected'] == [row['key']]
                tampered += 1
            texts = [i for i, e in enumerate(row['entries']) if e['text']]
            if texts:
                bad = copy.deepcopy(row)
                bad['entries'][texts[0]]['text'] += ' x'
                assert verify([bad], spec, cast())['rejected'] == [row['key']]
            judged = [m for m in row['meaning'] if row['entries'][m['entry']]['scene']]
            if judged and row['category'] not in (1,):
                m = judged[0]
                bad = copy.deepcopy(row)
                for item in bad['entries']:
                    if item['scene'] is not None:
                        item['scene'] = []
                if verify([bad], spec, cast())['rejected'] == [row['key']]:
                    tampered += 1
    assert tampered > 500


# 4 -------------------------------------------------------------------------

def test_every_word_is_declared_no_text_has_a_digit_and_the_longest_text_is_recorded():
    vocabulary = set(cast()['vocabulary'])
    assert manifest()['vocabulary'] == sorted(vocabulary)
    marks = {cast()['english']['statement_mark'], cast()['english']['question_mark'],
             *cast()['english']['joined_marks']}
    longest_words = longest_bytes = 0
    for entry in entries():
        for row in rows(entry):
            texts = [e['text'] for e in row['entries']] + [row['answer'] or '']
            for text in texts:
                assert not re.search(r'\d', text), row['key']
                for token in TOKEN.findall(text):
                    assert token in vocabulary or token in marks, (row['key'], token)
            for e in row['entries']:
                longest_words = max(longest_words, len(e['text'].split()))
                longest_bytes = max(longest_bytes, len(e['text'].encode()))
    assert manifest()['longest_text'] == dict(words=longest_words, bytes=longest_bytes)


# 5 -------------------------------------------------------------------------

def subject(tokens, i, boundaries):
    j = i - 1
    while j >= 0 and tokens[j] not in boundaries:
        j -= 1
    return tokens[j + 1:i]


def test_the_english_a_rule_can_check():
    en = cast()['english']
    singular, plural = set(en['singular_verbs']), set(en['plural_verbs'])
    plural_nouns = set(en['plural_nouns'])
    stop, ask = en['statement_mark'], en['question_mark']
    marks = {stop, ask, *en['joined_marks']}
    boundaries = marks | set(en['clause_words']) | set(en['verbs'])
    for entry in entries():
        for row in rows(entry):
            questions = {row['question']} if row['question'] is not None else set()
            for index, item in enumerate(row['entries']):
                text = item['text']
                if not text:
                    continue
                assert text == text.lower() and text == text.strip() and '  ' not in text, row['key']
                tokens = TOKEN.findall(text)
                terminal = [t for t in tokens if t in (stop, ask)]
                assert len(terminal) == 1 and tokens[-1] == terminal[0], row['key']
                if tokens[-1] == ask:
                    assert text.endswith(' ' + ask) and index in questions, row['key']
                else:
                    assert not text.endswith(' ' + stop) and index not in questions, row['key']
                for mark in en['joined_marks']:
                    assert ' ' + mark not in text, row['key']
                assert all(a != b for a, b in zip(tokens, tokens[1:])), row['key']
                for i, t in enumerate(tokens):
                    if t == en['article_before_vowel']:
                        assert tokens[i + 1] in en['vowel_words'], row['key']
                    if t == en['article']:
                        assert tokens[i + 1] not in en['vowel_words'], row['key']
                    if t in singular:
                        assert not set(subject(tokens, i, boundaries)) & plural_nouns, (row['key'], text)
                    if t in plural:
                        said = subject(tokens, i, boundaries - {en['conjunction']})
                        assert en['conjunction'] in said or set(said) & plural_nouns, (row['key'], text)


# 6 -------------------------------------------------------------------------

def test_held_out_casts_stay_out_and_train_holds_every_digit_frame_and_count():
    pairs = [tuple(p) for p in manifest()['held_out_pairs']]
    for entry in entries():
        spec = recipe(entry)
        order = [g['id'] for g in spec['statements']] + [q['id'] for q in spec.get('questions', ())]
        rule = spec['held_out']
        digits, ids, counts = set(), set(), set()
        for row in rows(entry):
            held = False
            for record in row['meaning']:
                frame = sentence(spec, record)
                named = text_digits(spec, frame, record['slots'], record.get('value'))
                assert sorted(named) == record['cast'], row['key']
                if rule == 'pairs':
                    held |= any(p in named and q in named for p, q in pairs)
                elif rule == 'frame_digit':
                    held |= any((order.index(record['frame']) + 1 + d) % 4 == 0 for d in named)
                elif rule == 'sums':
                    held |= frame['predicate'] == 'sum' and [record['slots']['a'], record['slots']['b']] in spec['sum_pairs']
            if rule == 'count_sets':
                held = sum(things(row['entries'][0]['scene'])) % 4 == 0
            if row['split'] in ('train', 'validation', 'test', 'witness'):
                assert not held, row['key']
            if row['split'] == 'generalization' and rule != 'reserved':
                assert held, row['key']
            if row['split'] == 'train':
                ids.add(row['variant'].split('+')[0])
                ids.update(row['variant'].split('+')[1:])
                for record in row['meaning']:
                    digits.update(record['cast'])
                for item in row['entries']:
                    if item['scene'] is not None:
                        digits.update(things(item['scene']))
                if rule == 'count_sets':
                    counts.add(len(things(row['entries'][0]['scene'])))
        assert digits == set(VALUES), entry['path']
        wanted = {g['id'] for g in spec['statements'] if not g.get('inference_only')}
        if spec.get('compose') != 'statement':
            wanted |= {q['id'] for q in spec.get('questions', ()) if not q.get('inference_only')}
        assert wanted <= ids, (entry['path'], wanted - ids)
        if rule == 'count_sets':
            assert counts == set(VALUES)
        if spec['category'] == 4:
            stated = {r['slots']['a'] for row in rows(entry) if row['split'] == 'train'
                      for r in row['meaning'] if r['predicate'] == 'member' and 'value' not in r}
            listed = {r['slots']['k'] for row in rows(entry) if row['split'] == 'train'
                      for r in row['meaning'] if r['predicate'] == 'kind_members'}
            assert stated == set(VALUES) or listed == {k['key'] for k in cast()['kinds']}


# 7 -------------------------------------------------------------------------

def test_no_image_exactly_where_nothing_is_seen_and_inference_follows_from_what_is_said():
    for entry in entries():
        spec = recipe(entry)
        for row in rows(entry):
            imageless = {m['entry'] for m in row['meaning'] if sentence(spec, m).get('image') is False}
            for index, item in enumerate(row['entries']):
                none = (row['split'] == 'inference' or spec.get('world') == 'none' or index in imageless)
                assert (item['scene'] is None) == none, row['key']
            if row['split'] != 'inference':
                assert row['inference'] is None
                continue
            asked = [m for m in row['meaning'] if 'value' in m]
            said = [m for m in row['meaning'] if 'value' not in m]
            assert len(asked) == 1 and said
            q = asked[0]
            meant = [q['value'] if a == '?' else a for a in q['args']]
            assert all((s['predicate'], s['args']) != (q['predicate'], meant) for s in said), row['key']
            assert all((s['predicate'], s['args']) != (q['predicate'], q['args']) for s in said), row['key']
            if row['inference'] == 'open':
                assert row['answer'] == '' and q['follows'] is None and q['value'] is None
            else:
                assert row['answer'] and q['follows'] is not None
    kinds = {e['category']: Counter(r['inference'] for r in rows(e) if r['split'] == 'inference')
             for e in entries() if 'inference' in e['rows']}
    assert sorted(kinds) == [6, 11, 14, 19, 24, 25]
    assert kinds[24] == Counter(rule=16, open=4)
    assert set(kinds[19]) == {'time'} and set(kinds[25]) == {'kind'}


# 8 -------------------------------------------------------------------------

def test_yes_and_no_and_first_and_second_are_balanced_in_every_split():
    truth = cast()['truth_words']
    for entry in entries():
        spec = recipe(entry)
        enumerated = [truth] + [q['answer']['words'] for q in spec.get('questions', ()) if 'words' in q['answer']]
        for split in entry['rows']:
            answers = Counter(row['answer'] for row in rows(entry) if row['split'] == split)
            for pair in enumerated:
                total = sum(answers[w] for w in pair)
                if total:
                    for word in pair:
                        assert 0.4 <= answers[word] / total <= 0.6, (entry['path'], split, dict(answers))


# 9 -------------------------------------------------------------------------

def test_every_contrast_set_has_twelve_rows_that_turn_on_the_exchange():
    found = 0
    for entry in entries():
        sets = {}
        for row in rows(entry):
            if row['split'] == 'contrast':
                sets.setdefault(row['set'], []).append(row)
        if not sets:
            continue
        spec = recipe(entry)
        assert spec['category'] == 14 and len(sets) == 20
        held = {frozenset(p) for p in manifest()['held_out_pairs']}
        kinds = Counter()
        for key, group in sets.items():
            found += 1
            assert len(group) == 12
            layouts = []
            for row in group:
                assert len(row['entries']) == 1 and row['question'] == 0
                layout = {d: (p[0][0], p[0][1]) for d, p in things(row['entries'][0]['scene']).items()}
                assert len(layout) == 2
                layouts.append(layout)
            pair = set(layouts[0])
            kinds[frozenset(pair) in held] += 1
            distinct = []
            for layout in layouts:
                if layout not in distinct:
                    distinct.append(layout)
            assert len(distinct) == 4 and all(layouts.count(l) == 3 for l in distinct)
            a, b = sorted(pair)
            first, swapped, moved, moved_swapped = distinct
            assert swapped == {a: first[b], b: first[a]}
            dx, dy = moved[a][0] - first[a][0], moved[a][1] - first[a][1]
            assert (dx, dy) != (0, 0)
            assert moved == {d: (p[0] + dx, p[1] + dy) for d, p in first.items()}
            assert moved_swapped == {d: (p[0] + dx, p[1] + dy) for d, p in swapped.items()}
            assert first[a][0] == first[b][0]
            answers = {}
            for row, layout in zip(group, layouts):
                record = row['meaning'][0]
                answers[(distinct.index(layout), record['frame'], tuple(record['args']))] = row['answer']
            assert len(answers) == 12
            for (config, fid, args), word in answers.items():
                # exchanging the two things turns every answer
                assert answers[(config ^ 1, fid, args)] != word
                # "is {b} below {a} ?" has the answer of "is {a} above {b} ?"
                if fid == 'q06':
                    assert answers[(config, 'q03', args)] == word
        assert kinds == Counter({False: 10, True: 10})
    assert found == 20


# 10 ------------------------------------------------------------------------

def test_counting_answers_are_the_first_counting_words():
    counting = cast()['counting']
    entry = next(e for e in entries() if e['category'] == 27)
    for row in rows(entry):
        assert len(row['entries']) == 1 and row['entries'][0]['text'] == '' and row['question'] == 0
        n = len(things(row['entries'][0]['scene']))
        assert row['answer'] == ' '.join(counting[:n])
        assert all(len(p) == 1 for p in things(row['entries'][0]['scene']).values())


# 11 ------------------------------------------------------------------------

def test_load_returns_documents_with_no_model_built():
    script = (
        'import sys, felt_world_corpus as F\n'
        'docs = F.load(14, "contrast")\n'
        'assert len(docs) == 240 and docs[0].scenes[0] is not None\n'
        'd = F.load(6, "inference")[0]\n'
        'assert d.scenes[d.question] is None and d.sentences[d.question].endswith("?")\n'
        'counts = {s: len(F.load(3, s)) for s in ("train", "validation", "test")}\n'
        'assert counts == {"train": 100, "validation": 20, "test": 20}, counts\n'
        'banned = [m for m in ("torch", "Language", "Models", "Spaces", "util") if m in sys.modules]\n'
        'assert not banned, banned\n'
        'print("ok")\n')
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'bin'))
    out = subprocess.run([sys.executable, '-c', script], env=env, capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0 and out.stdout.strip() == 'ok', out.stderr
    from felt_world_corpus import load
    for entry in entries():
        for split, count in entry['rows'].items():
            docs = load(entry['category'], split)
            assert len(docs) == count
            for doc, row in zip(docs, [r for r in rows(entry) if r['split'] == split]):
                assert doc.sentences == tuple(e['text'] for e in row['entries'])
                assert doc.question == row['question'] and doc.answer == row['answer']
                assert len(doc.scenes) == len(doc.sentences)
