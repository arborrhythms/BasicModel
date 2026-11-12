"""The ten lesson stages are frozen data the lesson driver can read.

Reads data/lessons/manifest.json, parts-of-speech.json and the recipes
(item 0.5 plan, section 9); never regenerates them.
"""
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
LESSONS = ROOT / 'data' / 'lessons'


def load():
    return json.loads((LESSONS / 'parts-of-speech.json').read_text())


def recipes():
    return {json.loads(p.read_text())['phase']: json.loads(p.read_text())
            for p in sorted((LESSONS / 'recipes').glob('*.json'))}


def words(records):
    found = set()
    for record in records:
        for sentence in record['sentences']:
            found.update(re.findall(r'[A-Za-z]+', sentence))
    return found


def test_the_file_matches_its_manifest():
    manifest = json.loads((LESSONS / 'manifest.json').read_text())
    text = (LESSONS / manifest['file']).read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == manifest['sha256']
    for name, digest in manifest['recipes'].items():
        assert hashlib.sha256((LESSONS / 'recipes' / name).read_bytes()).hexdigest() == digest
    data = load()
    assert [s['name'] for s in manifest['stages']] == [s['name'] for s in data['stages']]
    for entry, stage in zip(manifest['stages'], data['stages']):
        assert entry['records'] == sum(len(d['records']) for d in stage['documents'])
        assert entry['heldout'] == sum(len(d['records']) for d in stage['heldout'])


def test_stages_are_the_registered_phases_in_lesson_order():
    from lesson_stages import PHASES
    stages = load()['stages']
    assert [s['name'] for s in stages] == list(PHASES)
    assert [s['lesson'] for s in stages] == list(range(10))
    registry = json.loads((ROOT / 'test' / 'fixtures' / 'phase-artifacts.json').read_text())['phases']
    assert set(PHASES) <= set(registry)
    for stage in stages:
        assert stage['criterion']['measure'] and stage['criterion']['threshold'] is None


def test_records_are_documents_with_answers_from_their_own_sentences():
    from math_chain_corpus import ChainDocument
    for stage in load()['stages']:
        for part in ('documents', 'heldout'):
            for document in stage[part]:
                for record in document['records']:
                    doc = ChainDocument(**dict(record, sentences=tuple(record['sentences'])))
                    if doc.question is None:
                        continue
                    assert 0 <= doc.question < len(doc.sentences)
                    context = ' '.join(doc.sentences[:doc.question])
                    assert doc.answer == 'yes' or doc.answer in re.findall(r'[A-Za-z]+', context)


def test_held_out_words_stay_out_of_training():
    by_phase = recipes()
    for stage in load()['stages']:
        held = {form for entries in by_phase[stage['name']].get('heldout', {}).values()
                for filler in entries for form in filler.values()}
        train = [r for d in stage['documents'] for r in d['records']]
        heldout = [r for d in stage['heldout'] for r in d['records']]
        assert train and heldout
        assert not words(train) & held
        assert all(words([r]) & held for r in heldout)


def test_lesson_words_and_game_words_are_apart():
    games = ROOT / 'data' / 'games' / 'recipes'
    vocabulary = set(load()['vocabulary'])
    for path in games.glob('*.json'):
        recipe = json.loads(path.read_text())
        in_game = {form for entries in recipe['fillers'].values() for f in entries for form in f.values()}
        assert not in_game & vocabulary, (path.name, in_game & vocabulary)


def test_the_lesson_driver_reads_every_stage():
    from corpus_lessons import stage_documents
    for stage in load()['stages']:
        for part, split in (('documents', 'train'), ('heldout', 'validation')):
            data = SimpleNamespace(text_answers=None, source_addresses={})
            texts, targets = stage_documents(data, dict(stage, documents=stage[part]), split=split)
            assert texts and len(texts) == len(targets)
