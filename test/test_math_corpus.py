"""The frozen graded math corpus is data: hashes, splits and answers hold.

Reads data/math/manifest.json and the rows it names; never regenerates the
corpus (item 0.5 plan, section 4 rule 5). Answers are checked by a small
evaluator of the rows' own sentences, an oracle at the data boundary only.
"""
import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MATH = ROOT / 'data' / 'math'
SPLITS = ('train', 'validation', 'test', 'beyond')


def manifest():
    return json.loads((MATH / 'manifest.json').read_text())


def rows(grade):
    return [json.loads(line) for line in (MATH / grade['file']).read_text().splitlines() if line]


def test_files_match_their_manifest_hashes_and_counts():
    data = manifest()
    assert data['schema'] == 1 and isinstance(data['seed'], int)
    for grade in data['grades']:
        text = (MATH / grade['file']).read_text()
        assert hashlib.sha256(text.encode()).hexdigest() == grade['sha256'], grade['file']
        counts = {}
        for row in rows(grade):
            assert row['split'] in SPLITS
            counts[row['split']] = counts.get(row['split'], 0) + 1
        assert counts == grade['rows'], grade['file']


def test_grades_follow_the_corrected_curriculum_order():
    from math_chain_corpus import GRADES
    names = [grade['name'] for grade in manifest()['grades']]
    assert names == [name for name, _ in GRADES]
    assert names[:5] == ['counting', 'identity', 'alias', 'plus_zero', 'plus_one']


def test_rows_are_documents_of_number_words_only():
    from math_chain_corpus import ChainDocument, NUMBER_WORDS
    words = set(NUMBER_WORDS) | {'x', 'y', 'z', 'w', 'u', 'v', 'is', 'what', 'plus', 'the', 'answer', '?'}
    for grade in manifest()['grades']:
        for row in rows(grade):
            document = ChainDocument(**row['document'])
            for sentence in document.sentences:
                text = re.sub(r'\bRef\(\d+\)', '', sentence)
                assert not re.search(r'\d', text), sentence
                tokens = re.findall(r"[A-Za-z]+|\?", text.lower())
                assert set(tokens) <= words, (sentence, set(tokens) - words)


def evaluate(document):
    """Values of the names, from the premises; None for a counting fact."""
    from math_chain_corpus import NUMBER_WORDS
    value = {word: index for index, word in enumerate(NUMBER_WORDS)}
    names = {}
    pending = [s for s in document.sentences[:document.question]]
    while pending:
        progressed = False
        for sentence in list(pending):
            match = re.fullmatch(r'(\w+) is (\w+)(?: plus (\w+))?\.', sentence)
            assert match, sentence
            name, operand, addend = match.groups()
            base = value.get(operand, names.get(operand))
            if base is None:
                continue
            names[name] = base + (value[addend] if addend else 0)
            pending.remove(sentence)
            progressed = True
        assert progressed, pending
    return names


def test_every_answer_and_intermediate_answer_is_right():
    from math_chain_corpus import ChainDocument, NUMBER_WORDS
    for grade in manifest()['grades']:
        for row in rows(grade):
            document = ChainDocument(**row['document'])
            if document.question is None:
                continue
            names = evaluate(document)
            asked = re.fullmatch(r'what is (\w+) \?', document.sentences[document.question])
            assert asked and NUMBER_WORDS[names[asked.group(1)]] == document.answer, document.key
            for index, answer in document.questions:
                step = re.fullmatch(r'what is (\w+) plus one \?', document.sentences[index])
                assert step and NUMBER_WORDS.index(step.group(1)) + 1 == NUMBER_WORDS.index(answer)
            if row['split'] == 'train':
                assert document.sentences[-1] == f'the answer is {document.answer}.'
            else:
                assert not document.questions and document.sentences[-1].startswith('what is')


def test_held_out_and_beyond_pairs_never_reach_training():
    data = manifest()
    held = {tuple(p) for p in data['held_out_pairs']}
    beyond = {tuple(p) for p in data['beyond_pairs']}
    for grade in data['grades']:
        for row in rows(grade):
            pair = tuple(row['document']['pair'] or ())[:2]
            if grade['name'] in ('plus_k', 'distractors') and pair in held:
                assert row['split'] in ('validation', 'test')
            if grade['name'] in ('plus_k', 'distractors') and pair in beyond:
                assert row['split'] == 'beyond'
            if grade['name'] == 'depth_three':
                assert row['split'] == 'test'
    assert all(row['split'] == 'train' for row in rows(data['grades'][0]))
