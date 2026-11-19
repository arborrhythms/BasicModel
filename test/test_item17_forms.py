"""A form's code is the set of its parts: a string that is no form, and the
way back of a form from the form alone."""
import json
import re
import subprocess
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.assessment_rung(1)

ROOT = Path(__file__).resolve().parents[1]
EXTRA = (b'this', b'his', b'is', b'at', b'cat', b'caat', b'act', b'here', b'there', b'the')


def lesson_words():
    """The words of the lesson's first stage, in order of first use."""
    stage = json.loads((ROOT/'data/corpus-lessons.json').read_text())['stages'][0]
    words = []
    for document in stage['documents']:
        for record in document['records']:
            for sentence in record['sentences']:
                words.extend(re.findall(r'[^\W\d_]+', sentence))
            words.append(record['answer'])
    return tuple(dict.fromkeys(word.encode() for word in words))


def lesson_store(extra=EXTRA, *, suffix=True):
    """The store with the lesson's words, ``cats`` by a suffix lesson, and
    ``extra``; ``cat`` and ``caat`` (and ``cat`` and ``act``) have to be
    told apart."""
    from Layers import RadixLayer
    from WordIdentity import WordIdentity
    store = RadixLayer(64, initial_cap=1024)
    identity = WordIdentity(store)
    for word in lesson_words() + tuple(extra):
        identity.observe_nested(word)
    if suffix:
        with identity.nested_lesson(b's'):
            identity.observe_nested(b'cat')
            identity.observe_nested(b'cats')
    return store, identity


def _inside(identity, part, whole):
    from ConceptualMatch import match
    found, _ = match(identity.form(whole)[None], identity.form(part)[None])
    return float(found[0])


def _reads_as(identity, string):
    return identity.read(identity.form(string))


def _state(store, identity):
    return dict(size=len(store), rows=dict(identity.word_rows), atoms=dict(identity.atoms),
                codes=store._basis.fixed_row_codes.detach().clone(),
                hash_map=dict(store.hash_map), keys=dict(identity.word_keys),
                forms=dict(identity.forms), trie=len(store.radix_trie),
                added=dict(identity.added), bits=dict(identity.bits))


def _equal_states(a, b):
    assert a.keys() == b.keys()
    for name in a:
        if torch.is_tensor(a[name]):
            assert torch.equal(a[name], b[name]), name
        else:
            assert a[name] == b[name], name


# Certificate 28: a string that is no form.

def test_a_string_that_is_no_form():
    store, identity = lesson_store((b'felix',) + EXTRA)
    assert b'felixx' not in identity.words
    # (a) With felix known and felixx not, the two have the same parts.
    assert set(identity.atoms_of(b'felixx')) == set(identity.words[b'felix'])
    read = {text: _reads_as(identity, text.encode()) for text in ('felixx', 'caat', 'xileF')}
    # (b) The part that tells the string from the word is admitted, not the string.
    part = identity.part_between(b'felixx', b'felix')
    assert part == 'xx'
    took = identity.admit_part(part)
    assert ('part', 'xx') in identity.atoms_of(b'felixx')
    assert ('part', 'xx') not in identity.words[b'felix']
    assert set(identity.words[b'felix']) < set(identity.atoms_of(b'felixx'))
    assert _inside(identity, b'felix', b'felixx') == 1.
    assert b'felixx' not in identity.words and b'felixx' not in identity.word_rows
    assert store.get_id(b'felixx') is None and b'felixx' not in store.hash_map
    assert _reads_as(identity, b'felixx') is None
    assert identity.key(identity.form(b'felixx')) not in identity.forms
    # (c) Admitting the part again changes nothing.
    before = _state(store, identity)
    assert identity.admit_part('xx') == ()
    assert identity.admit_part(identity.part_between(b'felixx', b'felix')) == ()
    _equal_states(before, _state(store, identity))
    print(dict(read_before_the_part={k: None if v is None else v.decode() for k, v in read.items()},
               part=part, forms_that_took_it=[w.decode() for w in took],
               felixx_after=_reads_as(identity, b'felixx'),
               felix_inside_felixx=_inside(identity, b'felix', b'felixx')))


def test_a_part_once_added_is_carried_by_every_form_that_holds_it():
    store, identity = lesson_store()
    assert 'aa' in identity.added
    later = identity.admit(b'baaz')
    assert identity.atoms[('part', 'aa')] in later
    for word, parts in identity.words.items():
        assert (('part', 'aa') in parts) == ('aa' in word.decode())
    # No stale key or trie entry: every held key names its own row.
    for word, key in identity.word_keys.items():
        row = identity.word_rows[word]
        assert store.hash_map[identity.WORD_PREFIX + key] == row
        assert store.radix_trie.get(identity.WORD_PREFIX + key) == row
    keys = [k for k in store.hash_map if k.startswith(identity.WORD_PREFIX)]
    assert len(keys) == len(identity.words)
    assert sum(1 for k in keys if store.radix_trie.get(k) is not None) == len(keys)


def test_a_form_may_hold_a_space():
    _, identity = lesson_store()
    identity.admit(b'ice cream')
    for part in (b'ice', b'cream', b' '):
        assert bool((identity.form(part) <= identity.form(b'ice cream')).all()), part
    assert ('part', ' ') in identity.words[b'ice cream']


def test_the_code_holds_no_end_marks_and_no_lengths():
    _, identity = lesson_store()
    # No block of lengths: the parts are spread over every coordinate of
    # the parts half (rule 50).
    assert identity.width == identity.pair_dim == 64
    used = torch.stack([identity.form(word) for word in identity.words]).amax(0)
    assert bool(used[32:].any())
    assert all(kind == 'part' for kind, _ in identity.atoms)
    assert torch.equal(identity.form(b'is'), identity.form(b'is'))
    assert _inside(identity, b'is', b'this') == 1.


# Certificate 29: the way back of a form.

def test_every_form_the_lesson_stores_is_put_back_from_the_form_alone():
    store, identity = lesson_store()
    for word in identity.words:
        assert identity.read(identity.form(word)) == word
        assert store.inverse_table[identity.word_rows[word]] == word
        assert identity.spelling(identity.parts(word)) == word
    # A code that is no stored form has no spelling.
    assert identity.spelling([identity.atoms[('part', 'F')], identity.atoms[('part', 'C')]]) is None

def test_the_lesson_model_puts_back_every_stored_form(monkeypatch):
    from reading_fixtures import use_eager_reading
    import Language
    from Models import BaseModel
    from data import TheData
    from corpus_lessons import stage_documents
    from Corpus import admit_training_forms
    use_eager_reading(monkeypatch)
    TheData.processLM({split: dict(text=['.'] if split == 'train' else [], label=[])
                      for split in ('train', 'validation', 'test')})
    Language.TheGrammar._configured = False
    model = BaseModel.from_config('data/MM_corpus_lessons.xml', data=TheData)[0]
    try:
        curriculum = json.loads((ROOT/'data/corpus-lessons.json').read_text())
        texts, targets = stage_documents(TheData, curriculum['stages'][0])
        admit_training_forms(model, texts, targets)
        store = model.perceptualSpace.percept_store
        identity = store.identity
        assert identity.words
        for word in identity.words:
            assert identity.read(identity.form(word)) == word
            assert store.inverse_table[identity.word_rows[word]] == word
        print(dict(forms=len(identity.words), added=dict(identity.added)))
    finally:
        model.End()


def test_the_pair_spellings_are_named_nowhere():
    named = subprocess.run(['git', 'grep', '-n', '-E', r'read_parts|assemble_pairs', '--', 'bin/'],
                           cwd=ROOT, capture_output=True, text=True).stdout
    assert named == ''
