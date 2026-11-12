"""The training corpus manifest, its held-out list and their disjointness.

Default tier: reads data/training-corpus.json and the tracked data it names
(item 0.5 plan, section 8). Local part B data is checked by hash only where
it is present; nothing is downloaded or regenerated.
"""
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def manifest():
    return json.loads((ROOT / 'data/training-corpus.json').read_text())


def test_stages_are_in_order_and_part_a_files_match_their_hashes():
    from training_corpus import sha256_file
    data = manifest()
    assert data['order'] == ['A1', 'A2', 'A3', 'B1', 'B2', 'B3']
    assert data['stages']['B3']['prepared'] is False
    for name in ('A1', 'A2', 'A3'):
        for entry in data['stages'][name]['files']:
            assert sha256_file(ROOT / entry['path']) == entry['sha256'], entry['path']
    assert sha256_file(ROOT / data['stages']['B1']['recipe']['path']) == data['stages']['B1']['recipe']['sha256']


def test_no_held_out_item_is_a_training_item():
    from training_corpus import HELD, units, unit_text
    items = units(ROOT)
    trained = {unit_text(s) for kind, split, key, s in items if split not in HELD}
    held = [(key, unit_text(s)) for kind, split, key, s in items if split in HELD]
    assert held and not [key for key, text in held if text in trained]


def test_the_held_out_cost_set_is_the_union_of_part_a_validation_splits():
    from training_corpus import VALIDATION, units
    data = manifest()
    entry = next(e for e in data['heldout'] if e['source'] == 'data/eval/heldout-cost.json')
    cost = json.loads((ROOT / entry['source']).read_text())
    assert cost['sha256'] == entry['sha256']
    assert hashlib.sha256(json.dumps(cost['items'], sort_keys=True).encode()).hexdigest() == cost['sha256']
    wanted = sorted((kind, key) for kind, split, key, s in units(ROOT) if split == VALIDATION[kind])
    assert sorted((item['kind'], item['key']) for item in cost['items']) == wanted


def test_the_next_word_gate_reads_only_test_documents():
    from training_corpus import sha256_file
    import sys
    sys.path.insert(0, str(ROOT / 'bin'))
    from eval_nanochat_grammar import split_name
    data = manifest()
    entry = next(e for e in data['heldout'] if e['name'] == 'next-word gate')
    assert sha256_file(ROOT / entry['source']) == entry['sha256']
    gate = json.loads((ROOT / entry['source']).read_text())
    assert gate['metadata']['heldout_split'] == 'test'
    assert gate['metadata']['split_rule'] == 'document_mod10_8_1_1'
    assert all(split_name(item['doc_index']) == 'test' for item in gate['items'])


def test_the_recorded_part_b_scan_is_of_the_current_held_out_set():
    from training_corpus import held_out_sentences, units
    scan = manifest()['part_b_scan']
    sentences = held_out_sentences(units(ROOT))
    assert scan['held_out_sentences'] == len(sentences)
    assert scan['held_out_digest'] == hashlib.sha256('\n'.join(sentences).encode()).hexdigest()


def test_local_part_b_data_matches_its_recorded_hashes():
    from training_corpus import sha256_file
    stages = manifest()['stages']
    checks = [(ROOT / 'data/corpus/assembly-01/manifest.json', stages['B1']['manifest_sha256']),
              (ROOT / 'data/fineweb/shard_00000.parquet', stages['B2']['shard_sha256'])]
    present = [(path, digest) for path, digest in checks if path.exists()]
    if not present:
        pytest.skip('part B is local data and is not present in this checkout')
    for path, digest in present:
        assert sha256_file(path) == digest, path
