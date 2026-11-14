"""Model-free image data certificates; §22 explicitly asks for freeze reproduction."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pytest
import torch

import image_corpus
from local_data import require_mnist

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / 'data/images'
COUNTS = {'train': 100, 'validation': 20, 'test': 20}


def rows(split):
    return [json.loads(line) for line in (IMAGES / f'{split}.jsonl').read_text().splitlines()]


def decode(encoded):
    # This oracle does not use the builder's NumPy pack/unpack path.
    return np.array([int(bit) for bit in f'{int(encoded, 16):0784b}'], dtype=np.uint8).reshape(28, 28)


def test_freeze_reproduces_every_generated_file_byte_for_byte(tmp_path):
    manifest = json.loads((IMAGES / 'manifest.json').read_text())
    image_corpus.freeze(tmp_path / 'nested/images', seed=manifest['seed'])
    expected = {'manifest.json', 'train.jsonl', 'validation.jsonl', 'test.jsonl'}
    assert {path.name for path in (tmp_path / 'nested/images').iterdir()} == expected
    for name in expected:
        assert (tmp_path / 'nested/images' / name).read_bytes() == (IMAGES / name).read_bytes()


def test_manifest_hashes_and_default_counts():
    manifest = json.loads((IMAGES / 'manifest.json').read_text())
    assert manifest['schema'] == 1
    assert manifest['seed'] == 20261112
    assert manifest['shape'] == [28, 28]
    assert manifest['labels'] == list(range(10))
    assert list(manifest['classes'].values()) == [
        'horizontal bar', 'vertical bar', 'falling diagonal', 'rising diagonal',
        'plus', 'X', 'square outline', 'filled square', 'ring', 'filled disc',
    ]
    assert set(manifest['files']) == set(COUNTS)
    for split, count in COUNTS.items():
        entry = manifest['files'][split]
        assert entry['rows'] == 10 * count and entry['per_class'] == count
        assert hashlib.sha256((IMAGES / entry['path']).read_bytes()).hexdigest() == entry['sha256']
        assert Counter(row['label'] for row in rows(split)) == {label: count for label in range(10)}


def test_all_pixels_are_binary_and_all_splits_are_disjoint():
    seen = set()
    for split in COUNTS:
        for row in rows(split):
            assert set(row) == {'label', 'split', 'pixels'}
            assert type(row['label']) is int and 0 <= row['label'] <= 9
            assert row['split'] == split
            assert re.fullmatch('[0-9a-f]{196}', row['pixels'])
            pixels = decode(row['pixels'])
            assert pixels.shape == (28, 28) and set(np.unique(pixels)) == {0, 1}
            assert pixels.tobytes() not in seen
            seen.add(pixels.tobytes())
    assert len(seen) == 1400


@pytest.mark.parametrize('label', range(10))
def test_integer_labels_have_the_declared_geometry_and_variation(label):
    positions, sizes, densities = set(), set(), set()
    for split in COUNTS:
        for row in rows(split):
            if row['label'] != label:
                continue
            pixels = decode(row['pixels'])
            yy, xx = np.nonzero(pixels)
            crop = pixels[yy.min():yy.max()+1, xx.min():xx.max()+1]
            height, width = crop.shape
            positions.add((int(yy.min()), int(xx.min())))
            sizes.add(max(height, width))
            densities.add(float(crop.mean()))
            assert max(height, width) in range(9, 24, 2)
            if label in (0, 1):
                assert crop.all()
                assert (height if label == 0 else width) in (1, 3, 5)
            else:
                assert height == width
                middle = width // 2
                if label in (2, 3):
                    oriented = crop if label == 2 else np.fliplr(crop)
                    assert oriented.diagonal().all()
                    assert not oriented[0, -1] and not oriented[-1, 0]
                    assert np.array_equal(oriented, oriented.T)
                    assert all(np.array_equal(np.diagonal(oriented, offset=k),
                                              np.full(width-abs(k), abs(k) < oriented[0].sum()))
                               for k in range(-width+1, width))
                elif label == 4:
                    assert crop[middle].all() and crop[:, middle].all()
                    assert not crop[0, 0] and not crop[0, -1]
                    assert np.array_equal(crop, crop.T)
                    assert np.array_equal(crop, np.flipud(crop))
                    assert set(crop.sum(axis=1)) == {int(crop[0].sum()), width}
                    assert crop[0].sum() in (1, 3, 5)
                elif label == 5:
                    assert crop.diagonal().all() and np.fliplr(crop).diagonal().all()
                    assert not crop[0, middle] and not crop[middle, 0]
                    assert np.array_equal(crop, np.flipud(crop))
                    assert crop[middle].sum() in (1, 3, 5)
                elif label == 6:
                    assert crop[0].all() and crop[-1].all()
                    assert crop[:, 0].all() and crop[:, -1].all()
                    assert not crop[middle, middle]
                    holes_y, holes_x = np.nonzero(1 - crop)
                    hole = crop[holes_y.min():holes_y.max()+1, holes_x.min():holes_x.max()+1]
                    assert not hole.any() and hole.shape[0] == hole.shape[1]
                    assert holes_y.min() == holes_x.min() in (1, 2, 3)
                elif label == 7:
                    assert crop.all()
                else:
                    assert np.array_equal(crop, crop.T)
                    assert np.array_equal(crop, np.flipud(crop))
                    assert crop[0].sum() == crop[-1].sum() == 1
                    assert crop[0, middle] and crop[middle, 0]
                    assert not crop[0, 0]
                    assert crop[middle, middle] == (label == 9)
                    # Occupancy depends only on squared distance, with one outer
                    # boundary and (for the ring) one inner boundary.
                    radial = {}
                    for y in range(height):
                        for x in range(width):
                            radius2 = (x-middle)**2 + (y-middle)**2
                            assert radial.setdefault(radius2, crop[y, x]) == crop[y, x]
                    band = ''.join(str(radial[key]) for key in sorted(radial))
                    assert re.fullmatch('0+1+0+' if label == 8 else '1+0+', band)
    assert len(positions) > 1 and len(sizes) == 8
    if label not in (0, 1, 7):
        assert len(densities) > 1


def test_packing_is_row_major_most_significant_bit_first():
    pixels = np.zeros((28, 28), dtype=np.uint8)
    pixels[0, 0] = pixels[0, 7] = pixels[-1, -1] = 1
    encoded = '81' + '00' * 96 + '01'
    assert image_corpus.pack(pixels) == encoded
    np.testing.assert_array_equal(image_corpus.unpack(encoded), pixels)


@pytest.mark.parametrize('split', COUNTS)
def test_shapes_loader_agrees_with_frozen_rows(split):
    pixels, labels = image_corpus.load('shapes', split)
    frozen = rows(split)
    assert pixels.dtype == torch.float32 and pixels.shape == (10 * COUNTS[split], 28, 28)
    assert labels.dtype == torch.long and labels.shape == (len(frozen),)
    assert pixels.device.type == labels.device.type == 'cpu'
    np.testing.assert_array_equal(pixels.numpy(), np.stack([decode(row['pixels']) for row in frozen]))
    assert labels.tolist() == [row['label'] for row in frozen]


def test_mnist_manifest_reproduces_byte_for_byte():
    require_mnist()
    assert image_corpus.json_bytes(image_corpus.mnist_manifest()) == (IMAGES / 'mnist-manifest.json').read_bytes()


@pytest.mark.parametrize('split,count', [('train', 1000), ('test', 200)])
def test_mnist_loader_matches_prefix_threshold_and_manifest_hashes(split, count):
    require_mnist()
    original = np.loadtxt(ROOT / f'data/mnist_{split}.csv', delimiter=',', skiprows=1,
                          max_rows=count, dtype=np.int64)
    pixels, labels = image_corpus.load('mnist', split)
    assert pixels.dtype == torch.float32 and pixels.shape == (count, 28, 28)
    assert labels.dtype == torch.long and labels.shape == (count,)
    assert pixels.device.type == labels.device.type == 'cpu'
    assert set(pixels.unique().tolist()) == {0.0, 1.0}
    np.testing.assert_array_equal(pixels.numpy().reshape(count, 784), original[:, 1:] >= 128)
    np.testing.assert_array_equal(labels.numpy(), original[:, 0])
    manifest = json.loads((IMAGES / 'mnist-manifest.json').read_text())
    assert manifest['threshold'] == 128 and manifest['comparison'] == '>='
    entry = manifest['splits'][split]
    assert entry['rows'] == count and entry['shape'] == [count, 28, 28]
    assert hashlib.sha256(pixels.numpy().astype('u1').tobytes()).hexdigest() == entry['pixels_sha256']
    assert hashlib.sha256(labels.numpy().astype('<i8').tobytes()).hexdigest() == entry['labels_sha256']


def test_mnist_exact_threshold_and_prefix_boundary(tmp_path):
    header = 'label,' + ','.join(str(i) for i in range(784)) + '\n'
    sample = '7,0,127,128,255' + ',0' * 780 + '\n'
    (tmp_path / 'mnist_test.csv').write_text(header + sample * 200 + 'ignored suffix\n')
    pixels, labels = image_corpus.mnist_subset('test', data_dir=tmp_path)
    np.testing.assert_array_equal(pixels.reshape(200, 784)[:, :4], np.tile([0, 0, 1, 1], (200, 1)))
    assert not pixels.reshape(200, 784)[:, 4:].any()
    assert labels.tolist() == [7] * 200


def test_missing_mnist_skips_with_preparation_command(tmp_path, monkeypatch):
    from util import ProjectPaths
    monkeypatch.setattr(ProjectPaths, 'DATA_DIR', str(tmp_path))
    with pytest.raises(pytest.skip.Exception, match=r'run .venv/bin/python bin/prepare_data.py'):
        require_mnist()
    with pytest.raises(FileNotFoundError, match=r'run .venv/bin/python bin/prepare_data.py'):
        image_corpus.load('mnist', 'train', data_dir=tmp_path)


def test_unknown_corpus_or_split_is_rejected():
    for name, split in [('unknown', 'train'), ('shapes', 'unknown'), ('mnist', 'validation')]:
        with pytest.raises(ValueError, match='unknown'):
            image_corpus.load(name, split)
