"""Frozen binary images and an in-memory MNIST prefix; no learner or network."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import random

import numpy as np
import torch

from prepare_data import PREPARE_COMMAND, require_local, verify

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / 'data/images'
SEED = 20261112
SIDE = 28
SHAPE_COUNTS = {'train': 100, 'validation': 20, 'test': 20}
MNIST_COUNTS = {'train': 1000, 'test': 200}
SHAPES = (
    'horizontal bar', 'vertical bar', 'falling diagonal', 'rising diagonal',
    'plus', 'X', 'square outline', 'filled square', 'ring', 'filled disc',
)


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode('utf-8')


def pack(pixels):
    """First row first; each byte's first pixel occupies its most significant bit."""
    pixels = np.asarray(pixels)
    if pixels.shape != (SIDE, SIDE) or not np.isin(pixels, (0, 1)).all():
        raise ValueError('expected a binary 28 by 28 image')
    return np.packbits(pixels.astype(np.uint8).reshape(-1), bitorder='big').tobytes().hex()


def unpack(encoded):
    if len(encoded) != 196 or any(char not in '0123456789abcdef' for char in encoded):
        raise ValueError('expected 196 lowercase hexadecimal digits')
    raw = np.frombuffer(bytes.fromhex(encoded), dtype=np.uint8)
    return np.unpackbits(raw, bitorder='big').reshape(SIDE, SIDE)


def render_shape(label, *, radius, stroke, x, y):
    """Integer masks in a (2r+1)-square, with y increasing down the image."""
    if label not in range(10) or not 4 <= radius <= 11 or not 1 <= stroke <= 3:
        raise ValueError('shape parameters are outside the declared bounds')
    if not (radius <= x <= 27 - radius and radius <= y <= 27 - radius):
        raise ValueError('shape would cross the image boundary')
    rows, cols = np.indices((SIDE, SIDE))
    u, v = cols - x, rows - y
    au, av = np.abs(u), np.abs(v)
    box = (au <= radius) & (av <= radius)
    # A stroke parameter s gives an axial/diagonal band of 2s-1 pixels.
    predicates = (
        av < stroke,
        au < stroke,
        np.abs(u - v) < stroke,
        np.abs(u + v) < stroke,
        (au < stroke) | (av < stroke),
        (np.abs(u - v) < stroke) | (np.abs(u + v) < stroke),
        np.maximum(au, av) > radius - stroke,
        box,
        ((u*u + v*v) <= radius*radius) & ((u*u + v*v) > (radius-stroke)**2),
        (u*u + v*v) <= radius*radius,
    )
    return (box & predicates[label]).astype(np.uint8)


def freeze(out=IMAGE_DIR, *, seed=SEED):
    """Write the three shape splits and their manifest, independent of MNIST."""
    out = Path(out)
    rng = random.Random(seed)
    seen = set()
    splits = {split: [] for split in SHAPE_COUNTS}
    for label in range(10):
        for split, count in SHAPE_COUNTS.items():
            accepted = 0
            for _ in range(100000):
                radius = rng.randint(4, 11)
                stroke = rng.randint(1, 3) if label not in (7, 9) else 1
                x = rng.randint(radius, 27 - radius)
                y = rng.randint(radius, 27 - radius)
                encoded = pack(render_shape(label, radius=radius, stroke=stroke, x=x, y=y))
                if encoded in seen:
                    continue
                seen.add(encoded)
                splits[split].append(dict(label=label, split=split, pixels=encoded))
                accepted += 1
                if accepted == count:
                    break
            else:
                raise ValueError(f'cannot obtain {count} distinct images for class {label}/{split}')
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, rows in splits.items():
        payload = ''.join(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n'
                          for row in rows).encode('utf-8')
        path = out / f'{split}.jsonl'
        path.write_bytes(payload)
        files[split] = dict(path=path.name, rows=len(rows), per_class=SHAPE_COUNTS[split],
                            sha256=sha256(payload))
    manifest = dict(
        schema=1, name='shapes', seed=seed, shape=[SIDE, SIDE], labels=list(range(10)),
        classes={str(label): name for label, name in enumerate(SHAPES)},
        command=f'.venv/bin/python bin/image_corpus.py freeze --seed {seed} --out data/images',
        encoding='196 lowercase hex digits; row-major; first pixel is most significant bit',
        generator='Python random.Random (MT19937); inclusive randint; class then train/validation/test',
        bounds=dict(radius=[4, 11], side='2*radius+1', stroke=[1, 3],
                    center_x='radius..27-radius', center_y='radius..27-radius'),
        raster='integer masks; y increases downwards; axial/diagonal bands abs(offset)<stroke; '
               'square border max(abs(x),abs(y))>radius-stroke; '
               'ring (radius-stroke)^2<x^2+y^2<=radius^2; '
               'filled shapes ignore stroke and consume no stroke draw',
        uniqueness='reject repeated pixel arrays globally, including across labels and splits',
        order='ascending label, then accepted draw order within each split', files=files,
    )
    (out / 'manifest.json').write_bytes(json_bytes(manifest))
    return manifest


def mnist_subset(split, *, data_dir=None):
    """Read fixed source-row prefixes, threshold at 128, and retain them only in memory.

    Tests call local_data.require_mnist() before this function. Applications get
    a FileNotFoundError naming the preparation command when a CSV is absent.
    """
    if split not in MNIST_COUNTS:
        raise ValueError(f'unknown MNIST split: {split}')
    data_dir = ROOT / 'data' if data_dir is None else Path(data_dir)
    path = require_local(data_dir / f'mnist_{split}.csv')
    count = MNIST_COUNTS[split]
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        if len(header) != 785 or header[0] != 'label':
            raise ValueError(f'invalid MNIST CSV header in {path}; run {PREPARE_COMMAND}')
        rows = list(itertools.islice(reader, count))
    if len(rows) != count or any(len(row) != 785 for row in rows):
        raise ValueError(f'{path}: expected {count} complete rows after the header')
    values = np.asarray(rows, dtype=np.int64)
    labels, pixels = values[:, 0], values[:, 1:]
    if not ((0 <= labels) & (labels <= 9)).all() or not ((0 <= pixels) & (pixels <= 255)).all():
        raise ValueError(f'{path}: labels or pixels are out of range')
    return (pixels >= 128).astype(np.uint8).reshape(count, SIDE, SIDE), labels.copy()


def array_identity(pixels, labels):
    """Hashes have fixed dtype, byte order and C order on every host."""
    return dict(rows=len(labels), shape=list(pixels.shape),
                pixels_sha256=sha256(np.asarray(pixels, dtype=np.uint8).tobytes(order='C')),
                labels_sha256=sha256(np.asarray(labels, dtype='<i8').tobytes(order='C')))


def mnist_manifest(*, data_dir=None):
    data_dir = ROOT / 'data' if data_dir is None else Path(data_dir)
    external = json.loads((ROOT / 'data/external-files.json').read_text())
    sources, splits = {}, {}
    for split in MNIST_COUNTS:
        source_path = f'data/mnist_{split}.csv'
        entry = next(entry for entry in external['files'] if entry['path'] == source_path)
        verify(require_local(data_dir / Path(source_path).name), entry)
        sources[split] = dict(path=source_path, sha256=entry['sha256'])
        splits[split] = array_identity(*mnist_subset(split, data_dir=data_dir))
    return dict(schema=1, name='mnist', threshold=128, comparison='>=',
                selection='first rows after the CSV header; source order; no shuffle',
                shape=[SIDE, SIDE], labels=list(range(10)), sources=sources, splits=splits,
                pixels_hash_encoding='uint8 [N,28,28], C row-major bytes, values 0 or 1',
                labels_hash_encoding='signed little-endian int64 [N], C order bytes',
                command='.venv/bin/python bin/image_corpus.py freeze-mnist --out data/images',
                preparation_command=PREPARE_COMMAND,
                storage='derived arrays are in memory only; only this manifest is tracked')


def freeze_mnist(out=IMAGE_DIR):
    manifest = mnist_manifest()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'mnist-manifest.json').write_bytes(json_bytes(manifest))
    return manifest


def load(name, split, *, image_dir=None, data_dir=None):
    """Return CPU float32 [N,28,28] images and int64 [N] labels.

    The two positional arguments are the certificate API; keyword paths permit
    isolated data checks. No normalization, augmentation or model is involved.
    """
    image_dir = IMAGE_DIR if image_dir is None else Path(image_dir)
    if name == 'shapes':
        if split not in SHAPE_COUNTS:
            raise ValueError(f'unknown shapes split: {split}')
        manifest = json.loads((image_dir / 'manifest.json').read_text())
        entry = manifest['files'][split]
        payload = (image_dir / entry['path']).read_bytes()
        if sha256(payload) != entry['sha256']:
            raise ValueError(f'shapes/{split}: file hash mismatch')
        rows = [json.loads(line) for line in payload.splitlines()]
        if len(rows) != entry['rows'] or any(row['split'] != split or
                type(row['label']) is not int or row['label'] not in range(10) for row in rows):
            raise ValueError(f'shapes/{split}: invalid count, split or label')
        pixels = np.stack([unpack(row['pixels']) for row in rows])
        labels = np.array([row['label'] for row in rows], dtype=np.int64)
    elif name == 'mnist':
        pixels, labels = mnist_subset(split, data_dir=data_dir)
        manifest = json.loads((image_dir / 'mnist-manifest.json').read_text())
        if array_identity(pixels, labels) != manifest['splits'][split]:
            raise ValueError(f'mnist/{split}: derived array hash mismatch')
    else:
        raise ValueError(f'unknown image corpus: {name}')
    return torch.from_numpy(pixels.astype(np.float32)), torch.from_numpy(labels.astype(np.int64))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    shapes = commands.add_parser('freeze', help='freeze generated shapes (no prepared data needed)')
    shapes.add_argument('--seed', type=int, default=SEED)
    shapes.add_argument('--out', type=Path, default=IMAGE_DIR)
    mnist = commands.add_parser('freeze-mnist', help='record hashes of prepared binary MNIST prefixes')
    mnist.add_argument('--out', type=Path, default=IMAGE_DIR)
    args = parser.parse_args()
    manifest = freeze(args.out, seed=args.seed) if args.command == 'freeze' else freeze_mnist(args.out)
    print(json.dumps(manifest, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
