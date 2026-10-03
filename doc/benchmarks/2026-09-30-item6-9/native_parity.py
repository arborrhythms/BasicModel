"""Compare layouts from one freshly initialized checkpoint, without a seed."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from Models import BaseModel
import native_probe

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=('packed', 'single'), required=True)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--create', action='store_true')
parser.add_argument('--out', required=True)
args = parser.parse_args()
original = BaseModel.from_config
descriptor = BaseModel.__dict__['from_config']


def from_checkpoint(cls, *positional, **keywords):
    model, record = original(*positional, **keywords)
    if args.create:
        if args.checkpoint.exists():
            raise FileExistsError(args.checkpoint)
        model.save_weights(str(args.checkpoint))
    elif not model.load_weights(str(args.checkpoint), strict=True, require_match=True):
        raise RuntimeError('the shared unseeded initialization was not loaded')
    return model, record


BaseModel.from_config = classmethod(from_checkpoint)
try:
    sys.argv = [__file__, '--config', str(Path(__file__).resolve().parent.parent /
        '2026-09-26-item7-5/parity.xml'), '--parity', args.mode, '--out', args.out]
    native_probe.main()
finally:
    BaseModel.from_config = descriptor
out = Path(args.out)
report = json.loads(out.read_text())
report['initialization'] = dict(origin='one fresh unseeded checkpoint',
    created=args.create, checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
    reason='Both layouts must read the same weights and dictionaries; no seed is chosen.')
out.write_text(json.dumps(report, indent=2) + '\n')
