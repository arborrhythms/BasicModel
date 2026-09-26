"""Exact band-address checks on the available devices; no model training."""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'bin'), str(ROOT / 'test')]
from WhereRegistry import WhereRegistry
from Spaces import WhenEncoding
from bounded_tests import source_snapshot

torch.set_num_threads(1)
source = source_snapshot(ROOT)
registry = WhereRegistry([('input', 8192), ('parts', 400_000),
    ('wholes', 400_000), ('symbols', 2_000_000)], symbol_slots=256)
where = registry.encoding
when = WhenEncoding(n_when=4).set_capacity(1_048_576)
report = dict(source=source, torch=torch.__version__, devices=[],
              where_capacity=registry.capacity,
              where_periods=[where.maxVal, where.period_hf],
              when_periods=[when.maxVal, when.period_hf])
for device in ('cpu', 'mps'):
    if device == 'mps' and not torch.backends.mps.is_available():
        continue
    addresses = torch.cat((torch.arange(2048), torch.arange(2**24 - 1024, 2**24 + 1024),
        torch.arange(registry.capacity-2048, registry.capacity))).to(device)
    times = torch.cat((torch.arange(2048), torch.arange(when.capacity-2048, when.capacity))).to(device)
    bands = where.encode(addresses)
    torch.testing.assert_close(where.decode_index(bands), addresses)
    torch.testing.assert_close(when.decode_index(when.encode(times)), times)
    if device == 'cpu':
        compiled = torch.compile(lambda x: where.decode_index(where.encode(x)),
                                 backend='inductor', fullgraph=True)
        torch.testing.assert_close(compiled(addresses), addresses)
    report['devices'].append(dict(device=device, spatial_addresses=len(addresses),
        temporal_addresses=len(times), exact=True, compiled=device == 'cpu'))
report['source_unchanged'] = source_snapshot(ROOT) == source
assert report['source_unchanged']
Path(sys.argv[1]).write_text(json.dumps(report, indent=2)+'\n')
