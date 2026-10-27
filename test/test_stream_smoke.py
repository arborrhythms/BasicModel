"""End-to-end sanity that streaming keeps stream-slot -> sentence mapping."""
import os
from pathlib import Path

import pytest
import torch

_RUN_SLOW = os.getenv("RUN_SLOW") == "1"


def test_stream_slot_identity():
    """Run through every DataLoader step on a synthetic train_input and
    assert batch slot ``b`` always receives a sentence from its own
    contiguous slab (i.e. stream ``b``)."""
    import data as data_mod

    # 16 synthetic docs, B=4 -> stream_len=4. Slab 0 = [d0..d3],
    # slab 1 = [d4..d7], etc.
    td = data_mod.Data()
    td.train_input = [f"d{i}" for i in range(16)]
    td.train_output = [torch.zeros(1) for _ in td.train_input]
    loader = td.data_loader(split="train", num_streams=4, num_workers=0,
                            prefetch_factor=None)

    steps_seen = 0
    for step, (inp, _out) in enumerate(loader):
        for b, sentence in enumerate(inp):
            expected_idx = b * 4 + step
            assert sentence == f"d{expected_idx}", (
                f"step {step} row {b}: expected d{expected_idx}, "
                f"got {sentence}"
            )
        steps_seen += 1
    assert steps_seen == 4, f"expected 4 steps, saw {steps_seen}"


@pytest.mark.skipif(not _RUN_SLOW, reason="slow -- set RUN_SLOW=1")
def test_stream_smoke_runs_one_epoch(tmp_path):
    """Full training run of 1 epoch on stream_smoke.xml.

    Requires a downloaded shard. Skipped by default -- run with:
        RUN_SLOW=1 pytest test/test_stream_smoke.py
    """
    import subprocess

    proj = Path(__file__).resolve().parent.parent
    import sys
    python = Path(sys.executable)
    entry = proj / "bin" / "Models.py"
    source = proj / "data" / "stream_smoke.xml"
    # Keep all 200 documents and the one-epoch budget. Bound the native
    # dictionary before construction from that unchanged corpus: byte/pad
    # rows, one canonical row per unit, and every possible identity atom.
    import json
    import xml.etree.ElementTree as ET
    from data import Data
    from util import init_config
    from WordIdentity import base_atoms, text_of
    import Meronomy
    init_config(path=str(source), defaults_path=str(proj/'data/model.xml'))
    tree = ET.parse(source)
    section = tree.find('architecture/data')
    documents = int(section.findtext('maxDocs'))
    dataset = Data()
    dataset.load('text', max_docs=documents, num_shards=1,
                 shard_dir=str(proj/'data/fineweb'))
    width = int(tree.findtext('InputSpace/nOutput'))
    units = set()
    for split in ('train', 'validation', 'test'):
        for text in getattr(dataset, split+'_input'):
            raw = text.encode('utf8')
            for value in (raw, raw[:width]):
                units.update(value[lo:hi] for lo, hi in Meronomy.percept_spans(value))
    atoms = set()
    for raw in units:
        atoms.update(base_atoms(raw))
        marked = '#' + text_of(raw) + '#'
        # WordIdentity may mint a triple or quadruple at a differing position.
        # Include all such keys, independently of any sampled training path.
        for size in (3, 4):
            atoms.update(('mint', f'{marked[pos:pos+size]}@{pos}')
                         for pos in range(len(marked)-size+1))
    capacity = 257 + len(units) + len(atoms)
    tree.find('PartSpace/nVectors').text = str(capacity)
    section.find('shardDir').text = str(proj/'data/fineweb')
    tree.find('architecture/weightsPath').text = str(tmp_path/'stream_smoke.ckpt')
    config = tmp_path/'stream_smoke.xml'
    tree.write(config, encoding='unicode')
    (tmp_path/'fixture-capacity.json').write_text(json.dumps(dict(
        documents=documents, epochs=int(tree.findtext('architecture/training/numEpochs')),
        batch_size=int(tree.findtext('architecture/training/batchSize')),
        byte_and_padding_rows=257, units=len(units), atom_keys=len(atoms),
        percept_capacity=capacity), indent=2)+'\n')

    env = {**os.environ, "PYTHONPATH": str(proj / "bin")}
    result = subprocess.run(
        [str(python), str(entry), str(config)],
        cwd=str(proj / "bin"), env=env,
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, (
        f"training exited {result.returncode}\nSTDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    # Cheap signal: the run must have printed at least one loss line.
    assert "loss" in result.stdout.lower()
