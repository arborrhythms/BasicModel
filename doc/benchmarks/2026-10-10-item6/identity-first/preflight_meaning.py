"""Verify native meaning signal before spending a learning budget."""
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/'test'),str(ROOT/'bin')]
import numpy as np
import pytest
import torch
from attention_curriculum_gates import present
from bounded_tests import source_snapshot
from identity_experiment import make_model,component_signal,save,digest

out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(1)
random.seed(20261010);np.random.seed(20261010);torch.manual_seed(20261010)
source=source_snapshot(ROOT);reports=[]
with pytest.MonkeyPatch.context() as patch:
    model=make_model(out,patch)
    try:
        for repetition in range(4):
            report=present(model,dict(fields=['cat','dog','bird','horse']))
            reports.append(dict(repetition=repetition+1,signal=component_signal(model),
                                exact_readbacks=report['exact_readbacks']))
            save(out/'progress.json',reports)
        signal=reports[-1]['signal']
        valid=signal['meaning_pairs']>0 and signal['nonzero_basis_rows']>0 and signal['nonzero_observation_rows']>0
        save(out/'result.json',dict(protocol='Four ordinary word presentations; no optimizer or forced meaning.',
            seed=20261010,source=source,source_matched=source==source_snapshot(ROOT),
            script_sha256=digest(Path(__file__)),valid=valid,reports=reports))
        if not valid:raise RuntimeError('ordinary text did not reach nonzero component coordinates')
    finally:model.End()
print(json.dumps(reports,indent=2))
