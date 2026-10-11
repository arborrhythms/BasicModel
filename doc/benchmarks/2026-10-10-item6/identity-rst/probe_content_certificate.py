"""Audit the new content charge on the retained forced-operator certificate."""
import json
from pathlib import Path
from unittest.mock import patch
import torch
import util
from Models import BasicModel
from test_operators_final_b import forced_prefix_certificate

torch.set_num_threads(1)
util.TheCompileBackend='none'
out=Path('output/item6-rst-content-certificate');out.mkdir(exist_ok=False)
original=BasicModel._reconstruct_trial
records=[]


def read(model,record):
    result=original(model,record)
    start=record.primed.meaning_start
    target=torch.where(record.word_valid[...,None],record.word_values.detach(),0.)[...,start:]
    returned=torch.where(record.word_valid[...,None],result[0].detach(),0.)[...,start:]
    def bands(x):return torch.stack((x.clamp_min(0).sum(1),(-x).clamp_min(0).sum(1)),-1)
    records.append(dict(target=target.cpu(),returned=returned.cpu(),word_rows=record.word_rows.cpu(),
        positional=(target-returned).abs().sum((1,2)).tolist(),
        band_error=(bands(target)-bands(returned)).abs().sum((1,2)).tolist(),
        baseline=target.abs().sum((1,2)).tolist()))
    return result


with patch.object(BasicModel,'_reconstruct_trial',read):
    report=forced_prefix_certificate()
torch.save(records,out/'values.pt')
(out/'result.json').write_text(json.dumps(dict(report=report,
    errors=[{k:v for k,v in r.items() if not torch.is_tensor(v)} for r in records]),indent=2)+'\n')
