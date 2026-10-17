"""Inspect frozen admission inputs; no presentation, commit or optimizer step."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/'test'),str(ROOT/'bin')]
import pytest
import torch
import Models
import Language, util
from bounded_tests import source_snapshot
from identity_experiment import restore_warmup,digest,save
from identity_experiment import noun_frame
from Spaces import _concept_alloc_of


def stats(values):
    return dict(count=len(values),nonzero=sum(v>0 for v in values),
        minimum=min(values,default=None),maximum=max(values,default=None))


def inspect(path):
    native_restore=Models.BaseModel.load_weights
    with tempfile.TemporaryDirectory() as folder,pytest.MonkeyPatch.context() as patch:
        config=next((path.parent/name for name in ('identity_model.xml','tiny_chunk_model.xml')
                     if (path.parent/name).exists()),None)
        if config is None:raise RuntimeError('the checkpoint needs its retained model XML')
        patch.setattr(util,'TheCompileBackend','none')
        util.init_device('cpu')
        util.init_config(path=str(config),defaults_path=str(ROOT/'data/model.xml'))
        Language.TheGrammar._configured=False
        patch.setattr(Models.BaseModel,'load_weights',lambda self,*a,**k:False)
        model,_=Models.BasicModel.from_config(str(config),data=Models.TheData)
        model.train()
        try:
            drift=restore_warmup(model,path,native_restore)
            owner=model._concept_owner();component=owner.components
            chart=component._ready();store=owner._closed_clause_store()
            rows=list(_concept_alloc_of(owner).layer()._tensor_row_keys)
            raw=[];projected=[];missing=0
            for row in range(len(store)):
                if (int(store.rel_type[row])==store.REL_DEF
                        or store.KINDS[int(store.record_kind[row])]!='observation'):
                    continue
                meaning=store.meaning_of(row)
                if meaning is None:missing+=1;continue
                raw.append(float(meaning.roles.norm()))
                projected.append(float(noun_frame(chart.project(meaning.roles)).norm()))
            return dict(checkpoint=str(path),sha256=digest(path),native_loader_drift=drift,
                online_learning_frozen=bool(getattr(owner,'_online_learning_frozen',False)),
                chart_rows=len(chart.rows),basis_norms=stats(chart.basis.norm(dim=-1).tolist()),
                allocated_row_orders=dict(Counter(owner._row_order(row) for row in rows)),
                content_width=chart.content_width,meaning_start=chart.meaning_start,
                meaning_pairs=chart.meaning_pairs,pole_stride=chart.pole_stride,
                raw_observation_norms=stats(raw),projected_noun_norms=stats(projected),
                unavailable_observations=missing,noun_columns=len(component.nouns.ids),
                pending=len(component.nouns._pending),mint_threshold=component.nouns.mint_threshold)
        finally:model.End()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('checkpoints',nargs='+',type=Path)
    parser.add_argument('--out',required=True,type=Path);args=parser.parse_args()
    torch.set_num_threads(1)
    report=dict(protocol='Frozen checkpoint inspection; no new examples or learning.',
        source=source_snapshot(ROOT),script_sha256=digest(Path(__file__)),
        checkpoints=[inspect(path.resolve()) for path in args.checkpoints])
    save(args.out,report)
    print(json.dumps(report['checkpoints'],indent=2))


if __name__=='__main__':main()
