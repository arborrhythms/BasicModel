"""Inspect added recovery observations without presenting data or learning."""
import argparse
import hashlib
import json
from pathlib import Path

import pytest
import torch

import Language
import Models
import util
from IndependentComponents import noun_frame
from identity_experiment import ROOT, digest, restore_warmup


def inspect(path):
    native_restore = Models.BaseModel.load_weights
    with pytest.MonkeyPatch.context() as patch:
        config = path.parent/'identity_model.xml'
        patch.setattr(util, 'TheCompileBackend', 'none')
        util.init_device('cpu')
        util.init_config(path=str(config), defaults_path=str(ROOT/'data/model.xml'))
        Language.TheGrammar._configured = False
        patch.setattr(Models.BaseModel, 'load_weights', lambda self, *a, **k: False)
        model, _ = Models.BasicModel.from_config(str(config), data=Models.TheData)
        try:
            drift = restore_warmup(model, path, native_restore)
            owner = model._concept_owner()
            chart = owner.components._ready()
            store = owner._closed_clause_store()
            rows = {}
            for index in range(len(store)):
                if (int(store.rel_type[index]) == store.REL_DEF or
                        store.KINDS[int(store.record_kind[index])] != 'observation'):
                    continue
                meaning = store.meaning_of(index)
                if meaning is None:
                    continue
                value = meaning.roles.detach()
                band = value[..., chart.meaning_start:chart.content_width]
                rows[str(int(store.row_ids[index]))] = dict(
                    raw_norm=float(value.norm()), meaning_band_norm=float(band.norm()),
                    projected_norm=float(noun_frame(chart.project(value)).norm()),
                    content_sha256=hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest())
            return dict(checkpoint=str(path), sha256=digest(path),
                        native_loader_drift=drift, rows=rows)
        finally:
            model.End()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('warmup', type=Path)
    parser.add_argument('finals', type=Path, nargs='+')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    warm = inspect(args.warmup)
    finals = []
    for path in args.finals:
        final = inspect(path)
        added = sorted(set(final['rows'])-set(warm['rows']))
        values = [final['rows'][key] for key in added]
        final['comparison'] = dict(
            retained_warmup_rows=len(set(final['rows']) & set(warm['rows'])),
            removed_warmup_rows=len(set(warm['rows'])-set(final['rows'])),
            added_rows=len(added), added_ids=added,
            added_nonzero_raw=sum(row['raw_norm'] > 0 for row in values),
            added_nonzero_meaning=sum(row['meaning_band_norm'] > 0 for row in values),
            added_nonzero_projected=sum(row['projected_norm'] > 0 for row in values),
            added_meaning_max=max((row['meaning_band_norm'] for row in values), default=None),
            added_projection_max=max((row['projected_norm'] for row in values), default=None),
            added_unique_contents=len({row['content_sha256'] for row in values}))
        finals.append(final)
    result = dict(protocol=__doc__, script_sha256=digest(Path(__file__)), warmup=warm, finals=finals)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps([dict(checkpoint=r['checkpoint'], **{k:v for k,v in r['comparison'].items()
                        if k!='added_ids'}) for r in finals]), flush=True)


if __name__ == '__main__':
    main()
