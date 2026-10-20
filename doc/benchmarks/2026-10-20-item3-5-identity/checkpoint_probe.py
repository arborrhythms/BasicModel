"""Read the retained stage-3 checkpoint at its identity-migration boundary."""
from pathlib import Path
import json
import sys
import tempfile
import argparse

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'bin'), str(ROOT/'test')]
import pytest
import torch
from IndependentComponents import IndependentComponents
from IdentityRows import witnesses
from Models import BaseModel
from ThoughtReferences import bindings
from identity_experiment import make_model

records = []
restore = BaseModel.load_weights
parser = argparse.ArgumentParser()
parser.add_argument('--phase', choices=('stage3', 'np_definite', 'np_description'), default='stage3')
args = parser.parse_args()


def inspect(component):
    owner, store = component.owner, component.owner._closed_clause_store()
    for identity in getattr(component, '_migrating_individuals', ()):
        seat = owner._csw_row_of(identity)
        cached = owner.similarity_codebook.W[seat]
        members = []
        for row in witnesses(owner, identity):
            content = store.point_of_row(int(store.row_ids[row]))
            members.append(dict(row=row, address=int(store.row_ids[row]),
                content=content.tolist(), metadata=store.meaning_of(row).metadata(),
                refs=store.refs[row].tolist(), text=store.text_of(row),
                postings={str(k): v for k, v in store._leaf_postings.items() if row in v}))
        records.append(dict(identity=identity, seat=seat,
            cached=cached.tolist(), definitions=repr(tuple(owner.concept_parts(identity))),
            members=members))


with tempfile.TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as patch:
    model = make_model(Path(directory), patch)
    patch.setattr(IndependentComponents, 'migrate_identity_rows', inspect)
    try:
        phases = json.loads((ROOT/'test/fixtures/phase-artifacts.json').read_text())['phases']
        path = ROOT/phases[args.phase]['path']
        bundle = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        saved = bundle['state_dict']
        comparison = {k[len('comparison_reader.'):]: v for k,v in saved.items()
                      if k.startswith('comparison_reader.')}
        model.comparison_reader.load_state_dict(comparison, strict=True)
        catalog = json.loads((ROOT/'test/fixtures/grammar-catalog-63ece2e1.json').read_text())
        assert restore(model, str(path), require_match=True, grammar_catalog=catalog)
        owner = model._concept_owner()
        projection = next(v for k,v in saved.items() if k.endswith('.identity.projection'))
        identity = model.perceptualSpace.percept_store.identity
        forms = [(raw, identity.form(raw)) for raw in identity.words]
        projected = torch.stack([torch.nn.functional.pad(code @ projection,
            (0, owner.nWhat-projection.shape[1])) for _,code in forms])
        matches = []
        for record in records:
            for member in record['members']:
                point = projected.new_tensor(member['content'])
                distance = (projected-point).abs().amax(-1)
                scale = (projected*point).sum(-1)/projected.square().sum(-1).clamp_min(1e-12)
                scaled_error = (projected*scale[:,None]-point).abs().amax(-1)
                matches.append(dict(identity=record['identity'], row=member['row'],
                    min_error=float(distance.min()),
                    exact=[str(forms[i][0]) for i in (distance <= 1e-5).nonzero().flatten().tolist()],
                    scaled_exact=[dict(form=str(forms[i][0]), scale=float(scale[i]),
                        error=float(scaled_error[i])) for i in (scaled_error <= 1e-5).nonzero().flatten().tolist()],
                    nonzero=int(point.count_nonzero()), width=point.numel()))
        output = dict(source=str(path.relative_to(ROOT)), training_steps=0, identities=records,
                      exact_form_projection_matches=matches)
        destination = Path(__file__).with_name('checkpoint_probe-'+args.phase+'.json')
        destination.write_text(json.dumps(output, indent=2)+'\n')
        print(json.dumps([dict(identity=r['identity'], seat=r['seat'],
            cached_range=[min(r['cached']),max(r['cached'])], definitions=r['definitions'],
            members=[dict(row=m['row'], range=[min(m['content']),max(m['content'])],
                          refs=m['refs'], metadata=m['metadata']) for m in r['members']])
            for r in records[:2]], indent=2))
    finally:
        model.End()
        model.symbolSpace.soft_reset()
