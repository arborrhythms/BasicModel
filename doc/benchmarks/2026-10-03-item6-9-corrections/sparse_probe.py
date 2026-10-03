"""Keep the failing assertion and dump its field addresses and definitions."""
import json
from pathlib import Path
import pytest

def pytest_exception_interact(node, call, report):
    local = call.excinfo.traceback[-1].frame.f_locals
    model, owner = local.get('m'), local.get('cs0')
    if model is None or owner is None:
        return
    import torch, Spaces
    last = model._combine_last_cs_sub
    data = dict(field_slice=owner._field_order_slice(0), physical_slice=owner.order_slice(0),
        capacity=owner._field_caps(), rows=last._concept_inventory_rows.tolist(),
        activations=last._concept_activations.detach().tolist(),
        code_norms=owner.similarity_codebook.getW().detach().norm(dim=-1).tolist(),
        ids=owner._csw_concept_ids if hasattr(owner,'_csw_concept_ids') else None)
    layer = Spaces._concept_alloc_of(owner).layer()
    data['features'] = repr(layer.features._index)
    data['feature_groups'] = repr(layer.feature_groups)
    Path(__file__).with_name('sparse-failure-fields.json').write_text(json.dumps(data,indent=2,default=str))
