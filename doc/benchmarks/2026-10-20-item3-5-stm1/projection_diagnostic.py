"""Observe one batch of the existing projection fixture, without an extension."""
import json
from Spaces import ConceptualSpace, _concept_alloc_of
from test_priming_energy import _projected_model

original = ConceptualSpace.project_priming_to_towers


def project(owner, ps, ws, gain=1., **kw):
    report = dict(gain=gain, kwargs=kw, surface=owner.priming_weights().shape,
        targets=[dict(type=type(s).__name__, dim=s._priming_dim(),
                      canonical=s._priming_target() is s) for s in (ps, ws)])
    report['words'] = [dict(word=w, row=owner._csw_row_of(w),
        description=_concept_alloc_of(owner).word_descriptions.get(w) or owner.interpret._pending.get(w),
        objectless=owner.interpret.no_object(w)) for w in owner._word_definition_ids()]
    result = original(owner, ps, ws, gain=gain, **kw)
    report['after'] = [None if s.priming_weights() is None else s.priming_weights().tolist() for s in (ps, ws)]
    print(json.dumps(report, default=list))
    return result


ConceptualSpace.project_priming_to_towers = project
model = _projected_model(epochs=1)
model.End()
