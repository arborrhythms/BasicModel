"""Read-only native priming measurements, one record per certificate."""
import json
import os
from pathlib import Path

import pytest
import torch


@pytest.fixture(autouse=True)
def priming_probe(monkeypatch, request):
    import Priming
    from Spaces import ConceptualSpace
    original = Priming.diffuse
    project = ConceptualSpace.project_priming_to_towers
    projections = []
    totals = dict(events=0, overlap_events=0, max_energy_error=0.,
                  max_graph_edges=0, max_sources=0, max_part_reads=0,
                  max_candidates=0, max_comparisons=0, max_overlap_edges=0,
                  truncated_frontiers=0, nonzero_overlap_transfers=0)

    def measured(owner, surface, edges, spread, active, field):
        result = original(owner, surface, edges, spread, active, field)
        audit = getattr(owner, '_last_priming_diffusion', None)
        if spread <= 0 or not bool(active.any()) or audit is None:
            return result
        totals['events'] += 1
        totals['max_energy_error'] = max(totals['max_energy_error'],
            float((audit['after'] - audit['before']).abs().max()))
        totals['max_graph_edges'] = max(totals['max_graph_edges'], audit['graph_edges'])
        totals['overlap_events'] += bool(audit['overlap'])
        totals['nonzero_overlap_transfers'] += int(torch.count_nonzero(audit['overlap_flow']))
        for row in audit['overlap']:
            for key in ('sources', 'part_reads', 'candidates', 'comparisons'):
                totals['max_' + key] = max(totals['max_' + key], row[key])
            totals['max_overlap_edges'] = max(totals['max_overlap_edges'], row['edges'])
            totals['truncated_frontiers'] += row['frontier_truncated']
        return result
    monkeypatch.setattr(Priming, 'diffuse', measured)
    def projection(owner, *args, **kwargs):
        surface = owner.priming_weights()
        projections.append(dict(
            shape=None if surface is None else list(surface.shape),
            heat=None if surface is None else float(surface.max()),
            words=[dict(word=word, row=owner._csw_row_of(word),
                        objects=owner.definitions.objects(word),
                        objectless=owner.interpret.no_object(word),
                        pending=word in owner.interpret._field_pending)
                   for word in sorted(owner._word_definition_ids())]))
        return project(owner, *args, **kwargs)
    monkeypatch.setattr(ConceptualSpace, 'project_priming_to_towers', projection)
    yield
    directory = Path(os.environ['PRIMING_MEASUREMENT'])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / (request.node.name + '.json')).write_text(
        json.dumps(dict(test=request.node.nodeid, **totals, projections=projections), indent=2) + '\n')
