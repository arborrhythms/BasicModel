"""Read-only bounded-menu diagnostics at the original decomposition snapshots."""
import json
import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def inspect_binding_bank(monkeypatch, request):
    from Models import BasicModel
    original = BasicModel._sentence_reference_bank

    def bank(model, active, like):
        result = original(model, active, like)
        directory = Path(os.environ['BINDING_DIAGNOSTIC'])
        directory.mkdir(parents=True, exist_ok=True)
        disc = model.symbolSpace.expectation
        record = dict(test=request.node.nodeid, slot=model._open_sentence_slot,
            ids=result.ids.detach().cpu().tolist(), orders=result.orders.detach().cpu().tolist(),
            valid=result.valid.detach().cpu().tolist(),
            frames=[[dict(row=f.row_id, symbols=f.symbol_ids)
                     for f in disc.situation_references(b)] for b in range(len(active))])
        owner = model._concept_owner()
        from Spaces import _concept_alloc_of
        alloc = _concept_alloc_of(owner)
        record['individuals'] = sorted(alloc.individuals)
        record['definitions'] = {str(word): owner.definitions.objects(word)
                                 for word in owner.definitions.word_ids}
        with (directory/'banks.jsonl').open('a') as output:
            output.write(json.dumps(record)+'\n')
        return result

    monkeypatch.setattr(BasicModel, '_sentence_reference_bank', bank)
    from PartClosing import PartClosing
    preflight = PartClosing.preflight

    def closing(plan):
        directory = Path(os.environ['BINDING_DIAGNOSTIC'])
        directory.mkdir(parents=True, exist_ok=True)
        record = dict(test=request.node.nodeid, rows=[
            dict(symbol=r['symbol'], reference=r['value'].reference,
                 order=r['value'].order, operation=r['value'].operation,
                 leaves=r['value'].leaves, pair=r['pair'], seat=r['seat'],
                 types=r['types'])
            for r in plan._plan])
        with (directory/'closings.jsonl').open('a') as output:
            output.write(json.dumps(record)+'\n')
        return preflight(plan)
    monkeypatch.setattr(PartClosing, 'preflight', closing)
    from Language import LanguageSpace
    choose = LanguageSpace.choose_operation

    def operation(language, state, *args, **kwargs):
        result = choose(language, state, *args, **kwargs)
        if isinstance(result, tuple):
            choice, refs, relative, operands = result
            if bool(choice.applied.any()):
                directory = Path(os.environ['BINDING_DIAGNOSTIC'])
                directory.mkdir(parents=True, exist_ok=True)
                record = dict(test=request.node.nodeid,
                    depth=state[1].tolist(), order=state[2][:, :2].tolist(),
                    scope=kwargs['reference_scope'][:, :2].tolist(),
                    kind=choice.kind.tolist(), op=choice.local_op.tolist(),
                    refs=refs.tolist(), relative=relative.tolist())
                with (directory/'choices.jsonl').open('a') as output:
                    output.write(json.dumps(record)+'\n')
        return result
    monkeypatch.setattr(LanguageSpace, 'choose_operation', operation)
    from ReferenceClose import ReferenceClose
    preview = ReferenceClose.preview

    def inspect_close(resolver, clause, frames, **kwargs):
        result = preview(resolver, clause, frames, **kwargs)
        directory = Path(os.environ['BINDING_DIAGNOSTIC'])
        directory.mkdir(parents=True, exist_ok=True)
        record = dict(test=request.node.nodeid, point=clause.point is not None,
            relation=clause.relation, refs=str(clause.refs), items=[
                dict(mode=item['mode'], selected=str(item['selected']),
                     leaves=item['leaves'], role=item['role'],
                     choices=str(item.get('choices')))
                for item in result['items']])
        with (directory/'resolutions.jsonl').open('a') as output:
            output.write(json.dumps(record)+'\n')
        return result
    monkeypatch.setattr(ReferenceClose, 'preview', inspect_close)
