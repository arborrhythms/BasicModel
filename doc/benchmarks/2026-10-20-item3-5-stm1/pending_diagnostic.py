"""Capture the original failing arrival initialization and native stores."""
import json, os
from pathlib import Path
import pytest
import torch

@pytest.fixture(autouse=True)
def inspect_pending(monkeypatch,request):
    if 'test_forced_ordinary_pending_premise_in_both_orders' not in request.node.name:
        return
    import math_chain_ordinary
    folder=Path(os.environ['PENDING_DIAGNOSTIC'])/request.node.name
    folder.mkdir(parents=True,exist_ok=True)
    original=math_chain_ordinary.ordinary_model
    def construct(*args,**kwargs):
        replay=os.environ.get('PENDING_REPLAY')
        if replay:
            saved=Path(replay)/request.node.name/'initial-rng.pt'
            torch.set_rng_state(torch.load(saved,weights_only=True))
        state=torch.get_rng_state().clone()
        (folder/'seed.txt').write_text(str(torch.initial_seed())+'\n')
        torch.save(state,folder/'initial-rng.pt')
        model=original(*args,**kwargs)
        end=model.End
        def finish(*a,**kw):
            from ThoughtReferences import bindings,open_slots
            from Spaces import _concept_alloc_of
            store=model.symbolSpace.ltm_store
            alloc=_concept_alloc_of(model._concept_owner())
            def brief(value):
                return None if value is None else dict(refs=value.role_refs,bindings=bindings(value),open=open_slots(value),roles=value.roles)
            def native(value):
                return value.detach().tolist() if torch.is_tensor(value) else str(value)
            report=dict(words={str(k):list(v) for k,v in alloc.word_forms.items()},singletons=sorted(alloc.singletons),
                rows=[dict(index=i,stream=store.stream[i],raw=brief(store.raw_meaning_of(i)),working=brief(store.meaning_of(i))) for i in range(len(store))],
                parts={int(i):store.parts.row(i) for i in store.parts.indices().tolist()})
            from PartRead import expression_components
            from QueryWork import QueryWorkBudget
            targets = {ref for row in report['rows'] if row['working'] is not None
                       for ref in row['working']['refs'] if ref and ref[0] == 'sym'
                       and store.parts.contains(ref[1])}
            report['projections'] = {}
            for ref in targets:
                terms, scanned, missing = expression_components(store, (ref,),
                    max_records=64, work=QueryWorkBudget(64))
                report['projections'][str(ref)] = dict(scanned=scanned, incomplete=missing,
                    terms={str(k): {name: val for name,val in term.items() if name != 'meaning'}
                           for k,term in terms.items()})
            report['types'] = {str(k): sorted(v) for k,v in alloc.symbol_types.items()}
            report['occupations'] = {str(k): dict(v) for k,v in model.languageSpace.parts_of_speech.counts.items()}
            (folder/'store.json').write_text(json.dumps(report,default=native,indent=2)+'\n')
            return end(*a,**kw)
        model.End=finish
        return model
    monkeypatch.setattr(math_chain_ordinary,'ordinary_model',construct)
