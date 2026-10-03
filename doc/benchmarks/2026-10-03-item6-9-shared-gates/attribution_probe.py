"""Receipt-local removal of objective writers, never a production switch."""
import difflib,importlib.util,inspect,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test'),str(HERE)]
arm,output=sys.argv[1],Path(sys.argv[2])
import ObjectiveOwnership,Models
original=ObjectiveOwnership.registry_costs

def restricted(registry, **kwargs):
    costs=original(registry, **kwargs)
    kept={'reconstruction','compose_lesson','penalty.reconstruction'}
    if 'E' in arm:kept|={'expectation','penalty.expectation'}
    if 'A' in arm:kept|={'output','generate_lesson','penalty.output'}
    return {name:value for name,value in costs.items() if name in kept}
ObjectiveOwnership.registry_costs=restricted
if 'A' not in arm:
    def no_trial_answer(self,*args,**kwargs):
        self._sentence_answer_cost=None
        self._sentence_answer_raw_cost=None
        return None
    before=inspect.getsource(Models.BasicModel._sentence_answer_error)
    after=inspect.getsource(no_trial_answer)
    Models.BasicModel._sentence_answer_error=no_trial_answer
    (output/'answer-reader.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True))))
(output/'writer-patch.txt').write_text(inspect.getsource(original)+'\n'+inspect.getsource(restricted))
(output/'plan.json').write_text(json.dumps(dict(arm=arm,seed=None,epochs=400,
  change='Filter objective-owned backwards. A-off also omits the in-trial reader objective. Trial selection stays reconstruction-only. Same fixture, targets, bars and optimizer cadence.'),indent=2))
source=ROOT/'doc/benchmarks/2026-10-01-item6-9-review/separator_campaign.py'
spec=importlib.util.spec_from_file_location('prior_gate',source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
mod.child(arm,output,str(ROOT/'data/XOR_grammar.xml'))
