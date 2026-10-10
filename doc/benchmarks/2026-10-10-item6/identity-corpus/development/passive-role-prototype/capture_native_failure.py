"""Replay the one failed cold biased preflight; capture, do not repair it."""
import sys, json
from pathlib import Path
from types import SimpleNamespace
sys.path[:0]=['bin','test']
import torch
import Models
from identity_measurement import run
out=Path('output/item6-followup/biased-failure-replay');out.mkdir(parents=True,exist_ok=True)
original=Models.BasicModel._derivation_program

def traced(self,t=None,budget=None):
    try:return original(self,t,budget)
    except ValueError as error:
        trace=self._reconstruction_stack();isp=self.inputSpace
        rules,arities,mask=trace.choices()
        field=getattr(self,'_sentence_field',None)
        records=[]
        for b in range(len(isp._word_active_mask)):
            live=isp._word_active_mask[b].nonzero().flatten().tolist()
            leaves=isp._ar_grammar_leaf_mask[b].nonzero().flatten().tolist()
            records.append(dict(batch=b,active_words=live,grammar_leaves=leaves,
                sentence_ids=isp._packed_sentence_ids[b].tolist(),
                source_order=None if field is None else field.source_order(b),
                operations=[dict(slot=i,rule=int(rules[b,i]),arity=int(arities[b,i]),
                    position=int(trace._choice_positions[b,i])) for i in mask[b].nonzero().flatten().tolist()]))
        value=dict(error=str(error),sentence=t,trial=getattr(self,'_sentence_trial',None),records=records)
        (out/'failure-context.json').write_text(json.dumps(value,indent=2)+'\n')
        torch.save(dict(rules=rules.detach(),arities=arities.detach(),mask=mask.detach(),
            positions=trace._choice_positions.detach(),word_active=isp._word_active_mask.detach(),
            grammar_leaf=isp._ar_grammar_leaf_mask.detach(),sentence_ids=isp._packed_sentence_ids.detach()),out/'failure-context.pt')
        raise
Models.BasicModel._derivation_program=traced
run(SimpleNamespace(out=out,split='reversed_eval',train_split='biased_train',
    seed=20261010,batch_size=4,epochs=1,limit=4,train_limit=4))
