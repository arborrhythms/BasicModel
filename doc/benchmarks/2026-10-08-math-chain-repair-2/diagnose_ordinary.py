"""Development inspection of committed references, not a gate attempt."""
from pathlib import Path
import importlib.util,json,sys,time,faulthandler
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test')]
import torch
torch.set_num_threads(1)
faulthandler.dump_traceback_later(120,repeat=True)
from test_math_chain import build_model
from math_chain_corpus import ChainDocument,flatten
from MathChainTraining import present
from ThoughtReferences import open_slots,bindings
spec=importlib.util.spec_from_file_location('frozen_observer',HERE.parent/'2026-10-07-math-chain/math_observer.py')
frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
name=sys.argv[1]
out=HERE/name;out.mkdir(exist_ok=False)
config=out/'config.xml'
config.write_text((ROOT/'data/MM_math_chain.xml').read_text().replace('<ltmCapacity>1048576</ltmCapacity>','<ltmCapacity>8192</ltmCapacity>').replace('<attentionBudget>64</attentionBudget>','<attentionBudget>32</attentionBudget>'))
model=build_model(config)
data=model.inputSpace.data
docs=(ChainDocument('a',('y is five.','what is y ?','the answer is five.'),question=1,answer='five'),
      ChainDocument('b',('x is three.','what is x ?','the answer is three.'),question=1,answer='three'))
texts,labels,addresses=flatten(docs,supplied=True)
data.train_input,data.train_output=texts,[torch.zeros(1) for _ in texts]
data.text_answers['train']=labels
data.source_addresses['train']=[dict(a,split='train') for a in addresses]
data.math_chain_documents={'train':docs}
store=model.symbolSpace.ltm_store
all_rows=set()
start=time.monotonic()
def details(value):
    return dict(mask=value.role_mask.tolist(),refs=value.role_refs,kind=value.sentence_kind,
                opened=open_slots(value),bindings=bindings(value))
with frozen.Observer(out) as observer:
    from Models import BasicModel
    from unittest.mock import patch
    from episode_state import snapshot,difference
    original=BasicModel.run_selected_thought
    footprints=[]
    def footprint(model,*args,**kwargs):
        before=snapshot(model) if not footprints else None
        result=original(model,*args,**kwargs)
        if before is not None:
            footprints.append(difference(before,snapshot(model)))
            (out/'episode-state-diff.json').write_text(json.dumps(footprints,indent=2)+'\n')
        return result
    observer.stack.enter_context(patch.object(BasicModel,'run_selected_thought',footprint))
    commit=BasicModel._commit_sentence
    programs=(out/'programs.jsonl').open('w')
    observer.stack.callback(programs.close)
    def capture(model,state,sid,active,observations,predictions,wins):
        for row,win in enumerate(wins.tolist()):
            program=observations[int(win)]['entries'][row]
            if program is None: continue
            actions=[]
            for index,(kind,op,word) in enumerate(program.actions.tolist()):
                rules=model.languageSpace._compose_binary_rules if kind==1 else model.languageSpace._compose_unary_rules
                actions.append(dict(kind=kind,operation=None if kind not in (1,2) else rules[op].method_name,
                    word=word,refs=None if program.operation_refs is None else program.operation_refs[index].tolist()))
            programs.write(json.dumps(dict(epoch=observer.context['epoch'],row=row,
                forms=program.lexical_forms,ids=program.concept_ids.tolist(),actions=actions))+'\n')
            programs.flush()
        return commit(model,state,sid,active,observations,predictions,wins)
    observer.stack.enter_context(patch.object(BasicModel,'_commit_sentence',capture))
    def after(model,split,rows,result):
        observer.after_batch(model,split,rows,result)
        fields=model._sentence_fields[0]
        episodes=dict(model._last_closing_thoughts)
        for row,source in enumerate(rows):
            field=fields[row]
            print(json.dumps(dict(epoch=observer.context['epoch'],text=texts[source],
                source=source,field=details(field.meaning), query=None if field.query is None else details(field.query),
                row_id=field.row_id,episode=row in episodes,
                work=None if row not in episodes else episodes[row].work.spent)),flush=True)
            all_rows.add(field.row_id)
        for row in sorted(all_rows):
            index=store.index_of_row(row)
            if index is not None:
                print(json.dumps(dict(stored=row,record_kind=store.KINDS[int(store.record_kind[index])],
                    relation=int(store.rel_type[index]),native_refs=store.refs[index].tolist(),
                    **details(store.meaning_of(index)))),flush=True)
        print('seconds',time.monotonic()-start,'occupancy',len(store),flush=True)
    try:
        optimizer=model.getOptimizer(lr=.001)
        for epoch in range(1,1+(int(sys.argv[2]) if len(sys.argv)>2 else 3)):
            observer.context=dict(epoch=epoch,phase='development')
            print(present(model,split='train',optimizer=optimizer,after_batch=after),flush=True)
    finally:
        model.End()
