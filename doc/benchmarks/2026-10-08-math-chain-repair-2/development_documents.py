"""Development, not measurement: eight real documents through the frozen observer."""
from pathlib import Path
import importlib.util,json,sys,time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test')]
import torch
torch.set_num_threads(1)
from test_math_chain import build_model
from math_chain_corpus import MathChainCorpus,flatten
from MathChainTraining import present
from ThoughtReferences import open_slots,bindings
spec=importlib.util.spec_from_file_location('frozen_observer',HERE.parent/'2026-10-08-math-chain-repair/math_observer.py')
frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
name=sys.argv[1]
out=HERE/name;out.mkdir(exist_ok=False)
config=out/'config.xml'
config.write_text((ROOT/'data/MM_math_chain.xml').read_text().replace('<ltmCapacity>1048576</ltmCapacity>','<ltmCapacity>8192</ltmCapacity>').replace('<attentionBudget>64</attentionBudget>','<attentionBudget>32</attentionBudget>'))
model=build_model(config)
data=model.inputSpace.data
corpus=MathChainCorpus()
docs=tuple(corpus.problem((i,0),split='train',training=True) for i in range(8))
texts,labels,addresses=flatten(docs,supplied=True)
data.train_input,data.train_output=texts,[torch.zeros(1) for _ in texts]
data.text_answers['train']=labels
data.source_addresses['train']=[dict(a,split='train') for a in addresses]
data.math_chain_documents={'train':docs}
observed=[]
start=time.monotonic()
with frozen.Observer(out) as observer:
    observer.context={'epoch':1,'phase':'development'}
    def after(model,split,rows,result):
        observer.after_batch(model,split,rows,result)
        fields=model._sentence_fields[0]
        episodes=dict(model._last_closing_thoughts)
        store=model.symbolSpace.ltm_store
        for row,source in enumerate(rows):
            field=fields[row]
            item=dict(text=texts[source],source=source,episode=row in episodes,
                spent=episodes[row].work.spent if row in episodes else 0,
                slots=open_slots(field.meaning),refs=field.meaning.role_refs,
                forward=bindings(field.meaning).get('_forward_references'),
                row_id=field.row_id,query=None if field.query is None else open_slots(field.query))
            observed.append(item)
            print(json.dumps(item),flush=True)
        print('batch_seconds_total',time.monotonic()-start,'store',len(store),flush=True)
        (out/'observations.json').write_text(json.dumps(observed,indent=2)+'\n')
    try:
        print(present(model,split='train',optimizer=model.getOptimizer(lr=.001),after_batch=after),flush=True)
    finally:
        model.End()
