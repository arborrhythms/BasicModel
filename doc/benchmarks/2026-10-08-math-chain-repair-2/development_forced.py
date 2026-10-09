"""Development inspection of §14's explicitly forced ordinary fixtures."""
from pathlib import Path
import json, sys
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test')]
import torch
torch.set_num_threads(1)
from math_chain_ordinary import ordinary_model,train_documents
from math_chain_corpus import ChainDocument
from forced_math_grammar import ForcedGrammar
from Models import BasicModel
from ThoughtReferences import bindings,open_slots

folder=HERE/sys.argv[1]; folder.mkdir(exist_ok=False)
model=ordinary_model(folder)
sentences=('what is y ?', 'the answer is five.')
if len(sys.argv)>2 and sys.argv[2]=='pending':
    sentences=('y is x plus two.', 'x is three.')
documents=[ChainDocument(str(i),sentences,question=0 if sentences[0].endswith('?') else None,
                         answer='five' if sentences[0].endswith('?') else None) for i in range(2)]
commit=BasicModel._commit_sentence
def capture(model,state,sid,active,observations,predictions,wins):
    for row,win in enumerate(wins.tolist()):
        program=observations[int(win)]['entries'][row]
        if program is None:continue
        actions=[]
        for index,(kind,op,word) in enumerate(program.actions.tolist()):
            rules=model.languageSpace._compose_binary_rules if kind==1 else model.languageSpace._compose_unary_rules
            actions.append(dict(kind=kind,operation=None if kind not in (1,2) else rules[op].method_name,
                word=word,refs=program.operation_refs[index].tolist()))
        print(json.dumps(dict(program=True,row=row,explore_kept=win,ids=program.concept_ids.tolist(),actions=actions)),flush=True)
        if len(sys.argv)>2:
            def tree(value):
                return dict(refs=repr(value.refs),relation=value.relation,subject=value.subject_word_id,
                    open=open_slots(value.meaning),bindings=bindings(value.meaning),
                    children=[tree(child) for child in value.children])
            print(json.dumps(dict(clause=tree(observations[int(win)]['clauses'][row]))),flush=True)
    return commit(model,state,sid,active,observations,predictions,wins)
def after(model,split,rows,result,*_):
    store=model.symbolSpace.ltm_store
    print(json.dumps(dict(batch=rows,episodes=[(b,r.work.spent) for b,r in model._last_closing_thoughts])),flush=True)
    for index in range(len(store)):
        if int(store.rel_type[index])==store.REL_DEF:continue
        value=store.meaning_of(index)
        print(json.dumps(dict(index=index,address=store.occurrence_of(index),kind=store.KINDS[int(store.record_kind[index])],
            mode=value.mode,refs=value.role_refs,open=open_slots(value),bindings=bindings(value))),flush=True)
try:
    with ForcedGrammar(model,open_names=('what','x')) as force, patch.object(BasicModel,'_commit_sentence',capture):
        values=train_documents(model,documents,folder,after=after)
        (folder/'choices.json').write_text(json.dumps(force.records))
    print(json.dumps(values),flush=True)
finally:model.End()
