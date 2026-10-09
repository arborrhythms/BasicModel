"""Development-only record of the forced pending-arrival ordinary path."""
import json
from pathlib import Path
import sys
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT/'bin'), str(ROOT/'test')]
from forced_math_grammar import ForcedGrammar
from math_chain_ordinary import ordinary_model, train_documents
from math_chain_corpus import ChainDocument
from ThoughtReferences import bindings, open_slots

folder = Path(sys.argv[1])
folder.mkdir(exist_ok=False)
torch.set_num_threads(1)
model = ordinary_model(folder, budget=0)
records = []
def after(model, split, rows, result, observed, observer):
    store = model.symbolSpace.ltm_store
    values = []
    for index in range(len(store)):
        meaning = store.meaning_of(index)
        if store.KINDS[int(store.record_kind[index])] == 'estimate':
            continue
        values.append(dict(index=index, row_id=int(store.row_ids[index]),
            native=store.refs[index].tolist(), refs=meaning.role_refs,
            pair=[float(store.c_plus[index]),float(store.c_minus[index])],
            opened=open_slots(meaning), bindings=bindings(meaning)))
    records.append(dict(rows=rows, values=values, choices=list(forced.records)))
    (folder/'trace.json').write_text(json.dumps(records,indent=2,default=str)+'\n')
try:
    owner = model._concept_owner()
    (folder/'names.json').write_text(json.dumps({word:owner.word_concepts(word)
        for word in ('x','y','two','three')})+'\n')
    sentences = ('y is x plus two.', 'x is three.')
    if len(sys.argv)>2:
        sentences = sentences[::-1]
    docs = [ChainDocument(str(i),sentences) for i in range(8)]
    with ForcedGrammar(model,open_names=('x',),named_bindings={'x':'three'}) as forced:
        train_documents(model,docs,folder,after=after)
finally:
    model.End()
