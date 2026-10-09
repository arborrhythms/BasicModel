"""Observe mint/open probabilities on one new unforced development batch.

This supplements the retained 0/8 probe; it never replaces that result or changes
its certificate status. The forward values, choices and RNG are untouched.
"""
from pathlib import Path
import json,sys
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test')]
import torch
from Language import OperationSelectionLayer
from math_chain_corpus import MathChainCorpus,ChainDocument
from math_chain_ordinary import ordinary_model,train_documents

torch.set_num_threads(1)
folder=HERE/'development-binding-probabilities-01'
folder.mkdir(exist_ok=False)
model=ordinary_model(folder)
source=MathChainCorpus()
documents=[]
for index in range(8):
    doc=source.problem((index,0),split='development',training=True)
    documents.append(ChainDocument(doc.key,(doc.sentences[doc.question],),question=0))
original=OperationSelectionLayer.forward
pairs=[]
def observed(module,x,**kwargs):
    result=original(module,x,**kwargs)
    data=kwargs.get('reference_data')
    if not data or not data.get('binding_choices'):return result
    logits=result[2]['logits'].detach().cpu()
    nr=len(data['binary_ops']);nu=len(data['unary_ops']);width=x.shape[1]
    for label,operations,positions,offset in (
        ('binary',data['binary_ops'],width-1,0),
        ('unary',data['unary_ops'],width,(width-1)*nr)):
        refs=data[label+'_choices'].detach().cpu().tolist()
        for b in range(len(x)):
            for position in range(positions):
                buckets={}
                for column,operation in enumerate(operations):
                    action=offset+position*len(operations)+column
                    if not bool(torch.isfinite(logits[b,action])):continue
                    choice=refs[b][position][column]
                    for role in range(2 if label=='binary' else 1):
                        if choice[role] not in (0,-2):continue
                        key=(operation,role,choice[1-role])
                        buckets.setdefault(key,{})[choice[role]]=action
                for key,actions in buckets.items():
                    if set(actions)!={0,-2}:continue
                    probability=float(torch.softmax(logits[b,[actions[0],actions[-2]]],0)[0])
                    pairs.append(dict(row=b,position=position,arity=label,operation=key[0],role=key[1],
                        p_open_given_mint_or_open=probability,
                        selected=int(result[2]['action'][b]) in actions.values(),
                        selected_open=int(result[2]['action'][b])==actions[0]))
    return result
try:
    with patch.object(OperationSelectionLayer,'forward',observed):
        rows=train_documents(model,documents,folder)
    values=[item['p_open_given_mint_or_open'] for item in pairs]
    result=dict(kind='development',seed=None,questions=len(rows),
        committed_open=sum(bool(row['open']) for row in rows),episodes=sum(row['episode'] for row in rows),
        departures=[row['departure'] for row in rows],legal_mint_open_pairs=len(pairs),
        conditional_open_probability=dict(minimum=min(values),maximum=max(values),mean=sum(values)/len(values)),
        selected_pairs=sum(item['selected'] for item in pairs),selected_open_pairs=sum(item['selected_open'] for item in pairs),
        note='Conditional probabilities over legal mint/open variants of the same operation and other operand; not the probability over all backward candidates or a certificate of the empirical question-opening rate.')
    (folder/'pairs.json').write_text(json.dumps(pairs)+'\n')
    (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
finally:model.End()
