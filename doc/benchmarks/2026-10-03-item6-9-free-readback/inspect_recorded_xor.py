"""Arithmetic on saved audit tensors only; no model construction or forward."""
from pathlib import Path
import json
import torch
R=Path(__file__).resolve().parent
folder=R/'xor-ownership'
x=torch.load(folder/'geometry-start-stage-1.pt',weights_only=True,map_location='cpu')
row_to_code={int(r):c for r,c in zip(x['rows'],x['codes'])}
roots=torch.tensor(json.loads((folder/'geometry-start.json').read_text())['roots']['values'])

def conjunction(a,b):
    product=a*b
    norm=product.norm()
    return a.norm()*b.norm()*product/(norm if norm>0 else 1)

def score(leaf,c):
    square=c.square().sum()
    if square==0:return 0.
    activation=(leaf*c).sum()/square
    cosine=(leaf*c).sum()/(leaf.norm()*c.norm())
    return float(activation*cosine)

result=[]
# The four observed modal derivations, from derivation-stability.json.
# Own candidates have equal priming. A shared weight cancels the ranking.
for i,(ids,neg_before,neg_after) in enumerate([((0,1),False,False),((0,2),False,True),((4,1),True,False),((4,2),True,True)]):
    a,b=[row_to_code[r] for r in ids]
    expected=conjunction(-a if neg_before else a,b)
    if neg_after:expected=-expected
    parent=-roots[i] if neg_after else roots[i]
    pairs=[]
    for j in range(2):
        for k in range(2):
            u,v=(a,b)[j],(a,b)[k]
            composed=u if j==k else conjunction(u,v)
            pairs.append((float((composed-parent).square().mean()),j,k))
    residual,j,k=min(pairs)
    children=[(a,b)[j],(a,b)[k]]
    if neg_before:children[0]=-children[0]
    decoded=[ids[max(range(2),key=lambda q:score(c,(a,b)[q]))] for c in children]
    result.append(dict(sentence=i,own_rows=ids,negate_before=neg_before,negate_after=neg_after,
        root_vs_recorded_derivation_max_error=float((roots[i]-expected).abs().max()),
        best_raw_bank_pair=[ids[j],ids[k]],best_raw_bank_parent_mse=residual,
        recovered_word_rows=decoded, pair_residuals=pairs))
(R/'recorded-root-diagnostic.json').write_text(json.dumps(dict(
 scope='Saved first XOR audit only; no model was constructed, trained, or called. Equal own-row priming is a common positive score multiplier, omitted here.',
 observations=result),indent=2))
print(json.dumps(result,indent=2))
