"""Round-3 fixed point: sparse pair forms, their injectivity, and the binding kernel.
Forms: OR of boundary-marked adjacent-pair codes (s ones of D) plus length.
Binding variants for the XOR gate words: product on sparse codes, product on a
fixed dense projection, permutation-OR. Measures: anagram/injectivity over a
dictionary sample, XOR affine separability of the 4 roots, exact unbinding by
bank search."""
import numpy as np, itertools, re
rng=np.random.default_rng(0)
words=[w.strip().lower() for w in open('/usr/share/dict/words') if w.strip().isalpha()]
words=[w for w in words if 3<=len(w)<=12]
sample=list(dict.fromkeys(rng.choice(words,20000,replace=False).tolist()))
gate=['hello','world','loving','there']
def pairs(w): w='#'+w+'#'; return {w[i:i+2] for i in range(len(w)-1)}
inventory=sorted({p for w in sample+gate for p in pairs(w)})
print('pairs in inventory',len(inventory))
def codes(D,s):
    C={}
    for p in inventory:
        v=np.zeros(D,bool); v[rng.choice(D,s,replace=False)]=True; C[p]=v
    return C
def form(w,C,D):
    v=np.zeros(D,bool)
    for p in pairs(w): v|=C[p]
    return v
for D,s in [(32,2),(64,3),(128,3),(256,4)]:
    C=codes(D,s)
    keys={}
    coll=0
    for w in sample+gate:
        k=(form(w,C,D).tobytes(),len(w))
        if k in keys and keys[k]!=w: coll+=1
        keys.setdefault(k,w)
    # anagram pairs in the sample
    bysort={}
    for w in sample: bysort.setdefault(''.join(sorted(w)),[]).append(w)
    ana=[g for g in bysort.values() if len(g)>1]
    ana_sep=sum(len({(form(w,C,D).tobytes(),len(w)) for w in g})==len(g) for g in ana)
    print(f'D={D:3d} s={s}: collisions among {len(sample)+4} words: {coll}; anagram groups {len(ana)}, separated {ana_sep}')
# binding on the gate words at D=128,s=3
D,s=128,3; C=codes(D,s)
U={w:form(w,C,D).astype(float) for w in gate}
W=rng.normal(size=(D,64))/8
perm1=rng.permutation(D); perm2=rng.permutation(D)
def unit(x): n=np.linalg.norm(x); return x/n if n>0 else x
def bind(kind,u,v):
    if kind=='product-sparse': return unit(u*v)
    if kind=='product-projected': return unit((u@W)*(v@W))
    if kind=='perm-or': return unit(np.maximum(u[perm1],v[perm2]))
    if kind=='perm-sum': return unit(u[perm1]+v[perm2])
rows=[('hello','world',0),('hello','there',1),('loving','world',1),('loving','there',0)]
bank=[(a,b) for a in ['hello','loving'] for b in ['world','there']]
for kind in ['product-sparse','product-projected','perm-or','perm-sum']:
    R=np.array([bind(kind,U[a],U[b]) for a,b,_ in rows]); y=np.array([t for *_,t in rows],float)
    X=np.concatenate([R,np.ones((4,1))],1)
    coef,res,rank,_=np.linalg.lstsq(X,y,rcond=None); mse=np.mean((X@coef-y)**2)
    # unbinding by bank search over all 16 ordered pairs of the 4 words
    hits=0
    for (a,b,_),r in zip(rows,R):
        best=min(((np.linalg.norm(r-bind(kind,U[p],U[q])),(p,q)) for p in gate for q in gate))
        hits+= set(best[1])=={a,b}
    print(f'{kind:18s} root norms {np.round(np.linalg.norm(R,axis=1),2)} XOR affine fit mse {mse:.3g} rank {rank}  unbinding {hits}/4')
# which collisions / unseparated anagrams remain at D=128?
D,s=128,3; C=codes(D,s); keys={}
for w in sample+gate:
    k=(form(w,C,D).tobytes(),len(w)); keys.setdefault(k,[]).append(w)
print('colliding groups:',[g for g in keys.values() if len(g)>1][:5])
# genuine pair-set + length collisions (independent of codes)
pk={}
for w in sample: pk.setdefault((frozenset(pairs(w)),len(w)),[]).append(w)
print('same pair-set and length:',[g for g in pk.values() if len(g)>1][:5])
# short-word empty products at s=3: probability two random short words share no coordinate
import math
for npairs in (3,5,8):
    ones=min(D,npairs*s); p_empty=(1-ones/D)**ones
    print(f'{npairs} pairs/word -> ~{ones} ones; P(empty product) ~ {p_empty:.2f}')
