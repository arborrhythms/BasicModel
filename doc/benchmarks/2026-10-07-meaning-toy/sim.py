"""Round-4a fixed point: order-0 meaning as the mean of the identity codes of the
sentence rows containing the word (random indexing; codes fixed, seeded from the
row's content key); composition on the meaning block by Kleene min/max; the
semantic certificate (extent of a∧b = sentences with both; a∨b = either); the
XOR roots [projected form | meaning] under each operator and the sum control."""
import numpy as np, hashlib, itertools
def bits(key, K, s):
    seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')
    v=np.zeros(K); v[np.random.default_rng(seed).permutation(K)[:s]]=1; return v
def meanings(corpus, K, s):
    codes={S:bits('sentence/'+S,K,s) for S in corpus}
    occ={}
    for S in corpus:
        for w in S.split(): occ.setdefault(w,[]).append(S)
    return codes, occ, {w:np.mean([codes[S] for S in ss],0) for w,ss in occ.items()}
def member(root, code): return bool((root[code>0]>0).all())
def certificate(corpus,K,s,pairs=None):
    codes,occ,m=meanings(corpus,K,s)
    words=sorted(occ); errs={'and':0,'or':0}; n=0
    pairs=pairs or list(itertools.combinations(words,2))
    for a,b in pairs:
        A,B=set(occ[a]),set(occ[b])
        for S in corpus:
            n+=1
            errs['and']+= member(np.minimum(m[a],m[b]),codes[S]) != (S in A and S in B)
            errs['or'] += member(np.maximum(m[a],m[b]),codes[S]) != (S in A or S in B)
    return errs,n,m
xor=['hello world','hello there','loving world','loving there']
errs,n,m=certificate(xor,64,3)
print('XOR corpus, K=64 s=3: certificate errors',errs,'of',n,'; meanings distinct:',len({v.tobytes() for v in m.values()})==4)
# XOR separability of meaning roots under each operator and the mean (sum control)
rows=[('hello','world',0),('hello','there',1),('loving','world',1),('loving','there',0)]
for name,op in [('and',np.minimum),('or',np.maximum),('mean',lambda x,y:(x+y)/2)]:
    R=np.array([op(m[a],m[b]) for a,b,_ in rows]); y=np.array([t for *_,t in rows],float)
    X=np.concatenate([R,np.ones((4,1))],1); coef,*_=np.linalg.lstsq(X,y,rcond=None)
    sv=np.linalg.svd(R-R.mean(0),compute_uv=False)
    print(f'  meaning roots by {name:4s}: affine fit mse {np.mean((X@coef-y)**2):.3g}; centered singular values {np.round(sv,3)}')
# synthetic corpus: Zipfian words, 5-word sentences
rng=np.random.default_rng(0)
V=300; probs=1/np.arange(1,V+1); probs/=probs.sum()
corpus=list(dict.fromkeys(' '.join(f'w{i}' for i in rng.choice(V,5,replace=False,p=probs)) for _ in range(400)))
pairs=[tuple(rng.choice(sorted({w for S in corpus for w in S.split()}),2,replace=False)) for _ in range(200)]
for K,s in [(64,3),(256,4),(1024,6)]:
    errs,n,m=certificate(corpus,K,s,pairs)
    print(f'synthetic {len(corpus)} sentences, K={K} s={s}: certificate errors and {errs["and"]}, or {errs["or"]} of {n} membership checks')
# similarity: shared-sentence count vs meaning cosine
codes,occ,m=meanings(corpus,1024,6)
ws=sorted(occ); sims=[];shared=[]
for a,b in itertools.combinations(ws[:120],2):
    x,y=m[a],m[b]; sims.append(x@y/np.linalg.norm(x)/np.linalg.norm(y)); shared.append(len(set(occ[a])&set(occ[b]))/np.sqrt(len(occ[a])*len(occ[b])))
r=np.corrcoef(np.argsort(np.argsort(sims)),np.argsort(np.argsort(shared)))[0,1]
print(f'Spearman(meaning cosine, normalized shared sentences) = {r:.3f}')
