"""Round-4b fixed point on real text (the repository's docs): a word's wholes are
the spans containing it, the narrowest being its adjacent-word pairs; a whole's
form is the join of its constituents' forms (whole >= part); the ceiling
U(w) = meet over w's adjacent pairs = L(w) OR (what all its neighbours share);
the centroid c = L + alpha (U - L), alpha = W_U/(W_P+W_U)."""
import numpy as np, hashlib, re, glob, itertools
text=' '.join(open(f,errors='ignore').read() for f in glob.glob('/Users/arogers/github/WikiOracle/basicmodel/doc/*.md'))
sents=[re.findall(r'[a-z]+',s.lower()) for s in re.split(r'[.!?\n]+',text)]
sents=[s for s in sents if len(s)>=3]
D,S,L=64,3,32
def stream(atom):
    seed=int.from_bytes(hashlib.sha256(atom.encode()).digest()[:8],'little'); return np.random.default_rng(seed).permutation(D)
def form(w):
    v=np.zeros(D+L); m='#'+w+'#'
    for i in range(len(m)-1): v[stream(m[i:i+2])[:S]]=1
    v[D:D+min(len(w),L)]=1; return v
vocab=sorted({w for s in sents for w in s}); F={w:form(w) for w in vocab}
nbrs={w:[] for w in vocab}
for s in sents:
    for a,b in zip(s,s[1:]): nbrs[a].append(b); nbrs[b].append(a)
def centroid(w):
    Lw=F[w]; nb=nbrs[w]
    if not nb: return Lw, Lw, 0.
    U=np.minimum.reduce([np.maximum(Lw,F[x]) for x in nb])     # meet of the pair wholes' joins
    WP=float(Lw[:D].sum()>0)*len(set(m for m in range(1))) or 1.0
    WP=sum(1 for _ in range(len('#'+w+'#')-1))                   # evidence of parts: number of pair atoms
    WU=len(nb)                                                   # evidence of wholes: number of adjacent occurrences
    a=WU/(WP+WU); return Lw+a*(U-Lw), U, a
C={};Us={};A={}
for w in vocab: C[w],Us[w],A[w]=centroid(w)
changed=[w for w in vocab if np.abs(C[w]-F[w]).max()>0]
print(f'{len(sents)} sentences, {len(vocab)} words; words whose centroid differs from L: {len(changed)} ({len(changed)/len(vocab):.1%})')
print('identity: L recoverable as [c==1] for all words:', all(np.array_equal((C[w]==1).astype(float),F[w]) for w in vocab))
rng=np.random.default_rng(0); sub=rng.choice(vocab,500,replace=False)
def meancos(M):
    X=np.array([M[w] for w in sub]); X=X/np.linalg.norm(X,axis=1,keepdims=True); G=X@X.T; return (G.sum()-len(sub))/(len(sub)*(len(sub)-1))
print(f'mean pairwise cosine: forms {meancos(F):.3f} -> centroids {meancos(C):.3f}')
print('mean |c-L| over changed words:', round(float(np.mean([np.abs(C[w]-F[w]).sum() for w in changed])) if changed else 0,3),
      '; mean number of raised coordinates:', round(float(np.mean([(C[w]>F[w]).sum() for w in changed])) if changed else 0,2))
# examples: the most raised words
ex=sorted(changed,key=lambda w:-(C[w]>F[w]).sum())[:8]
print('most raised:',[(w,int((C[w]>F[w]).sum()),sorted(set(nbrs[w]))[:4]) for w in ex])
# collocation pull: for word pairs that are always adjacent, does c bring them closer?
def cos(x,y): return float(x@y/np.linalg.norm(x)/np.linalg.norm(y))
pairs=[(w,set(nbrs[w])) for w in vocab if len(set(nbrs[w]))==1 and len(nbrs[w])>=2]
d=[cos(C[w],F[list(n)[0]])-cos(F[w],F[list(n)[0]]) for w,n in pairs]
print(f'words with a single repeated neighbour: {len(pairs)}; mean cosine gain toward it: {np.mean(d) if d else 0:.3f}')
# containment order among comparable words after the centroid, and the projection
pset={w:set(('#'+w+'#')[i:i+2] for i in range(len(w)+1)) for w in vocab}
post={}
for w in vocab:
    for p in pset[w]: post.setdefault(p,set()).add(w)
cont={w:[y for y in set.intersection(*(post[p] for p in pset[w]))-{w} if len(y)>=len(w)] for w in vocab}
viol=sum(bool((C[x]>C[y]+1e-12).any()) for x in vocab for y in cont[x]); npairs=sum(len(v) for v in cont.values())
Cp=dict(C)
for x in sorted(vocab,key=lambda w:-len(pset[w])):
    for y in cont[x]: Cp[x]=np.minimum(Cp[x],Cp[y])
after=sum(bool((Cp[x]>Cp[y]+1e-12).any()) for x in vocab for y in cont[x])
below=sum(bool((Cp[w]<F[w]-1e-12).any()) for w in vocab)
print(f'containment pairs {npairs}: violations by the centroid {viol} -> after projection {after}; below L after projection {below}')
