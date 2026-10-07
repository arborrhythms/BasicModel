"""Round-3b fixed point: has-a wholes (extents of parts) weighted by narrowing,
the centroid between L and U, and the containment projection. Static computation
on the 3a forms (pairs OR, 3 of 64, thermometer 32) over 20k dictionary words."""
import numpy as np, hashlib
rng=np.random.default_rng(0)
words=[w.strip().lower() for w in open('/usr/share/dict/words') if w.strip().isalpha()]
words=[w for w in words if 3<=len(w)<=12]
sample=list(dict.fromkeys(rng.choice(words,20000,replace=False).tolist()))
D,S,L=64,3,32
def stream(atom):
    seed=int.from_bytes(hashlib.sha256(atom.encode()).digest()[:8],'little'); return np.random.default_rng(seed).permutation(D)
def pairs(w): w='#'+w+'#'; return list(dict.fromkeys(w[i:i+2] for i in range(len(w)-1)))
codes={}
def code(p):
    if p not in codes: v=np.zeros(D+L); v[stream(p)[:S]]=1; codes[p]=v
    return codes[p]
N=len(sample); W=D+L
Lform=np.zeros((N,W))
for i,w in enumerate(sample):
    for p in pairs(w): Lform[i]=np.maximum(Lform[i],code(p))
    Lform[i,D:D+min(len(w),L)]=1
# has-a wholes: extent of each pair; symbol = join of members (its U = 1, no higher wholes)
ext={}
for i,w in enumerate(sample):
    for p in pairs(w): ext.setdefault(p,[]).append(i)
whole_code={p:Lform[idx].max(0) for p,idx in ext.items()}
narrow={p:1-len(idx)/N for p,idx in ext.items()}
def ceiling(i,w,use_narrowing=True):
    U=np.ones(W)
    for p in pairs(w):
        s=narrow[p] if use_narrowing else 1.
        U=np.minimum(U,1-s*(1-whole_code[p]))
    return U
def centroid(i,w,use_narrowing=True,per_coordinate=False):
    U=ceiling(i,w,use_narrowing); WP=len(pairs(w))
    if per_coordinate:
        # a whole weighs on a coordinate only where it narrows it
        wU=np.zeros(W)
        for p in pairs(w):
            s=narrow[p] if use_narrowing else 1.
            wU+=s*(1-whole_code[p])
        return (WP*Lform[i]+wU*U)/(WP+wU)
    WU=sum((narrow[p] if use_narrowing else 1.) for p in pairs(w))
    return (WP*Lform[i]+WU*U)/(WP+WU)
for use,pc in ((True,False),(True,True)):
    C=np.array([centroid(i,w,use,pc) for i,w in enumerate(sample)])
    # identity: distinct rows?
    keys={}
    for i in range(N): keys.setdefault(C[i].round(6).tobytes(),[]).append(i)
    coll=sum(len(g)>1 for g in keys.values())
    # geometry: mean pairwise cosine on a subsample vs L
    sub=rng.choice(N,400,replace=False)
    def meancos(M):
        X=M[sub]; X=X/np.linalg.norm(X,axis=1,keepdims=True); G=X@X.T; return (G.sum()-400)/(400*399)
    # containment order on comparable pairs (part-set inclusion incl. length)
    pset=[set(pairs(w)) for w in sample]; ln=[len(w) for w in sample]
    post={}
    for i,w in enumerate(sample):
        for p in pairs(w): post.setdefault(p,set()).add(i)
    viol=0; pairs_n=0; viol_after=0
    Cp=C.copy()
    # projection: cap contained by containers, decreasing part count
    order=sorted(range(N),key=lambda i:-len(pset[i]))
    containers={}
    for i,w in enumerate(sample):
        cont=set.intersection(*(post[p] for p in pairs(w)))-{i}
        containers[i]=[j for j in cont if ln[j]>=ln[i]]
    for i in order:
        for j in containers[i]:
            pairs_n+=1; viol+=bool((C[i]>C[j]+1e-9).any())
    for i in sorted(range(N),key=lambda i:-len(pset[i])):   # containers first (more parts)
        pass
    for i in sorted(range(N),key=lambda i:len(pset[i]),reverse=True):
        for j in containers[i]: Cp[i]=np.minimum(Cp[i],Cp[j])
    for i in range(N):
        for j in containers[i]: viol_after+=bool((Cp[i]>Cp[j]+1e-9).any())
    print(f"narrowing={use} per_coordinate={pc}: centroid collisions {coll}/{N}; mean |c-L| {np.abs(C-Lform).mean():.3f}; mean cos L {meancos(Lform):.3f} -> centroid {meancos(C):.3f}; "
          f"containment pairs {pairs_n}, violations before {viol}, after projection {viol_after}; below-L after projection {int((Cp<Lform-1e-9).any(1).sum())}")
