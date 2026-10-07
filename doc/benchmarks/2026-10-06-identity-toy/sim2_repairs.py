"""Repairs for Codex's three findings: reserved thermometer length block; a mint
that draws bits from each word's own positional atom until the forms separate;
corrected containment witnesses. Same 20k dictionary sample as sim.py."""
import numpy as np, hashlib
rng=np.random.default_rng(0)
words=[w.strip().lower() for w in open('/usr/share/dict/words') if w.strip().isalpha()]
words=[w for w in words if 3<=len(w)<=12]
sample=list(dict.fromkeys(rng.choice(words,20000,replace=False).tolist()))
D,S,L=64,3,32
def stream(atom):
    """Deterministic bit positions for an atom: a permutation of D seeded from its bytes."""
    seed=int.from_bytes(hashlib.sha256(atom.encode()).digest()[:8],'little')
    return np.random.default_rng(seed).permutation(D)
def pairs(w): w='#'+w+'#'; return [w[i:i+2] for i in range(len(w)-1)]
def code(atom,bits): v=np.zeros(D,bool); v[stream(atom)[:bits]]=True; return v
extra={}   # word -> list of (atom, bits) minted
def form(w):
    v=np.zeros(D+L,bool)
    for p in set(pairs(w)): v[:D]|=code(p,S)
    v[D:D+min(len(w),L)]=True                  # reserved thermometer, exact up to L
    for atom,bits in extra.get(w,[]): v[:D]|=code(atom,bits)
    return v
def key(w): return form(w).tobytes()
def triples_at(w,i): t='#'+w+'#'; return t[i:i+3] if i+3<=len(t) else None
def mint(a,b):
    """Each word gets its own atom at the first differing position; draw bits until separated."""
    ta,tb='#'+a+'#','#'+b+'#'
    for i in range(max(len(ta),len(tb))-2):
        x,y=triples_at(a,i),triples_at(b,i)
        if x is None or y is None or x==y: continue
        for bits in range(S,D+1):
            extra[a]=extra.get(a,[])+[(f'{x}@{i}',bits)]; extra[b]=extra.get(b,[])+[(f'{y}@{i}',bits)]
            if key(a)!=key(b): return (f'{x}@{i}',f'{y}@{i}',bits)
            extra[a].pop(); extra[b].pop()
    return None
groups={}
for w in sample: groups.setdefault(key(w),[]).append(w)
coll=[g for g in groups.values() if len(g)>1]
print('collision groups before minting:',len(coll),coll[:6])
mints=[]
for g in coll:
    for i in range(len(g)):
        for j in range(i+1,len(g)):
            if key(g[i])==key(g[j]): mints.append((g[i],g[j],mint(g[i],g[j])))
print('mints:',mints)
groups={}
for w in sample: groups.setdefault(key(w),[]).append(w)
print('collision groups after minting:',sum(len(g)>1 for g in groups.values()))
# length-only collisions
print('aaaaaa vs aaaaaaa distinct:', key('aaaaaa')!=key('aaaaaaa'))
# containment witnesses with boundary pairs
def contained(x,y): return set(pairs(x))<=set(pairs(y)) and len(x)<=len(y)
for x,y in [('bana','banana'),('cat','concat'),('aba','ababa'),('an','and'),('an','ant')]:
    fx,fy=form(x),form(y); print(f'{x}<={y}: parts {contained(x,y)}, form order {bool((fx<=fy).all())}')
