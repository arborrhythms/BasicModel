"""Can a learned VP generalize 'plus' over opaque numerals? Numerals 0..19: form =
random sparse identity; meaning = context mean of the sentence codes of the facts
each numeral appears in (counting facts + the training addition facts). A verb map
f(meaning(a), meaning(b)) -> meaning(a+b), read out as the nearest numeral, trained
on 70% of pairs, tested on the held-out 30%. MLP, 3 seeds."""
import numpy as np
rng=np.random.default_rng(0); N=20; K=128; s=4
def code(key,r): v=np.zeros(K); v[r.permutation(K)[:s]]=1; return v
pairs=[(a,b) for a in range(N) for b in range(N) if a+b<N]
def run(seed,hidden=64,epochs=3000):
    r=np.random.default_rng(seed); r.shuffle(pairs)
    cut=int(.7*len(pairs)); train,test=pairs[:cut],pairs[cut:]
    facts=[('count',n) for n in range(N-1)]+[('add',a,b) for a,b in train]
    fcode={f:code(f,r) for f in facts}
    occ={n:[] for n in range(N)}
    for f in facts:
        for n in (f[1:] if f[0]=='count' else (f[1],f[2],f[1]+f[2])): occ[n].append(fcode[f])
        if f[0]=='count': occ[f[1]+1].append(fcode[f])
    if LINE:   # a smooth number line in meaning: each numeral's meaning = a Gaussian bump over position (what rich counting/ordering contexts would give)
        pos=np.arange(N); M=np.exp(-(pos[:,None]-np.linspace(0,N-1,K)[None,:])**2/(2*2.0**2))
    else:
        M=np.array([np.mean(occ[n],0) for n in range(N)])
    M/=np.linalg.norm(M,axis=1,keepdims=True)+1e-9
    X=lambda ps: np.array([np.concatenate([M[a],M[b]]) for a,b in ps]); Y=lambda ps: np.array([M[a+b] for a,b in ps])
    W1=r.normal(size=(2*K,hidden))*.05; b1=np.zeros(hidden); W2=r.normal(size=(hidden,K))*.05; b2=np.zeros(K)
    Xt,Yt=X(train),Y(train); lr=.05
    for ep in range(epochs):
        H=np.tanh(Xt@W1+b1); P=H@W2+b2; G=2*(P-Yt)/len(Xt)
        W2-=lr*H.T@G; b2-=lr*G.sum(0); dH=(G@W2.T)*(1-H**2); W1-=lr*Xt.T@dH; b1-=lr*dH.sum(0)
    def acc(ps):
        P=np.tanh(X(ps)@W1+b1)@W2+b2; pred=np.argmax(P@M.T,1); return np.mean([p==a+b for p,(a,b) in zip(pred,ps)])
    return acc(train),acc(test),1/N
for LINE in (False,True):
    for seed in range(3):
        tr,te,chance=run(seed); print(f'line={LINE} seed {seed}: train {tr:.2f}  held-out pairs {te:.2f}  (chance {chance:.2f})')
