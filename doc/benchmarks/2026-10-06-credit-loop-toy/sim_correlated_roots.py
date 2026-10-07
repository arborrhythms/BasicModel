"""Toy of the credit loop: 2 operators, affine reader, SCG with the answer term.
Conjunction roots: 4 independent directions (XOR-separable). Disjunction roots:
sums of word vectors (XOR affinely inseparable, floor .25). Walk-first draw:
half the sentences get a compose departure, the other half tie (no credit)."""
import numpy as np, sys
D=8; EPOCHS=400; SEEDS=20
rows=[(0,0),(0,1),(1,0),(1,1)]; y=np.array([0,1,1,0.])
def adam():
    return dict(m=0.,v=0.,t=0)
def step(p,g,st,lr):
    st['t']+=1; st['m']=.9*st['m']+.1*g; st['v']=.999*st['v']+.001*g*g
    mh=st['m']/(1-.9**st['t']); vh=st['v']/(1-.999**st['t'])
    return p-lr*mh/(np.sqrt(vh)+1e-8)
def run(rule,seed,lr_reader=.01,lr_policy=.01,comp_reader=False):
    rng=np.random.default_rng(seed)
    a=rng.normal(size=(2,D)); b=rng.normal(size=(2,D)); e=rng.normal(size=(4,D))
    rc=np.array([e[k]/np.linalg.norm(e[k]) for k in range(4)])           # conjunction roots
    rd=np.array([(a[i]+b[j]-0.7*rc[k])/np.linalg.norm(a[i]+b[j]-0.7*rc[k]) for k,(i,j) in enumerate(rows)])  # disjunction roots
    X=lambda r: np.concatenate([r,np.ones((4,1))],1)
    w=rng.normal(size=D+1)*.1; wst=adam()
    wc=w.copy(); wcst=adam()            # comparison reader (fallback variant)
    theta=rng.normal()*.5; tst=adam()   # p(conj)=sigmoid(theta)
    flip=None
    for ep in range(EPOCHS):
        pc=1/(1+np.exp(-theta)); greedy_conj=pc>=.5
        if greedy_conj and flip is None: flip=ep
        rg=rc if greedy_conj else rd; rx=rd if greedy_conj else rc
        p_dep=(1-pc) if greedy_conj else pc           # policy prob of the departure action
        dep=rng.random(4)<.5                           # walk-first: half compose departures
        reader_cmp=wc if comp_reader else w
        Ag=(X(rg)@reader_cmp-y)**2; Ax=(X(rx)@reader_cmp-y)**2
        adv=np.where(dep,Ax-Ag,0.)                     # R ties; advantage = answer difference
        # surrogate K*R_walk*W*p(a_dep)*adv, K=1,R_walk=1,W=2 ; d/dtheta of p_dep
        dp=p_dep*(1-p_dep)*(-1 if greedy_conj else 1)
        g_theta=np.sum(2*dp*adv)/4
        theta=step(theta,g_theta,tst,lr_policy)
        # reader step by rule
        if rule=='kept': wg=np.ones(4); wx=np.zeros(4)
        elif rule=='both': wg=np.where(dep,.5,1.); wx=np.where(dep,.5,0.)
        elif rule=='policy': wg=np.where(dep,1-p_dep,1.); wx=np.where(dep,p_dep,0.)
        elif rule=='fallback': wg=np.ones(4); wx=np.zeros(4)
        def grad(wv,wg_,wx_):
            g=np.zeros(D+1)
            for k in range(4):
                g+=wg_[k]*2*(X(rg)[k]@wv-y[k])*X(rg)[k]+wx_[k]*2*(X(rx)[k]@wv-y[k])*X(rx)[k]
            return g/4
        w=step(w,grad(w,wg,wx),wst,lr_reader)
        if comp_reader:
            wc=step(wc,grad(wc,np.where(dep,.5,1.),np.where(dep,.5,0.)),wcst,lr_reader)
    pc=1/(1+np.exp(-theta)); rg=rc if pc>=.5 else rd
    mse=np.mean((X(rg)@w-y)**2); return pc>=.5, mse, flip
SEEDS=40
for rule,cr in [('policy',False),('fallback',True)]:
    res=[run(rule,s,comp_reader=cr) for s in range(SEEDS)]
    conj=sum(c for c,_,_ in res); at0=sum(m<.05 for _,m,_ in res)
    bad=[(round(m,3),f) for c,m,f in res if m>=.05]
    print(f"{rule:9s} final conjunction {conj}/{SEEDS}  class at 0 {at0}/{SEEDS}  misses (mse, flip epoch): {bad}")
