"""Thinking-loop fixed point: can greedy-vs-one-departure credit (SCG, the
compose rule) teach a chooser to chain two symbolic lookups and conclude?
World: chains a<b<c (part-of). Question row: 'a part-of ?' with the open
reference to be filled by the transitive whole c (the direct fact a<b is stored;
a<c is not). Actions each step: QUERY (fill the open ref with the best direct
match: b), CHAIN (query from the last result: b -> c), CONCLUDE (stop; the
filled reference is the answer), with a work budget of 4.
Credit (a) answer: +1 if the final binding is c, minus 0.05 per step;
(b) expectation only: the next sentence states 'a part-of c'; reward =
-(surprise) = +1 if the concluded row equals it (no supplied answer)."""
import numpy as np
rng=np.random.default_rng(0)
A={'QUERY':0,'CHAIN':1,'CONCLUDE':2}; names=list(A)
def features(state):
    # state: (filled, depth, steps)  filled in {none,b,c}
    filled,depth,steps=state; f=np.zeros(9); f[filled]=1; f[3+min(depth,2)]=1; f[6+min(steps,2)]=1; return f
def step(state,a):
    filled,depth,steps=state
    if a==A['QUERY']: return (1 if filled==0 else filled, depth, steps+1), False
    if a==A['CHAIN']: return (2 if filled==1 else filled, depth+1, steps+1), False   # b -> c
    return state, True
def episode(theta,departure=None,budget=4):
    state=(0,0,0); logp=[]; acts=[]; done=False; t=0
    while not done and t<budget:
        logits=features(state)@theta; p=np.exp(logits-logits.max()); p/=p.sum()
        a=int(np.argmax(p))
        if departure is not None and departure[0]==t: a=departure[1]
        logp.append((features(state),p,a)); acts.append(a); state,done=step(state,a); t+=1
    return state,acts,logp
def reward(state,acts,kind):
    filled,depth,steps=state
    if kind=='answer': return (1.0 if filled==2 else 0.0)-0.05*steps
    # expectation: the next sentence says 'a part-of c'; concluded row equals it iff filled==c and concluded
    concluded=acts and acts[-1]==A['CONCLUDE']
    return (1.0 if (filled==2 and concluded) else 0.0)-0.05*steps
def train(kind,seeds=20,epochs=300,lr=.3):
    wins=0
    for s in range(seeds):
        r=np.random.default_rng(s); theta=r.normal(size=(9,3))*.1
        for ep in range(epochs):
            g_state,g_acts,g_logp=episode(theta); Cg=-reward(g_state,g_acts,kind)
            # one departure: uniform round, uniform alternative action
            if not g_logp: continue
            t=r.integers(len(g_logp)); alt=[a for a in range(3) if a!=g_acts[t]]; a_dep=alt[r.integers(2)]
            x_state,x_acts,x_logp=episode(theta,departure=(t,a_dep)); Cx=-reward(x_state,x_acts,kind)
            adv=Cx-Cg
            if adv==0: continue
            f,p,_=g_logp[t]; K=2; R=len(g_logp)
            # surrogate K*R*p(a_dep)*adv ; d p(a_dep)/d theta
            dp=np.outer(f, -p*p[a_dep]); dp[:,a_dep]+=f*p[a_dep]
            theta-=lr*K*R*adv*dp
        final,acts,_=episode(theta); wins+= (final[0]==2 and acts[-1]==A['CONCLUDE'])
    return wins
for kind in ('answer','expectation'):
    print(f'{kind:12s}: chooser learns QUERY->CHAIN->CONCLUDE in {train(kind)}/20 seeds (300 episodes each)')
