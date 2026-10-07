"""Round-5 fixed point: a soft attention filter m_i = sigmoid(theta . f_i + b) on
detached word codes; an unattended word drops out of the conjunction by
interpolating toward the operation's identity (all-ones direction); the
affine answer reader reads the normalized product root; budget lambda*sum(m).
Gate: XOR over a content pair embedded among filler words."""
import numpy as np
rng=np.random.default_rng(0); d=64
content_a=['hello','loving']; content_b=['world','there']; fillers=[f'f{i}' for i in range(8)]
codes={w:rng.normal(size=d)/np.sqrt(d)+1.0/np.sqrt(d) for w in content_a+content_b+fillers}
for w in codes: codes[w]/=np.linalg.norm(codes[w])
def make(n):
    data=[]
    for _ in range(n):
        i,j=rng.integers(2),rng.integers(2); words=[content_a[i],content_b[j]]+list(rng.choice(fillers,rng.integers(2,5),replace=False))
        rng.shuffle(words); data.append((words,float(i^j)))
    return data
train,test=make(64),make(256)
def forward(words,theta,b,use_filter):
    F=np.array([codes[w] for w in words])
    m=1/(1+np.exp(-(F@theta+b))) if use_filter else np.ones(len(words))
    X=m[:,None]*F+(1-m)[:,None]*np.ones(d)/np.sqrt(d)*np.sqrt(d)/np.sqrt(d)
    X=m[:,None]*F+(1-m)[:,None]*1.0          # identity of the product = ones
    p=np.prod(X,0); n=np.linalg.norm(p)+1e-12; return F,m,X,p,p/n,n
def run(use_filter,lam=.01,epochs=600,lr=.05,b0=0.,warm=0):
    W=np.zeros(d); c=0.; theta=np.zeros(d); b=b0
    sW=[0,0];sc=[0,0];st=[0,0];sb=[0,0]
    def adam(x,g,s,t):
        s[0]=.9*s[0]+.1*g; s[1]=.999*s[1]+.001*g*g; return x-lr*(s[0]/(1-.9**t))/(np.sqrt(s[1]/(1-.999**t))+1e-8)
    for t in range(1,epochs+1):
        gW=np.zeros(d); gc=0.; gt=np.zeros(d); gb=0.
        for words,y in train:
            F,m,X,p,r,n=forward(words,theta,b,use_filter)
            e=r@W+c-y; gW+=2*e*r; gc+=2*e
            if use_filter:
                dr=2*e*W                       # dL/dr
                dp=(dr-r*(r@dr))/n             # through normalization
                for k in range(len(words)):
                    others=np.prod(np.delete(X,k,0),0)
                    dX=dp*others               # dL/dX_k
                    dm=dX@(F[k]-1.0)+(lam if t>warm else 0.)      # X_k = m F + (1-m) 1 ; budget
                    s=m[k]*(1-m[k]); gt+=dm*s*F[k]; gb+=dm*s
        N=len(train); W=adam(W,gW/N,sW,t); c=adam(c,gc/N,sc,t)
        if use_filter: theta=adam(theta,gt/N,st,t); b=adam(b,gb/N,sb,t)
    acc=np.mean([((forward(w,theta,b,use_filter)[4]@W+c)>.5)==(y>.5) for w,y in test])
    mse=np.mean([((forward(w,theta,b,use_filter)[4]@W+c)-y)**2 for w,y in test])
    mc=np.mean([forward([w],theta,b,True)[1][0] for w in content_a+content_b]) if use_filter else 1
    mf=np.mean([forward([w],theta,b,True)[1][0] for w in fillers]) if use_filter else 1
    def hard(words,mode):
        F,m,_,_,_,_=forward(words,theta,b,True)
        keep=(m>.5) if mode=='half' else np.isin(np.arange(len(words)),np.argsort(-m)[:2])
        X=np.where(keep[:,None],F,1.0); p=np.prod(X,0); r=p/(np.linalg.norm(p)+1e-12); return r@W+c
    hacc={md:np.mean([(hard(w,md)>.5)==(y>.5) for w,y in test]) for md in ('half','top2')}
    picks=np.mean([set(np.array(w)[np.argsort(-forward(w,theta,b,True)[1])[:2]])<=set(content_a+content_b) for w,_ in test])
    return acc,mse,mc,mf,hacc,picks
for name,kw in [('start attending, lam .01',dict(b0=3.)),('start attending, lam .001',dict(b0=3.,lam=.001)),
                ('start attending, budget after 200',dict(b0=3.,warm=200)),('start attending, lam .001, budget after 200',dict(b0=3.,lam=.001,warm=200))]:
    acc,mse,mc,mf,hacc,picks=run(True,**kw); print(f'{name:44s} soft acc {acc:.3f} | hard m>.5 {hacc["half"]:.3f} top-2 {hacc["top2"]:.3f} | top-2 = content pair {picks:.3f} | m content {mc:.2f} fillers {mf:.2f}')
