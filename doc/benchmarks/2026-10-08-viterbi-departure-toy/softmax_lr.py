import numpy as np, itertools
from softmax_toy import softmax_run
D,K,n=5,3,50; rng=np.random.default_rng(0)
inst=[dict(zip(itertools.product(range(K),repeat=D),rng.random(K**D))) for _ in range(n)]
for tau,lr in ((1.0,0.1),(1.0,2.0),(0.3,0.05),(0.3,0.2),(0.1,0.01),(0.1,0.05)):
    v=np.array([softmax_run(D,K,'graded',12000,tau,lr,np.random.default_rng(i),c) for i,c in enumerate(inst)])
    print(f"softmax tau={tau} lr={lr}: graded optimal {np.mean(v<1e-12):.2f} regret {v.mean():.3f}")
