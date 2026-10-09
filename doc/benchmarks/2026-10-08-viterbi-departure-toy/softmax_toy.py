"""Tunable softmax vs one departure (same toy as toy.py).

softmax: every round sampled from softmax(logits / tau); one derivation per
episode; score-function credit at EVERY round, baseline = a running value
estimate per state (V[s] <- V[s] + 0.1 (C - V[s])). Given 2x the episodes of
the departure regime (a departure trial runs two derivations).
departure: toy.py's rule (greedy + one departure, mixed alternative).
Both read on the graded order and on root-only.
"""
import numpy as np, itertools, sys
from toy import run as departure_run

def softmax_run(D, K, regime, episodes, tau, lr, rng, costs):
    logits, V = {}, {}
    lg = lambda s: logits.setdefault(s, rng.normal(0, 1e-3, K))
    def episode(start):
        s, path = start, []
        while len(s) < D:
            z = lg(s) / tau; p = np.exp(z - z.max()); p /= p.sum()
            a = rng.choice(K, p=p); path.append((s, a, p)); s = s + (a,)
        C = costs[s]
        for st, a, p in path:
            b = V.get(st, 0.5); g = -p; g[a] += 1
            logits[st] = lg(st) + lr * (b - C) * g / tau
            V[st] = b + 0.1 * (C - b)
    states_at = lambda d: list(itertools.product(range(K), repeat=d))
    if regime == 'root':
        for _ in range(episodes): episode(())
    else:
        per = episodes // D
        for k in range(1, D + 1):
            st = states_at(D - k)
            for _ in range(per): episode(st[rng.integers(len(st))])
    s = ()
    while len(s) < D: s = s + (int(np.argmax(lg(s))),)
    return costs[s] - min(costs.values())

def main(D=5, K=3, n=200, episodes=6000, seed=0):
    rng = np.random.default_rng(seed)
    inst = []
    for i in range(n):
        leaves = list(itertools.product(range(K), repeat=D))
        inst.append(dict(zip(leaves, rng.random(len(leaves)))))
    rows = [('departure phi=0.3', lambda r, c, g: departure_run(D, K, r, episodes, 0.3, 2.0, g, c))]
    for tau in (1.0, 0.3, 0.1, 0.03):
        rows.append((f'softmax tau={tau}', lambda r, c, g, t=tau: softmax_run(D, K, r, 2*episodes, t, 0.5, g, c)))
    print(f"D={D} K={K} instances={n} departure episodes={episodes} (softmax {2*episodes})")
    for name, f in rows:
        out = []
        for regime in ('root', 'graded'):
            v = np.array([f(regime, c, np.random.default_rng(seed*1000+i)) for i, c in enumerate(inst)])
            out.append(f"{regime}: optimal {np.mean(v<1e-12):.2f} regret {v.mean():.3f}")
        print(f"  {name:18s} " + "   ".join(out))

if __name__ == '__main__':
    main(**{k: int(v) for k, v in (a.split('=') for a in sys.argv[1:])})
