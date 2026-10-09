"""One departure + greedy completion, credited at the end (score function).

Tabular chooser over a depth-D, branching-K derivation; terminal cost only.
Regimes (same episode budget):
  root     every episode starts at the root (full-depth items only)
  graded   stage k = 1..D: episodes start at random depth-(D-k) states
           ("one more layer at a time"), the last stage from the root
  mixed    start depth drawn uniformly each episode (all subproblems, unordered)
Alternative at the departure: phi uniform over non-greedy, else the policy
with greedy excluded (support over every alternative while phi > 0).
Reported: fraction of instances whose final greedy path from the root is
optimal; mean regret. Also the 'trap' subset: instances where the optimal
first step's subtree has a worse MEAN cost than another first step.
"""
import numpy as np, itertools, sys

def run(D, K, regime, episodes, phi, lr, rng, costs):
    logits = {}
    def lg(s):
        if s not in logits: logits[s] = rng.normal(0, 1e-3, K)
        return logits[s]
    def greedy_from(s):
        while len(s) < D: s = s + (int(np.argmax(lg(s))),)
        return s
    def episode(start):
        g = greedy_from(start)
        t = rng.integers(len(start), D)            # where: uniform over rounds
        s = g[:t]; ga = g[t]
        others = [a for a in range(K) if a != ga]
        if rng.random() < phi: a = rng.choice(others)
        else:
            p = np.exp(lg(s) - lg(s).max()); p[ga] = 0
            p = p / p.sum() if p.sum() > 0 else np.where(np.arange(K) == ga, 0., 1. / (K - 1))
            a = rng.choice(K, p=p)
        e = greedy_from(s + (a,))
        adv = costs[g] - costs[e]                  # >0: explore better
        p = np.exp(lg(s) - lg(s).max()); p /= p.sum()
        grad = -p; grad[a] += 1
        logits[s] = lg(s) + lr * adv * grad
    states_at = lambda d: list(itertools.product(range(K), repeat=d))
    if regime == 'root':
        for _ in range(episodes): episode(())
    elif regime == 'graded':
        per = episodes // D
        for k in range(1, D + 1):
            starts = states_at(D - k)
            for _ in range(per): episode(starts[rng.integers(len(starts))])
    elif regime == 'mixed':
        for _ in range(episodes):
            d = rng.integers(0, D); starts = states_at(d)
            episode(starts[rng.integers(len(starts))])
    g = greedy_from(())
    return costs[g] - min(costs.values())

def main(D=5, K=3, n=200, episodes=6000, phi=0.3, lr=2.0, seed=0):
    rng = np.random.default_rng(seed)
    res = {r: [] for r in ('root', 'graded', 'mixed')}; trap = []
    for i in range(n):
        leaves = list(itertools.product(range(K), repeat=D))
        costs = dict(zip(leaves, rng.random(len(leaves))))
        best = min(costs, key=costs.get)
        means = [np.mean([c for l, c in costs.items() if l[0] == a]) for a in range(K)]
        trap.append(int(np.argmin(means)) != best[0])
        for r in res: res[r].append(run(D, K, r, episodes, phi, lr, np.random.default_rng(seed*1000+i), costs))
    trap = np.array(trap)
    print(f"D={D} K={K} instances={n} episodes={episodes} phi={phi}  (trap instances: {trap.sum()})")
    for r, v in res.items():
        v = np.array(v)
        print(f"  {r:7s} optimal {np.mean(v<1e-12):.2f}  regret {v.mean():.3f}"
              f"  | on traps: optimal {np.mean(v[trap]<1e-12):.2f}")

if __name__ == '__main__':
    kw = dict(a.split('=') for a in sys.argv[1:])
    main(**{k: (float(v) if k in ('phi','lr') else int(v)) for k, v in kw.items()})
