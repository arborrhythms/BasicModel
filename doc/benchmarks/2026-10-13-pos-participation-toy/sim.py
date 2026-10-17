#!/usr/bin/env python
"""Parts of speech from operand participation: a numpy toy of plan §6 / §6.1
(doc/plans/2026-10-10-item-6-stored-idea-generativity.md).

What is simulated
-----------------
* A hidden grammar with hidden word classes generates sentences and their true
  parse. A parse is a list of fills (word, operand position). As in
  ``record_category_observations`` (bin/Language.py) only a word that is the
  DIRECT operand of an operation is a fill; a composed phrase is not
  (``heads=True`` adds the head word of a phrase as a fill, a variant).
* Learner (plan §6.1): one pointer per word (word -> class id), class vectors
  in position space (one coordinate per operand position), no per-word
  profile.  E step = column share (softmax over classes of log typicality at
  the position, i.e. column normalisation with a null-class floor ``eps0``);
  sticky switch by ``margin``; M step = online k-means (rate ``eta``); on a
  switch the old and new class are pulled together (rate ``rho``); growth when
  the best share at the position is below ``grow``; merge when two class
  vectors are within ``merge`` (L2), the smaller class remapped.
* Chooser: the true parse competes with every single-fill relocation (a word
  placed at another position). score = m*[true] + w*sum log(share+eps_s)
  + tau*Gumbel.  Credit = chosen parse is the true parse; only credited
  sentences are observed by the learner (plan: a kept use is the E step's
  observation).  ``gate='dominant'|'support'`` are the hard-gate ablations.

No oracle class label is read by the learner or the chooser; hidden classes
are used only for scoring.  Run: ``../../../.venv/bin/python sim.py``
(about 3 minutes); ``--quick`` for a 30 s smoke run.
"""
import argparse
import json
import sys
import time

import numpy as np

POSITIONS = ['det.argin1', 'det.argin2', 'adj.argin1', 'adj.argin2',
             'clause.argin1', 'clause.argin2', 'vp.argin1', 'vp.argin2',
             'bind.argin2']
PIDX = {p: i for i, p in enumerate(POSITIONS)}
P = len(POSITIONS)
ADJ2, VP2, CL1, CL2 = (PIDX['adj.argin2'], PIDX['vp.argin2'],
                       PIDX['clause.argin1'], PIDX['clause.argin2'])

LEX = {
    'det': ['a', 'the'],
    'adj': ['black', 'white', 'small', 'big'],
    'noun': ['cat', 'dog', 'bird', 'fish'],
    'name': ['tom', 'max', 'sam', 'rex'],
    'vi': ['sleep', 'purr', 'bark', 'swim'],
    'vt': ['see', 'chase', 'like', 'bite'],
    'pro': ['it', 'they'],
    'amb': ['run', 'walk'],          # noun and intransitive verb (corpus C)
}
NOVEL = {                            # stream 3 of Q4: unseen words
    'adj': ['red', 'old'], 'noun': ['cow', 'frog'], 'name': ['ann', 'bob'],
    'vi': ['jump', 'sing'], 'vt': ['hit', 'want'], 'pro': ['he'],
}
HIDDEN = {'det': 'det', 'adj': 'adj', 'noun': 'noun', 'name': 'noun',
          'vi': 'vi', 'vt': 'vt', 'pro': 'pro'}      # 6 scored classes
NP_MIX = {   # det N | det A N | name | pronoun | bare N
    'A': dict(detN=.35, detAN=.25, name=.25, pro=.15, bare=.0),
    'B': dict(detN=.25, detAN=.20, name=.10, pro=.05, bare=.40),
    'C': dict(detN=.35, detAN=.25, name=.25, pro=.15, bare=.0),
}
P_AMB = {'A': .0, 'B': .0, 'C': .1}

DEFAULT = dict(eta=.02, rho=.05, margin=.2, grow=.05, eps0=.05, merge=.35,
               w=.5, m=1., tau=.5, eps_s=.02, gate='soft', gate_floor=.02,
               order='EM')


class Generator:
    def __init__(self, corpus, seed, heads=False, lex=None):
        self.rng = np.random.default_rng(seed)
        self.mix = NP_MIX[corpus]
        self.p_amb = P_AMB[corpus]
        self.heads = heads
        self.lex = lex or {k: list(v) for k, v in LEX.items()}
        self.np_keys = ['detN', 'detAN', 'name', 'pro', 'bare']
        self.np_p = np.array([self.mix[k] for k in self.np_keys])
        self.np_p /= self.np_p.sum()
        self.hidden = {}
        for cls, words in self.lex.items():
            for w in words:
                self.hidden[w] = cls

    def pick(self, cls):
        words = self.lex[cls]
        return words[self.rng.integers(len(words))]

    def noun(self):
        if self.p_amb and self.rng.random() < self.p_amb:
            return self.pick('amb')
        return self.pick('noun')

    def vi(self):
        if self.p_amb and self.rng.random() < self.p_amb:
            return self.pick('amb')
        return self.pick('vi')

    def np_(self, slot, toks, fills):
        kind = self.np_keys[self.rng.choice(5, p=self.np_p)]
        if kind in ('detN', 'detAN'):
            toks.append(self.pick('det'))
            fills.append((len(toks) - 1, PIDX['det.argin1']))
            if kind == 'detAN':
                toks.append(self.pick('adj'))
                fills.append((len(toks) - 1, PIDX['adj.argin1']))
                toks.append(self.noun())
                i = len(toks) - 1
                fills.append((i, ADJ2))
                if self.heads:
                    fills.append((i, PIDX['det.argin2']))
            else:
                toks.append(self.noun())
                i = len(toks) - 1
                fills.append((i, PIDX['det.argin2']))
            if self.heads:
                fills.append((i, slot))
        elif kind == 'name':
            toks.append(self.pick('name'))
            fills.append((len(toks) - 1, slot))
        elif kind == 'pro':
            toks.append(self.pick('pro'))
            fills.append((len(toks) - 1, PIDX['bind.argin2']))
            if self.heads:
                fills.append((len(toks) - 1, slot))
        else:
            toks.append(self.noun())
            fills.append((len(toks) - 1, slot))

    def sentence(self):
        toks, fills = [], []
        self.np_(CL1, toks, fills)
        if self.rng.random() < .5:
            toks.append(self.vi())
            fills.append((len(toks) - 1, CL2))
        else:
            toks.append(self.pick('vt'))
            i = len(toks) - 1
            fills.append((i, PIDX['vp.argin1']))
            self.np_(VP2, toks, fills)
            if self.heads:
                fills.append((i, CL2))
        return toks, fills


class Learner:
    """Plan §6.1: pointer per word, class vectors in position space."""

    def __init__(self, cfg, max_classes=256):
        self.cfg = cfg
        self.C = np.zeros((max_classes, P))
        self.size = np.zeros(max_classes, int)
        self.remap = np.arange(max_classes)
        self.live = []                      # live class ids
        self.n_created = 0
        self.ptr = {}                       # word -> class id (raw)
        self.n_switch = self.n_grow = self.n_merge = 0
        self.switch_log = []                # (t, word)
        self.merge_log = []                 # (t, small, big)
        self.t = 0

    def resolve(self, k):
        while self.remap[k] != k:
            k = self.remap[k]
        return k

    def cls(self, w):
        k = self.ptr.get(w)
        if k is None:
            return None
        r = self.resolve(k)
        self.ptr[w] = r
        return r

    def shares(self):
        """Column share: softmax over classes of log typicality at each
        position = C[k,p] / (eps0 + sum_j C[j,p]); eps0 is a null class."""
        ids = np.asarray(self.live, int)
        Cl = self.C[ids]
        return ids, Cl / (self.cfg['eps0'] + Cl.sum(0, keepdims=True))

    def new_class(self, p):
        k = self.n_created
        self.n_created += 1
        if k >= len(self.C):
            raise RuntimeError('class table exhausted')
        self.C[k] = 0.
        self.C[k, p] = 1.
        self.size[k] = 0
        self.live.append(k)
        self.n_grow += 1
        return k

    def assign(self, w, k):
        self.ptr[w] = k
        self.size[k] += 1

    def switch(self, w, old, new):
        rho = self.cfg['rho']
        co, cn = self.C[old].copy(), self.C[new].copy()
        self.C[old] += rho * (cn - co)
        self.C[new] += rho * (co - cn)
        self.size[old] -= 1
        self.size[new] += 1
        self.ptr[w] = new
        self.n_switch += 1
        self.switch_log.append((self.t, w))

    def merge(self, a, b):
        if (self.size[a], -a) > (self.size[b], -b):
            a, b = b, a                       # a = smaller -> remapped onto b
        na, nb = max(self.size[a], 0), max(self.size[b], 0)
        tot = na + nb
        self.C[b] = ((na * self.C[a] + nb * self.C[b]) / tot if tot
                     else .5 * (self.C[a] + self.C[b]))
        self.size[b] += self.size[a]
        self.remap[a] = b
        self.live.remove(a)
        self.n_merge += 1
        self.merge_log.append((self.t, a, b))
        return b

    def check_merge(self, k):
        ids = np.asarray([i for i in self.live if i != k], int)
        if not len(ids):
            return
        d = np.linalg.norm(self.C[ids] - self.C[k], axis=1)
        j = int(np.argmin(d))
        if d[j] < self.cfg['merge']:
            self.merge(k, int(ids[j]))

    def mstep(self, k, p):
        c = self.C[k]
        c *= 1. - self.cfg['eta']
        c[p] += self.cfg['eta']

    def use(self, w, p):
        """One kept use of word w at position p (plan §6.1).

        order='EM' (the plan as written): E step, then the chosen class takes
        the M step.  order='ME': the word's current class takes the M step
        first (the use belongs to the class the word was in when the parse
        was chosen), then the E step; a switch also moves the new class."""
        cfg = self.cfg
        k = self.cls(w)
        if k is not None and cfg['order'] == 'ME':
            self.mstep(k, p)
        ids = np.asarray(self.live, int)
        typ = self.C[ids, p] if len(ids) else np.zeros(0)
        old = None
        if not len(ids) or typ.max() < cfg['grow']:
            new = self.new_class(p)              # fits no class: growth
            if k is None:
                self.assign(w, new)
            else:
                old = k
                self.switch(w, k, new)
            chosen = new
        else:
            col = typ / (cfg['eps0'] + typ.sum())   # column share at p
            bi = int(np.argmax(col))
            best = int(ids[bi])
            if k is None:
                self.assign(w, best)
                chosen = best
            elif best != k and col[bi] - float(col[ids == k][0]) > cfg['margin']:
                old = k
                self.switch(w, k, best)
                chosen = best
            else:
                chosen = k
        if not (chosen == k and cfg['order'] == 'ME'):
            self.mstep(chosen, p)
        self.check_merge(chosen)
        if old is not None and self.remap[old] == old:
            self.check_merge(old)


def choose(learner, toks, fills, rng, cfg):
    """True parse vs every single-fill relocation. Returns (credit, novel)."""
    n = len(fills)
    ids, sh = learner.shares()
    L = np.zeros((n, P))
    A = np.ones((n, P), bool)
    novel = np.zeros(n, bool)
    for f, (i, p) in enumerate(fills):
        k = learner.cls(toks[i])
        if k is None:
            continue
        row = sh[np.flatnonzero(ids == k)[0]]
        L[f] = cfg['w'] * np.log(row + cfg['eps_s'])
        typ = learner.C[k]
        novel[f] = typ[p] < cfg['grow']
        if cfg['gate'] == 'dominant':
            A[f] = False
            A[f, int(np.argmax(typ))] = True
        elif cfg['gate'] == 'support':
            A[f] = typ > cfg['gate_floor']
    ar = np.arange(n)
    ptrue = np.array([p for _, p in fills])
    Lt = L[ar, ptrue]
    bias_T = Lt.sum()
    ok = A[ar, ptrue]
    allowed_T = bool(ok.all())
    n_bad = int((~ok).sum())
    # relocation (f -> q) is allowed iff A[f, q] and every other fill is ok
    others_ok = (n_bad == 0) | ((n_bad == 1) & ~ok)
    S = bias_T - Lt[:, None] + L
    S[ar, ptrue] = -np.inf
    S[~(A & others_ok[:, None])] = -np.inf
    tau = cfg['tau']
    score_T = (cfg['m'] + bias_T + tau * rng.gumbel()) if allowed_T else -np.inf
    best_alt = (S + tau * rng.gumbel(size=S.shape)).max()
    credit = bool(np.isfinite(score_T) and score_T > best_alt)
    return credit, novel


# ---------------------------------------------------------------- scoring
def snapshot(learner, hidden):
    cls = {w: learner.cls(w) for w in list(learner.ptr)}
    return dict(cls=cls, K=len(learner.live), t=learner.t,
                C={int(k): learner.C[k].copy() for k in learner.live},
                size={int(k): int(learner.size[k]) for k in learner.live})


def majority(snap, hidden, cls_name, pool=None):
    pool = pool if pool is not None else [w for w, h in hidden.items()
                                          if h == cls_name]
    ks = [snap['cls'][w] for w in pool if w in snap['cls']]
    if not ks:
        return None
    vals, counts = np.unique(ks, return_counts=True)
    return int(vals[np.argmax(counts)])


def purity_completeness(snap, hidden):
    words = [w for w in snap['cls'] if hidden.get(w) in HIDDEN]
    if not words:
        return 0., 0., 0
    lab = np.array([HIDDEN[hidden[w]] for w in words])
    kid = np.array([snap['cls'][w] for w in words])
    pur = sum(np.bincount(np.unique(lab[kid == k], return_inverse=True)[1]).max()
              for k in np.unique(kid))
    com = sum(np.bincount(np.unique(kid[lab == h], return_inverse=True)[1]).max()
              for h in np.unique(lab))
    return pur / len(words), com / len(words), len(words)


def positions_same(learner, p, q):
    """Do positions p and q read as the same class (argmax of the column
    share)?  False until both positions have been seen."""
    if not learner.live:
        return False
    Cl = learner.C[np.asarray(learner.live)]
    if Cl[:, p].max() <= 0. or Cl[:, q].max() <= 0.:
        return False
    return int(np.argmax(Cl[:, p])) == int(np.argmax(Cl[:, q]))


TRACE = (50, 100, 200, 500, 1000, 2000, 5000)


def run(corpus, seed, cfg, N=5000, checkpoints=(200, 1000, 5000),
        heads=False, streams=False):
    rng = np.random.default_rng(10_000 + seed)
    gen = Generator(corpus, seed, heads)
    hidden = dict(gen.hidden)
    L = Learner(cfg)
    checkpoints = tuple(sorted(set(checkpoints) | {N}))
    same, Ks = [], []
    stats = dict(credit=0, novel_uses=0, novel_credit=0, amb_uses=0)
    snaps = {}
    window_start = N - 1000
    if window_start < 1:
        snaps['window'] = snapshot(L, hidden)

    def present(gen, t):
        L.t = t
        toks, fills = gen.sentence()
        credit, novel = choose(L, toks, fills, rng, cfg)
        stats['novel_uses'] += int(novel.sum())
        if credit:
            stats['credit'] += 1
            stats['novel_credit'] += int(novel.sum())
            for i, p in fills:
                if toks[i] in LEX['amb']:
                    stats['amb_uses'] += 1
                L.use(toks[i], p)

    for t in range(1, N + 1):
        present(gen, t)
        Ks.append(len(L.live))
        same.append(positions_same(L, ADJ2, VP2))
        if t == window_start:
            snaps['window'] = snapshot(L, hidden)
        if t in checkpoints:
            snaps[t] = snapshot(L, hidden)
    out = dict(corpus=corpus, seed=seed, heads=heads, N=N)
    for t in checkpoints:
        s = snaps[t]
        pur, com, nw = purity_completeness(s, hidden)
        noun_k, name_k = majority(s, hidden, 'noun'), majority(s, hidden, 'name')
        vi_k, vt_k = majority(s, hidden, 'vi'), majority(s, hidden, 'vt')
        pro_k = majority(s, hidden, 'pro')
        out[f'K@{t}'] = s['K']
        out[f'purity@{t}'] = pur
        out[f'completeness@{t}'] = com
        out[f'names_separate@{t}'] = int(noun_k is not None and name_k is not None
                                          and noun_k != name_k)
        out[f'vivt_merged@{t}'] = int(vi_k is not None and vi_k == vt_k)
        out[f'pro_separate@{t}'] = int(pro_k is not None and pro_k != noun_k
                                        and pro_k != name_k)
        out[f'noun_verb_separate@{t}'] = int(noun_k is not None and noun_k != vi_k)
    # Q2: collapse of adj.argin2 and vp.argin2
    same = np.array(same)
    last_diff = int(np.flatnonzero(~same).max()) + 1 if (~same).any() else 0
    out['collapsed'] = int(bool(same[-1]))
    out['collapse_at'] = (last_diff + 1) if same[-1] else None
    out['K_max'] = int(max(Ks))
    out['K_trace'] = {i: int(Ks[i - 1]) for i in TRACE if i <= N}
    # Q3: ambiguous words
    s = snaps[N]
    noun_k, vi_k = majority(s, hidden, 'noun'), majority(s, hidden, 'vi')
    amb = [w for w in LEX['amb'] if w in s['cls']]
    out['amb_final'] = {w: ('noun' if s['cls'][w] == noun_k else
                            'verb' if s['cls'][w] == vi_k else 'other')
                        for w in amb}
    out['amb_flips_total'] = sum(1 for t_, w in L.switch_log if w in LEX['amb'])
    out['amb_uses'] = stats['amb_uses']
    out['amb_flip_rate'] = out['amb_flips_total'] / max(stats['amb_uses'], 1)
    if noun_k is not None and vi_k is not None and noun_k != vi_k:
        out['noun_verb_dist'] = float(np.linalg.norm(s['C'][noun_k] - s['C'][vi_k]))
    else:
        out['noun_verb_dist'] = 0.
    # Q4: stability in the last 1000 presentations
    sw = snaps['window']
    switched = {w for t_, w in L.switch_log if t_ > window_start}
    words = [w for w in s['cls'] if w in sw['cls']]
    changed = [w for w in words if s['cls'][w] != sw['cls'][w] or w in switched]
    out['unstable_frac'] = len(changed) / max(len(words), 1)
    out['flips_last1000'] = sum(1 for t_, w in L.switch_log if t_ > window_start)
    out['n_grow'] = L.n_grow
    out['n_switch'] = L.n_switch
    out['n_merge'] = L.n_merge
    out['credit_rate'] = stats['credit'] / N
    out['novel_credit_rate'] = stats['novel_credit'] / max(stats['novel_uses'], 1)
    out['novel_uses'] = stats['novel_uses']
    if streams:
        # stream 2: a fresh document, same vocabulary
        before = snapshot(L, hidden)
        g2 = Generator(corpus, 500 + seed, heads)
        grow0 = L.n_grow
        for t in range(N + 1, N + 1001):
            present(g2, t)
        after = snapshot(L, hidden)
        out['s2_new_classes'] = L.n_grow - grow0
        out['s2_K'] = after['K']
        out['s2_changed_frac'] = (sum(1 for w in before['cls']
                                      if after['cls'].get(w) != before['cls'][w])
                                  / max(len(before['cls']), 1))
        # stream 3: a fresh document with novel words of the known classes
        lex = {k: list(v) for k, v in LEX.items()}
        for k, ws in NOVEL.items():
            lex[k] = lex[k] + ws
        g3 = Generator(corpus, 900 + seed, heads, lex)
        grow0 = L.n_grow
        for t in range(N + 1001, N + 2001):
            present(g3, t)
        s3 = snapshot(L, hidden)
        reused = total = 0
        for k, ws in NOVEL.items():
            old_major = majority(s3, hidden, k)      # class of the OLD words
            for w in ws:
                if w in s3['cls']:
                    total += 1
                    reused += int(s3['cls'][w] == old_major)
        out['s3_new_classes'] = L.n_grow - grow0
        out['s3_K'] = s3['K']
        out['s3_novel_reused'] = reused / max(total, 1)
        out['s3_novel_seen'] = total
    return out


# ---------------------------------------------------------------- driver
def ms(xs, fmt='{:.2f}'):
    xs = np.array([x for x in xs if x is not None], float)
    if not len(xs):
        return 'n/a'
    if len(xs) == 1:
        return fmt.format(xs[0])
    return fmt.format(xs.mean()) + ' ± ' + fmt.format(xs.std())


def count(xs):
    return f'{int(sum(xs))}/{len(xs)}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--seeds', type=int, default=5)
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    N = 1000 if args.quick else 5000
    cps = (200, 1000) if args.quick else (200, 1000, 5000)
    t0 = time.time()
    results = {}
    lines = []

    def table(title, header, rows):
        lines.append(f'\n### {title}\n')
        lines.append('| ' + ' | '.join(header) + ' |')
        lines.append('|' + '---|' * len(header))
        for r in rows:
            lines.append('| ' + ' | '.join(str(x) for x in r) + ' |')

    def runs(corpus, cfg, heads=False, streams=False, n=N):
        return [run(corpus, s, cfg, n, cps, heads, streams) for s in seeds]

    # ---- main runs (default = ME order) and the plan's EM order
    main_runs = {}
    for corpus, heads in (('A', False), ('B', False), ('C', False), ('A', True)):
        key = corpus + ('+heads' if heads else '')
        main_runs[key] = runs(corpus, DEFAULT, heads, streams=True)
    em_runs = {c: runs(c, dict(DEFAULT, order='ME')) for c in ('A', 'B', 'C')}
    em_runs['A+heads'] = runs('A', dict(DEFAULT, order='ME'), heads=True)
    print(f'main done {time.time() - t0:.0f}s', file=sys.stderr)
    results['main'] = main_runs
    results['me'] = em_runs

    # Q1
    rows = []
    for label, store in (('EM (plan)', main_runs), ('ME (old class takes M)', em_runs)):
        for key, rs in store.items():
            for t in cps:
                rows.append([label, key, t, ms([r[f'K@{t}'] for r in rs], '{:.1f}'),
                             ms([r[f'purity@{t}'] for r in rs]),
                             ms([r[f'completeness@{t}'] for r in rs]),
                             count([r[f'names_separate@{t}'] for r in rs]),
                             count([r[f'pro_separate@{t}'] for r in rs]),
                             count([r[f'vivt_merged@{t}'] for r in rs])])
    table('Q1 class count, purity, completeness (6 hidden classes; names scored as noun)',
          ['order', 'corpus', 'N', 'classes', 'purity', 'completeness', 'names separate',
           'pronoun separate', 'vi/vt merged'], rows)
    rows = []
    for label, store in (('EM', main_runs), ('ME', em_runs)):
        for key, rs in store.items():
            rows.append([label, key] + [ms([r['K_trace'][i] for r in rs], '{:.1f}')
                                        for i in TRACE if i <= N]
                        + [ms([r['n_grow'] for r in rs], '{:.0f}'),
                           ms([r['n_merge'] for r in rs], '{:.0f}')])
    table('Q1b class count over presentations (mean ± sd over seeds)',
          ['order', 'corpus'] + [f'K@{i}' for i in TRACE if i <= N] + ['growths', 'merges'], rows)

    # Q2
    rows = []
    for label, key, rs in (('EM', 'A', main_runs['A']), ('EM', 'B', main_runs['B']),
                           ('EM', 'C', main_runs['C']), ('EM', 'A+heads', main_runs['A+heads']),
                           ('ME', 'A', em_runs['A']), ('ME', 'B', em_runs['B'])):
        rows.append([label, key, count([r['collapsed'] for r in rs]),
                     ' '.join(str(r['collapse_at']) for r in rs),
                     count([r[f'names_separate@{N}'] == 0 for r in rs]),
                     ms([r['n_merge'] for r in rs], '{:.1f}'),
                     ms([r[f'K@{N}'] for r in rs], '{:.1f}')])
    table('Q2 collapse: adj.argin2 and vp.argin2 read as one class',
          ['order', 'corpus', 'collapsed (seeds)', 'collapse at (per seed)',
           'noun+name one class', 'merges', 'classes'], rows)

    # Q3
    rows = []
    for label, key, rs in (('EM', 'C', main_runs['C']), ('EM', 'A', main_runs['A']),
                           ('ME', 'C', em_runs['C'])):
        finals = [v for r in rs for v in r['amb_final'].values()]
        fc = {k: finals.count(k) for k in ('noun', 'verb', 'other')}
        rows.append([label, key, count([r[f'noun_verb_separate@{N}'] for r in rs]),
                     ms([r['noun_verb_dist'] for r in rs]),
                     ms([r['amb_uses'] for r in rs], '{:.0f}'),
                     ms([r['amb_flip_rate'] for r in rs]),
                     f"noun {fc['noun']} / verb {fc['verb']} / other {fc['other']}",
                     ms([r['novel_credit_rate'] for r in rs])])
    table('Q3 ambiguity (corpus C: run/walk are noun and verb; 10% of noun and of vi picks)',
          ['order', 'corpus', 'noun/verb separate', 'noun-verb distance', 'kept uses of run+walk',
           'flips per kept use', 'final class (10 word-seeds)', 'credit on novel uses'], rows)

    # Q4
    rows = []
    for key, rs in main_runs.items():
        rows.append([key, ms([r['unstable_frac'] for r in rs]),
                     ms([r['flips_last1000'] for r in rs], '{:.1f}'),
                     ms([r['s2_new_classes'] for r in rs], '{:.1f}'),
                     ms([r['s2_changed_frac'] for r in rs]),
                     ms([r['s3_new_classes'] for r in rs], '{:.1f}'),
                     ms([r['s3_novel_reused'] for r in rs]),
                     ms([r['s3_K'] for r in rs], '{:.1f}')])
    table('Q4 stability (presentations 4000-5000) and reuse across document streams (EM)',
          ['corpus', 'words with id change', 'switches', 'stream 2: new classes',
           'stream 2: ids changed', 'stream 3: new classes', 'stream 3: novel words reused',
           'final classes'], rows)

    # Q5 soft vs hard gate, and the bias weight
    q5 = {}
    rows = []
    for corpus in ('A', 'C'):
        for label, over in (('soft w=0 (no bias)', dict(w=0.)), ('soft w=.25', dict(w=.25)),
                            ('soft w=.5 (default)', {}), ('soft w=1', dict(w=1.)),
                            ('hard: support', dict(gate='support')),
                            ('hard: dominant', dict(gate='dominant'))):
            cfg = dict(DEFAULT, **over)
            rs = main_runs[corpus] if not over else runs(corpus, cfg)
            q5[f'{corpus}/{label}'] = rs
            rows.append([corpus, label, ms([r[f'K@{N}'] for r in rs], '{:.1f}'),
                         ms([r[f'purity@{N}'] for r in rs]),
                         ms([r[f'completeness@{N}'] for r in rs]),
                         ms([r['n_grow'] for r in rs], '{:.0f}'),
                         ms([r['n_switch'] for r in rs], '{:.0f}'),
                         ms([r['n_merge'] for r in rs], '{:.0f}'),
                         ms([r['credit_rate'] for r in rs]),
                         ms([r['novel_credit_rate'] for r in rs]),
                         count([r[f'noun_verb_separate@{N}'] for r in rs])])
    print(f'Q5 done {time.time() - t0:.0f}s', file=sys.stderr)
    results['q5'] = q5
    table('Q5 soft participation vs hard gate (N=5000)',
          ['corpus', 'chooser', 'classes', 'purity', 'completeness', 'growths',
           'switches', 'merges', 'credit rate', 'credit on novel uses', 'noun/verb separate'], rows)

    # Q6 sticky margin
    q6 = {}
    rows = []
    for corpus in ('A', 'B', 'C'):
        for margin in (0., DEFAULT['margin'], .4):
            cfg = dict(DEFAULT, margin=margin)
            rs = main_runs[corpus] if margin == DEFAULT['margin'] else runs(corpus, cfg)
            q6[f'{corpus}/{margin}'] = rs
            rows.append([corpus, margin, ms([r[f'K@{N}'] for r in rs], '{:.1f}'),
                         ms([r[f'purity@{N}'] for r in rs]),
                         ms([r[f'completeness@{N}'] for r in rs]),
                         ms([r['n_switch'] for r in rs], '{:.0f}'),
                         ms([r['flips_last1000'] for r in rs], '{:.1f}'),
                         ms([r['unstable_frac'] for r in rs]),
                         count([r['collapsed'] for r in rs]),
                         count([r[f'noun_verb_separate@{N}'] for r in rs])])
    print(f'Q6 done {time.time() - t0:.0f}s', file=sys.stderr)
    results['q6'] = q6
    table('Q6 sticky margin (N=5000)',
          ['corpus', 'margin', 'classes', 'purity', 'completeness', 'switches',
           'switches last 1000', 'words with id change', 'B-collapse', 'noun/verb separate'], rows)

    # parameter sweep: order x pull rate x merge distance
    sweep = {}
    rows = []
    Ns = 1000 if args.quick else 5000
    for order in ('EM', 'ME'):
        for rho in (0., .02, .03, .05, .08, .1):
            row = [order, rho]
            for merge in (.2, .35, .5):
                cfg = dict(DEFAULT, rho=rho, merge=merge, order=order)
                ra = runs('A', cfg, n=Ns)
                rb = runs('B', cfg, n=Ns)
                rc = runs('C', cfg, n=Ns)
                sweep[f'{order}/{rho}/{merge}'] = dict(A=ra, B=rb, C=rc)
                row.append(f"{np.mean([r[f'K@{Ns}'] for r in ra]):.1f} / "
                           f"{count([r['collapsed'] for r in rb])} / "
                           f"{count([r[f'noun_verb_separate@{Ns}'] for r in rc])}")
            rows.append(row)
            print(f'sweep {order} rho={rho} done {time.time() - t0:.0f}s', file=sys.stderr)
    results['sweep'] = sweep
    table(f'Parameter sweep (N={Ns}): cell = A classes (want 7) / B collapsed (want 5/5) / C noun-verb separate (want 5/5)',
          ['order', 'pull rate rho', 'merge .2', 'merge .35', 'merge .5'], rows)

    text = '\n'.join(lines)
    print(text)
    with open('results.md', 'w') as f:
        f.write(f'# pos-participation toy results (N={N}, seeds {seeds}, {time.time() - t0:.0f}s)\n'
                f'\nDEFAULT = {json.dumps(DEFAULT)}\n' + text + '\n')
    with open('results.json', 'w') as f:
        json.dump(results, f, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print(f'total {time.time() - t0:.0f}s', file=sys.stderr)


if __name__ == '__main__':
    main()
