#!/usr/bin/env python3
"""Identity by unmixing: does ICA-shaped teaching data teach identity?

A numpy toy for item 6. One online learner mirrors
bin/IndependentComponents.SparseDictionary (encode / observe / loss); four
questions are run on it. See README.md for the setup and the results.

    ../../../.venv/bin/python sim.py            # all four questions, 5 seeds
    ../../../.venv/bin/python sim.py --q 1 3    # a subset

Writes results.json beside this file and prints the tables as markdown.
"""
import argparse
import json
import os
import time
import warnings

import numpy as np

try:
    from sklearn.decomposition import FastICA, MiniBatchDictionaryLearning
    from sklearn.exceptions import ConvergenceWarning
    HAVE_SKLEARN = True
except ImportError:          # the batch references are optional
    HAVE_SKLEARN = False

D = 64            # event-coordinate dimension
NOISE = 0.01      # per-coordinate noise (norm ~0.08, below the mint threshold)
LAM = 0.05        # independencePriorScale (code default)
TAU = 0.2         # independenceMintThreshold (code default)
REC = 4           # recurrence (code default)
LR = 0.5          # SGD step on the population loss (toy choice, fixed for all runs)
BATCH = 64        # population sample per gradient step
CAP = 128         # column inventory; a full inventory drops admissions, as _allocate does


def unit(x, axis=-1):
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12)


def make_codes(rng, n, kind='dense', nnz=6):
    if kind == 'dense':
        return unit(rng.standard_normal((n, D)))
    out = np.zeros((n, D))
    for i in range(n):
        idx = rng.choice(D, nnz, replace=False)
        out[i, idx] = rng.uniform(.5, 1., nnz) * rng.choice([-1., 1.], nnz)
    return unit(out)


def ms(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std())


def fmt(pair, digits=2):
    mean, sd = pair
    return f'{mean:.{digits}f} ± {sd:.{digits}f}'


# ---------------------------------------------------------------------------
# The learner: a numpy mirror of SparseDictionary
# ---------------------------------------------------------------------------

class OnlineIC:
    """Unit columns, relevance scales, top-k least-squares codes, softshrink.

    encode   = SparseDictionary.encode (top-k by |projection * relevance|,
               normal equations on the selected columns, softshrink(LAM) * relevance)
    observe  = SparseDictionary.observe (residual norm > TAU starts or extends a
               pending prototype matched at |cos| >= 1 - TAU; REC distinct
               witnesses mint a column). mint='value' mints the normalized
               observation, as the code does; mint='residual' mints the
               normalized residual (a variant, not the code).
    step     = one SGD step on SparseDictionary.loss over a sample of the
               population (reconstruction + LAM|codes| + LAM*coherence +
               LAM*mean relevance). Codes are held fixed in the column
               gradient; the code backpropagates through the solve, and at the
               least-squares optimum the two differ only by the shrinkage term.
    """

    def __init__(self, k, mint='value', lr=LR, cap=CAP, select='topk'):
        self.k, self.mint, self.lr, self.cap, self.select = int(k), mint, lr, cap, select
        self.C = np.zeros((0, D))
        self.s = np.zeros(0)
        self.pending_dirs = np.zeros((0, D))
        self.pending_wit = []
        self.drops = 0
        self.mints = 0

    def encode(self, X, k=None):
        X = np.atleast_2d(X)
        n = len(self.C)
        if n == 0:
            return np.zeros((len(X), 0)), np.zeros_like(X), X.copy()
        k = min(self.k if k is None else int(k), n)
        scales = np.maximum(self.s, 0.)
        P = X @ self.C.T
        if self.select == 'topk':                  # the code: one-shot top-k by |projection|
            sel = np.argsort(-np.abs(P * scales), axis=1, kind='stable')[:, :k]
        else:                                      # diagnostic: greedy pursuit (OMP order)
            sel = self._greedy(X, P, scales, k)
        A = self.C[sel]
        G = A @ A.transpose(0, 2, 1) + 1e-6 * np.eye(k)
        b = np.take_along_axis(P, sel, 1)
        coord = np.linalg.solve(G, b[..., None])[..., 0]
        sparse = np.sign(coord) * np.maximum(np.abs(coord) - LAM, 0.) * scales[sel]
        codes = np.zeros((len(X), n))
        np.put_along_axis(codes, sel, sparse, 1)
        rec = codes @ self.C
        return codes, rec, X - rec

    def _greedy(self, X, P, scales, k):
        N, rows = len(X), np.arange(len(X))
        sel = np.zeros((N, 0), dtype=int)
        R = X.copy()
        for _ in range(k):
            score = np.abs((R @ self.C.T) * scales)
            if sel.shape[1]:
                score[rows[:, None], sel] = -1.
            sel = np.hstack([sel, score.argmax(1)[:, None]])
            A = self.C[sel]
            G = A @ A.transpose(0, 2, 1) + 1e-6 * np.eye(sel.shape[1])
            coord = np.linalg.solve(G, np.take_along_axis(P, sel, 1)[..., None])[..., 0]
            R = X - np.einsum('nk,nkd->nd', coord, A)
        return sel

    def observe(self, x, witness):
        _, _, r = self.encode(x)
        r = r[0]
        if np.linalg.norm(r) <= TAU or not x.any():
            return
        d = unit(x if self.mint == 'value' else r)
        hits = np.nonzero(np.abs(self.pending_dirs @ d) >= 1 - TAU)[0] if len(self.pending_wit) else []
        if len(hits):
            j = int(hits[0])
        else:
            self.pending_dirs = np.vstack([self.pending_dirs, d])
            self.pending_wit.append([])
            j = len(self.pending_wit) - 1
        wit = self.pending_wit[j]
        if witness in wit:
            return
        if len(wit) < REC:
            wit.append(witness)
        if len(wit) < REC:
            return
        if len(self.C) >= self.cap:       # _allocate returned None: keep pending
            self.drops += 1
            return
        self.C = np.vstack([self.C, self.pending_dirs[j]])
        self.s = np.append(self.s, 1.)
        self.mints += 1
        self.pending_dirs = np.delete(self.pending_dirs, j, 0)
        del self.pending_wit[j]

    def step(self, X):
        n = len(self.C)
        if n == 0:
            return
        codes, _, R = self.encode(X)
        B = len(X)
        g = -(codes.T @ R) / B
        if n > 1:
            G = self.C @ self.C.T
            np.fill_diagonal(G, 0.)
            g += LAM * 4. * (G @ self.C) / (n * (n - 1))
        g -= (g * self.C).sum(1, keepdims=True) * self.C       # through the normalization
        live = self.s > 0
        u = np.zeros_like(codes)
        u[:, live] = codes[:, live] / self.s[live]
        gs = (-((R @ self.C.T) * u).sum(0) + LAM * np.abs(u).sum(0)) / B + LAM / n
        self.C = unit(self.C - self.lr * g)
        self.s = np.where(live, np.maximum(self.s - self.lr * gs, 0.), 0.)   # clamp_min(0)

    def train(self, rows, rng, checkpoints=(), on_checkpoint=None):
        rows = np.asarray(rows)
        for t, x in enumerate(rows):
            self.observe(x, t)
            # population, not batch: sample the retained chain seen so far
            self.step(rows[rng.integers(0, t + 1, min(BATCH, t + 1))])
            if on_checkpoint is not None and t + 1 in checkpoints:
                on_checkpoint(t + 1, self)
        return self


ONLINE = (('code: mint=value, top-k, k=3', 'value', 3, 'topk'),
          ('variant: mint=residual, top-k, k=3', 'residual', 3, 'topk'),
          ('variant: mint=residual, greedy, k=3', 'residual', 3, 'greedy'),
          ('variant: mint=residual, top-k, k=4', 'residual', 4, 'topk'))


def recovery(columns, atoms):
    if len(columns) == 0:
        return np.zeros(len(atoms))
    return np.abs(unit(atoms) @ unit(columns).T).max(1)


def clean_fraction(columns, atoms):
    """Share of learned columns that are one atom (|cos| >= .9), not a conjunction."""
    if len(columns) == 0:
        return 0.
    return float((np.abs(unit(columns) @ unit(atoms).T).max(1) >= .9).mean())


def identification(ic, X, present, atoms):
    """Share of present atoms carried, in the row's own code, by a column that is that atom.

    present[i] lists the atom indices mixed into held-out row X[i]. A row
    explained by a memorized frame column scores 0 for its atoms.
    """
    if len(ic.C) == 0:
        return 0.
    codes, _, _ = ic.encode(X)
    match = np.abs(unit(atoms) @ unit(ic.C).T) >= .9            # atoms x columns
    hits = total = 0
    for i, idx in enumerate(present):
        active = codes[i] != 0
        for a in idx:
            hits += bool((match[a] & active).any())
            total += 1
    return hits / max(total, 1)


# ---------------------------------------------------------------------------
# Q1 factorial vs confounded data
# ---------------------------------------------------------------------------

def q1_corpus(rng, kind, corpus, n_rows):
    O, P, V = make_codes(rng, 8, kind), make_codes(rng, 8, kind), make_codes(rng, 6, kind)
    rows, present = [], []
    for _ in range(n_rows):
        o = int(rng.integers(8))
        if corpus == 'A':                          # factorial
            p = int(rng.integers(8))
        elif corpus == 'B1':                       # o0 <-> p0 both ways
            p = 0 if o == 0 else int(rng.integers(1, 8))
        else:                                      # B2: p0 only with o0; o0 also elsewhere
            p = (0 if rng.random() < .5 else int(rng.integers(1, 8))) if o == 0 else int(rng.integers(1, 8))
        v = int(rng.integers(6))
        m = rng.uniform(.7, 1.3, 3)
        rows.append(m[0] * O[o] + m[1] * P[p] + m[2] * V[v] + NOISE * rng.standard_normal(D))
        present.append((o, 8 + p, 16 + v))
    return np.asarray(rows), present, O, P, V


def batch_references(X, n_components, seed):
    out = {}
    if not HAVE_SKLEARN:
        return out
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ica = FastICA(n_components=n_components, whiten='unit-variance', max_iter=1000,
                      random_state=seed).fit(X)
        out['ref: FastICA (22 given)'] = ica.mixing_.T
        # alpha .3 and OMP with the declared ceiling of 3; sklearn's default
        # (LARS, alpha 1) gave 0.05 one-atom columns on the same data.
        dl = MiniBatchDictionaryLearning(n_components=n_components, alpha=.3, batch_size=64,
                                         max_iter=200, transform_algorithm='omp',
                                         transform_n_nonzero_coefs=3, random_state=seed).fit(X)
        out['ref: DictLearning (22 given)'] = dl.components_
    return out


def frozen(columns, k):
    """Read a reference dictionary through the same top-k encoder."""
    ic = OnlineIC(k)
    ic.C, ic.s = unit(np.asarray(columns)), np.ones(len(columns))
    return ic


def run_q1(seeds, n_rows=3000, n_test=500):
    results = []
    for kind in ('dense', 'sparse'):
        for corpus in ('A', 'B1', 'B2'):
            per = {}
            for seed in seeds:
                rng = np.random.default_rng(1000 + seed)
                X, present, O, P, V = q1_corpus(rng, kind, corpus, n_rows + n_test)
                X, Xt, present_t = X[:n_rows], X[n_rows:], present[n_rows:]
                atoms = np.vstack([O, P, V])
                merged = unit(O[0] + P[0])[None]
                learned = {}
                for name, mint, k, select in ONLINE:
                    learned[name] = OnlineIC(k, mint=mint, cap=1000, select=select).train(
                        X, np.random.default_rng(seed))
                if kind == 'dense':
                    for name, cols in batch_references(X, 22, seed).items():
                        learned[name] = frozen(cols, 3)
                for name, ic in learned.items():
                    rec = recovery(ic.C, atoms)
                    rm = recovery(ic.C, merged)[0]
                    e = per.setdefault(name, {key: [] for key in
                        ('mean', 'cols', 'clean', 'ident', 'o0', 'p0', 'merged', 'merge_rate')})
                    e['mean'].append(rec.mean())
                    e['cols'].append(len(ic.C))
                    e['clean'].append(clean_fraction(ic.C, atoms))
                    e['ident'].append(identification(ic, Xt, present_t, atoms))
                    e['o0'].append(rec[0])
                    e['p0'].append(rec[8])
                    e['merged'].append(rm)
                    # one column for the pair, and neither member on its own
                    e['merge_rate'].append(float(rm >= .9 and max(rec[0], rec[8]) < .9))
            for name, e in per.items():
                results.append(dict(codes=kind, corpus=corpus, learner=name, seeds=len(seeds),
                                    **{key: ms(val) for key, val in e.items()}))
    return results


def run_q1_ica_conditions(seeds, n_rows=3000):
    """FastICA when each role holds exactly one source vs sources present independently."""
    out = {'one per role (Q1 corpus A)': [], 'independent presence': []}
    if not HAVE_SKLEARN:
        return {}
    for seed in seeds:
        rng = np.random.default_rng(1000 + seed)
        X, _, O, P, V = q1_corpus(rng, 'dense', 'A', n_rows)
        atoms = np.vstack([O, P, V])
        S = (rng.random((n_rows, 22)) < 3 / 22) * rng.uniform(.7, 1.3, (n_rows, 22))
        Xi = S @ atoms + NOISE * rng.standard_normal((n_rows, D))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for key, data in (('one per role (Q1 corpus A)', X), ('independent presence', Xi)):
                mix = FastICA(n_components=22, whiten='unit-variance', max_iter=1000,
                              random_state=seed).fit(data).mixing_.T
                out[key].append(recovery(mix, atoms).mean())
    return {key: ms(val) for key, val in out.items()}


def table_q1(results):
    lines = ['| codes | corpus | learner | recovery | columns | one-atom columns | identification | o0 / p0 | o0+p0 column | merged |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for r in results:
        lines.append(f"| {r['codes']} | {r['corpus']} | {r['learner']} | {fmt(r['mean'])} | {r['cols'][0]:.0f} | "
                     f"{fmt(r['clean'])} | {fmt(r['ident'])} | {r['o0'][0]:.2f} / {r['p0'][0]:.2f} | {fmt(r['merged'])} | "
                     f"{r['merge_rate'][0] * r['seeds']:.0f}/{r['seeds']} |")
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Q2 support curriculum
# ---------------------------------------------------------------------------

def q2_rows(rng, atoms, sizes):
    rows = []
    for s in sizes:
        idx = rng.choice(len(atoms), int(s), replace=False)
        rows.append(rng.uniform(.7, 1.3, int(s)) @ atoms[idx] + NOISE * rng.standard_normal(D))
    return np.asarray(rows)


def q2_sizes(rng, curriculum, n_rows):
    third = n_rows // 3
    if curriculum == 'staged 1->2->3':
        return np.repeat([1, 2, 3], [third, third, n_rows - 2 * third])
    if curriculum == 'mixed 1-3':
        return rng.integers(1, 4, n_rows)
    return np.full(n_rows, 3)                     # 'three only'


def run_q2(seeds, n_rows=3000, checkpoints=(250, 500, 1000, 2000, 3000)):
    results = []
    for curriculum in ('staged 1->2->3', 'mixed 1-3', 'three only'):
        for name, mint, k, select in ONLINE:
            curve = {c: [] for c in checkpoints}
            cols = {c: [] for c in checkpoints}
            ident = {c: [] for c in checkpoints}
            for seed in seeds:
                rng = np.random.default_rng(2000 + seed)
                atoms = make_codes(rng, 22)
                X = q2_rows(rng, atoms, q2_sizes(rng, curriculum, n_rows))
                # held-out test rows: three sources each (the hardest support)
                present_t = [rng.choice(22, 3, replace=False) for _ in range(500)]
                Xt = np.asarray([rng.uniform(.7, 1.3, 3) @ atoms[i] + NOISE * rng.standard_normal(D)
                                 for i in present_t])

                def record(t, ic):
                    curve[t].append(recovery(ic.C, atoms).mean())
                    cols[t].append(len(ic.C))
                    ident[t].append(identification(ic, Xt, present_t, atoms))
                OnlineIC(k, mint=mint, cap=1000, select=select).train(X, np.random.default_rng(seed),
                                                       checkpoints=set(checkpoints), on_checkpoint=record)
            results.append(dict(curriculum=curriculum, learner=name,
                                recovery={c: ms(v) for c, v in curve.items()},
                                columns={c: ms(v) for c, v in cols.items()},
                                identification={c: ms(v) for c, v in ident.items()}))
    return results


def table_q2(results):
    cps = list(results[0]['recovery'])
    head = '| curriculum | learner | ' + ' | '.join(f'n={c}' for c in cps) + ' |'
    lines = [head, '|---|---|' + '---|' * len(cps)]
    for r in results:
        cells = [f"{r['identification'][c][0]:.2f}±{r['identification'][c][1]:.2f} / "
                 f"{r['recovery'][c][0]:.2f} ({r['columns'][c][0]:.0f})" for c in cps]
        lines.append(f"| {r['curriculum']} | {r['learner']} | " + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def residual_floor(seeds, supports=(1, 2, 3, 4, 5), n_rows=2000):
    """Residual norm under the TRUE atoms, by support: the floor the mint gate must clear."""
    out = {s: ([], []) for s in supports}
    for seed in seeds:
        rng = np.random.default_rng(2500 + seed)
        atoms = make_codes(rng, 22)
        for s in supports:
            X = np.asarray([rng.uniform(.7, 1.3, s) @ atoms[rng.choice(22, s, replace=False)]
                            + NOISE * rng.standard_normal(D) for _ in range(n_rows)])
            r = np.linalg.norm(frozen(atoms, s).encode(X)[2], axis=1)
            out[s][0].append(r.mean())
            out[s][1].append((r > TAU).mean())
    return {s: dict(mean=ms(m), above_tau=ms(a)) for s, (m, a) in out.items()}


def run_q2_gap(seeds, n_rows=3000):
    """Staged corpus whose middle stage drops the verb class (Q3's first corpus)."""
    results = []
    learners = [entry for entry in ONLINE if entry[0].startswith('code') or 'greedy' in entry[0] or 'k=4' in entry[0]]
    for (label, verbs_in_stage2), (name, mint, k, select) in [
            (corpus, learner) for corpus in (('verbs absent from stage 2', False), ('verbs kept in stage 2', True))
            for learner in learners]:
        cols, clean, ident = [], [], []
        for seed in seeds:
            rng = np.random.default_rng(3500 + seed)
            atoms = make_codes(rng, 26)            # 8 kinds, 8 + 4 predicates, 6 verbs
            rows = []
            for t in range(n_rows):
                if t < n_rows // 3:
                    idx = [int(rng.integers(26))]
                elif t < 2 * n_rows // 3:
                    kind = int(rng.integers(8))
                    if verbs_in_stage2 and rng.random() < .5:
                        idx = [kind, 20 + int(rng.integers(6))]
                    else:
                        idx = [kind, 8 + kind if rng.random() < CHAR_RATE else 16 + int(rng.integers(4))]
                else:                              # "the X V the Y ."
                    a, b = rng.choice(8, 2, replace=False)
                    idx = [int(a), 20 + int(rng.integers(6)), int(b)]
                rows.append(rng.uniform(.7, 1.3, len(idx)) @ atoms[idx] + NOISE * rng.standard_normal(D))
            present_t = [[int(a), 20 + int(rng.integers(6)), int(b)] for a, b in
                         (rng.choice(8, 2, replace=False) for _ in range(500))]
            Xt = np.asarray([rng.uniform(.7, 1.3, 3) @ atoms[i] + NOISE * rng.standard_normal(D) for i in present_t])
            ic = OnlineIC(k, mint=mint, cap=1000, select=select).train(np.asarray(rows), np.random.default_rng(seed))
            cols.append(len(ic.C))
            clean.append(clean_fraction(ic.C, atoms))
            ident.append(identification(ic, Xt, present_t, atoms))
        results.append(dict(corpus=label, learner=name, columns=ms(cols), clean=ms(clean), identification=ms(ident),
                            per_seed_columns=cols))
    return results


def table_q2_extra(floor, gap):
    lines = ['| support (sources per row) | ' + ' | '.join(str(s) for s in floor) + ' |',
             '|---|' + '---|' * len(floor),
             '| residual under the true atoms | ' + ' | '.join(f"{v['mean'][0]:.3f}" for v in floor.values()) + ' |',
             '| rows above tau = .2 | ' + ' | '.join(f"{v['above_tau'][0]:.3f}" for v in floor.values()) + ' |',
             '',
             '| staged corpus, then "the X V the Y ." | learner | columns (per seed) | one-atom columns | identification |',
             '|---|---|---|---|---|']
    for r in gap:
        lines.append(f"| {r['corpus']} | {r['learner']} | {r['columns'][0]:.0f} ({', '.join(str(c) for c in r['per_seed_columns'])}) | "
                     f"{fmt(r['clean'])} | {fmt(r['identification'])} |")
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Q3 pronoun binding credited by prediction
# ---------------------------------------------------------------------------

K3 = 8                      # kinds; kind k's characteristic predicate is CHAR[k] (purr -> cat)
CHAR_RATE = .7              # share of "X P" sentences whose P is the kind's characteristic one
# (subject, recent) of the (older, more recent) STM candidate, per template
TEMPLATES = {'T1 "the X V the Y ."': ((1, 0), (0, 1)),
             'T2 "the X V . the Y V ."': ((1, 0), (1, 1)),
             'T3 "with the X , the Y V ."': ((0, 0), (1, 1))}


class Q3World:
    """Kinds, characteristic and generic predicates, and a dictionary + predictor
    learned by the code's mint rule from a staged corpus (Q2's curriculum)."""

    def __init__(self, rng, n_rows=3000):
        self.O, self.CHAR = make_codes(rng, K3), make_codes(rng, K3)
        self.GEN, self.V = make_codes(rng, 4), make_codes(rng, 6)
        atoms = np.vstack([self.O, self.CHAR, self.GEN, self.V])
        # Staged as in Q2: single words, then "the X P ." / "the X V ." (every
        # column stays in use in every stage). Three-source "the X V the Y ."
        # frames are left out: the Q2 notes show a stage that drops a column
        # class lets its relevance drift and then mints frames; the binding
        # test only needs the predicate statistics.
        rows, half = [], n_rows // 3
        for t in range(n_rows):
            if t < half:                                          # single words
                rows.append(rng.uniform(.7, 1.3) * atoms[rng.integers(len(atoms))])
            else:
                k = rng.integers(K3)
                if rng.random() < .5:                             # "the X P ."
                    p = self.CHAR[k] if rng.random() < CHAR_RATE else self.GEN[rng.integers(4)]
                else:                                             # "the X V ."
                    p = self.V[rng.integers(6)]
                rows.append(rng.uniform(.7, 1.3, 2) @ np.vstack([self.O[k], p]))
            rows[-1] = rows[-1] + NOISE * rng.standard_normal(D)
        X = np.asarray(rows)
        self.ic = OnlineIC(3, mint='value', cap=1000).train(X, rng)
        self.atom_recovery = recovery(self.ic.C, atoms).mean()
        self.columns, self.clean = len(self.ic.C), clean_fraction(self.ic.C, atoms)
        self.true_atoms = frozen(atoms, 3)
        codes = np.abs(self.ic.encode(X)[0])
        # co-occurrence predictor: the mean frame in which each column is active
        self.E = (codes.T @ X) / np.maximum(codes.sum(0)[:, None], 1e-9)

    def predict(self, occurrence):
        """Expected follow-up content if a pronoun is bound to this STM occurrence."""
        a = np.abs(self.ic.encode(occurrence)[0][0])
        e = a @ self.E / max(a.sum(), 1e-9)
        u = unit(occurrence)
        return e - (e @ u) * u                    # the candidate's own content is not a prediction

    def reconstruction_credit(self, occurrence, follow):
        """Sparse-coding energy of the bound row "it P" = occurrence + P (two sources)."""
        codes, _, r = self.ic.encode(occurrence + follow, k=2)
        return -(.5 * float((r * r).sum()) + LAM * float(np.abs(codes).sum()))

    def item(self, rng, template, ante, follow='ante'):
        """Two STM candidates (older, recent) and the follow-up "it P ."."""
        kinds = rng.choice(K3, 3, replace=False)
        occ = [self.O[kinds[i]] * rng.uniform(.7, 1.3) + NOISE * rng.standard_normal(D) for i in (0, 1)]
        if follow == 'ante':                      # strong: "it purrs ."
            p = self.CHAR[kinds[ante]]
        elif follow == 'weak':                    # weak: a generic predicate, faint characteristic one
            p = .35 * self.CHAR[kinds[ante]] + self.GEN[rng.integers(4)]
        elif follow == 'generic':
            p = self.GEN[rng.integers(4)]
        else:                                     # 'neither': a third kind's predicate
            p = self.CHAR[kinds[2]]
        o = rng.uniform(.7, 1.3) * p + NOISE * rng.standard_normal(D)
        pos = TEMPLATES[template]
        phi = np.array([[np.dot(unit(self.predict(occ[i])), unit(o)), pos[i][1], pos[i][0]] for i in (0, 1)])
        recon = {'learned': np.array([self.reconstruction_credit(occ[i], o) for i in (0, 1)])}
        learned, self.ic = self.ic, self.true_atoms           # same credit, factorial dictionary
        recon['true atoms'] = np.array([self.reconstruction_credit(occ[i], o) for i in (0, 1)])
        self.ic = learned
        return phi, recon, ante


def q3_set(world, rng, n, regime, informative=CHAR_RATE, strength='ante'):
    names = list(TEMPLATES)
    items = []
    for _ in range(n):
        if regime in ('subject-biased', 'reversed-subject'):
            template = names[[0, 2][rng.integers(2)]]             # T1/T3: role varies
        else:
            template = names[rng.integers(3)]
        subj = [TEMPLATES[template][i][0] for i in (0, 1)]
        if regime in ('counterbalanced', 'control'):
            ante = int(rng.integers(2))
        elif regime == 'recency-biased':
            ante = 1
        elif regime == 'reversed-recency':
            ante = 0
        elif regime == 'subject-biased':
            ante = subj.index(1)
        else:                                                     # reversed-subject
            ante = subj.index(0)
        if regime == 'control':
            follow = 'neither'
        else:
            follow = strength if rng.random() < informative else 'generic'
        items.append((template,) + world.item(rng, template, ante, follow))
    return items


def train_q3(items, rng, credit, lr=.5):
    """REINFORCE: sample a binding from softmax(w . phi), credit it, no labels."""
    w, baseline = np.zeros(3), 0.
    for _, phi, recon, _ in items:
        z = phi @ w
        p = np.exp(z - z.max())
        p /= p.sum()
        c = int(rng.choice(2, p=p))
        reward = phi[c, 0] if credit == 'prediction' else recon[credit.split(': ')[1]][c]
        w += lr * (reward - baseline) * (phi[c] - p @ phi)
        baseline += .05 * (reward - baseline)
    return w


def run_q3(seeds, n_train=3000, n_test=1000):
    rows = []
    trainings = (('prediction', 'counterbalanced'), ('prediction', 'recency-biased'),
                 ('prediction', 'subject-biased'), ('prediction, lr .1', 'counterbalanced'),
                 ('reconstruction: learned', 'counterbalanced'), ('reconstruction: true atoms', 'counterbalanced'))
    keys = ('cb', 'rev_rec', 'rev_subj', 'cb_weak', 'rev_rec_weak', 'rev_subj_weak', 'ctl_recent', 'ctl_subject')
    acc = {t: {key: [] for key in keys + ('w',)} for t in trainings}
    atom_rec, columns, clean, margin, weak_margin = [], [], [], [], []
    recon_sign = {'learned': [], 'true atoms': []}
    for seed in seeds:
        rng = np.random.default_rng(3000 + seed)
        world = Q3World(rng)
        atom_rec.append(world.atom_recovery)
        columns.append(world.columns)
        clean.append(world.clean)
        tests = {}
        for strength, suffix in (('ante', ''), ('weak', '_weak')):
            for name, regime in (('cb', 'counterbalanced'), ('rev_rec', 'reversed-recency'),
                                 ('rev_subj', 'reversed-subject')):
                tests[name + suffix] = q3_set(world, rng, n_test, regime, informative=1., strength=strength)
        tests['ctl'] = q3_set(world, rng, n_test, 'control')
        weak_margin.append(np.mean([phi[ante, 0] - phi[1 - ante, 0] for _, phi, _, ante in tests['cb_weak']]))
        margin.append(np.mean([phi[ante, 0] - phi[1 - ante, 0] for _, phi, _, ante in tests['cb']]))
        for key in ('learned', 'true atoms'):
            recon_sign[key].append(np.mean([rec[key][ante] > rec[key][1 - ante] for _, _, rec, ante in tests['cb']]))
        for credit, regime in trainings:
            lr = .1 if credit.endswith('lr .1') else .5
            w = train_q3(q3_set(world, rng, n_train, regime), np.random.default_rng(seed),
                         credit.split(',')[0], lr=lr)
            a = acc[(credit, regime)]
            a['w'].append(w)
            for key in ('cb', 'rev_rec', 'rev_subj', 'cb_weak', 'rev_rec_weak', 'rev_subj_weak'):
                a[key].append(np.mean([int(np.argmax(phi @ w)) == ante for _, phi, _, ante in tests[key]]))
            picks = [(int(np.argmax(phi @ w)), phi) for _, phi, _, _ in tests['ctl']]
            a['ctl_recent'].append(np.mean([c == 1 for c, _ in picks]))
            a['ctl_subject'].append(np.mean([phi[c, 2] == 1 for c, phi in picks if phi[0, 2] != phi[1, 2]]))
    for (credit, regime), a in acc.items():
        W = np.asarray(a['w'])
        rows.append(dict(credit=credit, train=regime,
                         **{key: ms(a[key]) for key in keys},
                         w_content=ms(W[:, 0]), w_recent=ms(W[:, 1]), w_subject=ms(W[:, 2])))
    return dict(rows=rows, dictionary_recovery=ms(atom_rec), dictionary_columns=ms(columns),
                dictionary_clean=ms(clean), content_margin=ms(margin), weak_margin=ms(weak_margin),
                recon_favours_antecedent={k: ms(v) for k, v in recon_sign.items()})


def table_q3(res):
    def pair(r, key):
        return f"{r[key][0]:.2f} / {r[key + '_weak'][0]:.2f}"
    lines = ['| credit | training data | held-out CB (strong / weak) | reversed recency | reversed subject | '
             'control: picks recent | control: picks subject | w content | w recent | w subject |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for r in res['rows']:
        lines.append(f"| {r['credit']} | {r['train']} | {pair(r, 'cb')} | {pair(r, 'rev_rec')} | {pair(r, 'rev_subj')} | "
                     f"{fmt(r['ctl_recent'])} | {fmt(r['ctl_subject'])} | {fmt(r['w_content'])} | "
                     f"{fmt(r['w_recent'])} | {fmt(r['w_subject'])} |")
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Q4 same-kind individuals and determiners
# ---------------------------------------------------------------------------

N_KIND, N_COL, N_SIZ = 4, 8, 4
DET_RELIABILITY = .9         # "a" marks a first mention, "the" a re-mention, this often


class Q4World:
    """Individuals = (kind, colour, size). A mention names the kind and, each
    with probability 1/2, the colour and the size. Determiners are not content."""

    def __init__(self, rng):
        self.atoms = make_codes(rng, N_KIND + N_COL + N_SIZ)
        self.col0, self.siz0 = N_KIND, N_KIND + N_COL
        self.exclusive = self._learn_exclusivity(rng)

    def _learn_exclusivity(self, rng, n=2000):
        """Property pairs that never share one noun phrase although both are common."""
        props = N_COL + N_SIZ
        counts, single = np.zeros((props, props)), np.zeros(props)
        for _ in range(n):
            present = self.mention(rng, self.individual(rng), .5, .5)[1:]
            idx = [a - N_KIND for a in present]
            single[idx] += 1
            for i in idx:
                for j in idx:
                    if i != j:
                        counts[i, j] += 1
        expected = np.outer(single, single) / n
        exclusive = (counts == 0) & (expected >= 3)
        np.fill_diagonal(exclusive, False)     # a phrase never repeats a property: unobserved, not exclusive
        return exclusive

    def individual(self, rng, kind=None):
        return (int(rng.integers(N_KIND)) if kind is None else kind,
                self.col0 + int(rng.integers(N_COL)), self.siz0 + int(rng.integers(N_SIZ)))

    def mention(self, rng, ind, p_col=.5, p_siz=.5):
        atoms = [ind[0]]
        if rng.random() < p_col:
            atoms.append(ind[1])
        if rng.random() < p_siz:
            atoms.append(ind[2])
        return atoms

    def vector(self, rng, atoms):
        return rng.uniform(.7, 1.3, len(atoms)) @ self.atoms[atoms] + NOISE * rng.standard_normal(D)

    def features(self, np_vec, np_atoms, file_atoms):
        """[residual of the NP under the candidate's columns, learned exclusivity conflict]."""
        A = self.atoms[sorted(set(file_atoms))]
        coef = np.linalg.lstsq(A.T, np_vec, rcond=None)[0]
        residual = float(np.linalg.norm(np_vec - coef @ A))
        conflict = float(any(self.exclusive[i - N_KIND, j - N_KIND]
                             for i in np_atoms[1:] for j in file_atoms if j >= N_KIND))
        return residual, conflict

    def credit(self, rng, file_atoms, truth_ind):
        """Third sentence "it is <colour> <size> ." predicted from the chosen file."""
        o = self.atoms[truth_ind[1]] + self.atoms[truth_ind[2]] + NOISE * rng.standard_normal(D)
        props = sorted({a for a in file_atoms if a >= N_KIND})
        e = self.atoms[props].sum(0) if props else np.zeros(D)
        return -float(((o - e) ** 2).sum())

    def discourse(self, rng, kind='ordinary'):
        """S1 introduces i1; S2 mentions i1 again or a new same-kind individual."""
        i1 = self.individual(rng)
        if kind == 'ordinary':
            same = rng.random() < .5
            det_the = (rng.random() < DET_RELIABILITY) == same
            s1, i2 = self.mention(rng, i1), None
            i2 = i1 if same else self.individual(rng, i1[0])
            s2 = self.mention(rng, i2)
        else:
            same = kind in ('new-info', 'same-content the')
            det_the = kind in ('new-info', 'conflict', 'same-content the')
            if kind == 'conflict':                 # "a black dog V . the white dog V ."
                i2 = self.individual(rng, i1[0])
                while i2[1] == i1[1]:
                    i2 = self.individual(rng, i1[0])
                s1, s2 = [i1[0], i1[1]], [i2[0], i2[1]]
            elif kind == 'new-info':               # "a dog V . the black dog V ."
                i2, s1, s2 = i1, [i1[0]], [i1[0], i1[1]]
            elif kind == 'same-prop':              # "a black dog V . a black dog V ."
                i2 = (i1[0], i1[1], self.siz0 + int(rng.integers(N_SIZ)))
                s1, s2 = [i1[0], i1[1]], [i2[0], i2[1]]
            else:                                  # "a dog V . a/the dog V ."
                i2 = i1 if same else self.individual(rng, i1[0])
                s1, s2 = [i1[0]], [i2[0]]
        n_vec = self.vector(rng, s2)
        residual, conflict = self.features(n_vec, s2, s1)
        return dict(same=bool(same), the=float(det_the), residual=residual, conflict=conflict,
                    s1=s1, s2=s2, i2=i2)


def phi4(d, features):
    out = [d['the'], d['residual']]
    if 'conflict' in features:
        out.append(d['conflict'])
    return np.array(out + [1.])


def train_q4_credit(world, data, rng, features, lr=.2):
    """REINFORCE on bind (to the STM candidate) vs mint; credit = third-sentence prediction."""
    w, baseline = np.zeros(len(phi4(data[0], features))), 0.
    for d in data:
        x = phi4(d, features)
        p = 1. / (1. + np.exp(-np.clip(x @ w, -30, 30)))
        bind = rng.random() < p
        file_atoms = d['s1'] + d['s2'] if bind else d['s2']
        reward = world.credit(rng, file_atoms, d['i2'])
        w += lr * (reward - baseline) * (float(bind) - p) * x
        baseline += .05 * (reward - baseline)
    return w


def train_q4_labels(data, features, lr=.5, epochs=200):
    """Reference: logistic regression on the true identity (labels the toy has, the model does not)."""
    X = np.stack([phi4(d, features) for d in data])
    y = np.array([d['same'] for d in data], dtype=float)
    w = np.zeros(X.shape[1])
    for _ in range(epochs):
        p = 1. / (1. + np.exp(-np.clip(X @ w, -30, 30)))
        w -= lr * X.T @ (p - y) / len(y)
    return w


Q4_TESTS = ('ordinary', 'same-content a', 'same-content the', 'conflict', 'new-info', 'same-prop')


def run_q4(seeds, n_train=3000, n_test=1000):
    choosers = {}
    apart = {key: [] for key in ('named again', 'not named again', 'never named')}
    one_column = []
    for seed in seeds:
        rng = np.random.default_rng(4000 + seed)
        world = Q4World(rng)
        train = [world.discourse(rng) for _ in range(n_train)]
        tests = {kind: [world.discourse(rng, kind) for _ in range(n_test)] for kind in Q4_TESTS}
        rules = {'content only: bind iff residual <= tau': None}
        for features in (('residual',), ('residual', 'conflict')):
            tag = 'det + residual' + (' + learned exclusivity' if 'conflict' in features else '')
            rules[f'credit, {tag}'] = (features, train_q4_credit(world, train, np.random.default_rng(seed), features))
            rules[f'labels (reference), {tag}'] = (features, train_q4_labels(train, features))
        for name, rule in rules.items():
            entry = choosers.setdefault(name, {kind: [] for kind in Q4_TESTS + ('w',)})
            for kind in Q4_TESTS:
                if rule is None:
                    pred = [d['residual'] <= TAU for d in tests[kind]]
                else:
                    pred = [phi4(d, rule[0]) @ rule[1] > 0 for d in tests[kind]]
                entry[kind].append(np.mean([p == d['same'] for p, d in zip(pred, tests[kind])]))
            if rule is not None:
                entry['w'].append(rule[1])
        # content-only unmixing of "a dog V . a dog V ." vs "a dog V . the dog V .": the row
        # of the second sentence is the same vector either way; a dictionary holding the
        # kind columns explains it, so no second column for the second dog can be minted.
        ic = frozen(world.atoms, 3)
        rows = np.stack([world.vector(rng, [d['s2'][0]]) for d in tests['same-content a']])
        one_column.append(np.mean(np.linalg.norm(ic.encode(rows)[2], axis=1) <= TAU))
        # two same-kind individuals in STM, then "the <colour> dog V ." / "the dog V ."
        features, w = rules['credit, det + residual + learned exclusivity']
        for case in apart:
            hits = []
            for _ in range(n_test):
                kind = int(rng.integers(N_KIND))
                a = world.individual(rng, kind)
                b = world.individual(rng, kind)
                while b[1] == a[1]:
                    b = world.individual(rng, kind)
                files = ([[a[0], a[1]], [b[0], b[1]]] if case != 'never named' else [[a[0]], [b[0]]])
                s3 = [a[0], a[1]] if case != 'not named again' else [a[0]]
                n_vec = world.vector(rng, s3)
                scores = []
                for f in files:
                    residual, conflict = world.features(n_vec, s3, f)
                    scores.append(phi4(dict(the=1., residual=residual, conflict=conflict), features) @ w)
                top = np.flatnonzero(np.isclose(scores, max(scores)))
                best = int(rng.choice(top)) if max(scores) > 0 else -1     # ties: coin; -1: mint
                hits.append(best == 0)
            apart[case].append(np.mean(hits))
    rows = []
    for name, e in choosers.items():
        row = dict(chooser=name, **{kind: ms(e[kind]) for kind in Q4_TESTS})
        if e['w']:
            W = np.asarray([np.pad(w, (0, 4 - len(w))) if len(w) == 3 else w for w in e['w']])
            # columns: the, residual, conflict (0 when absent), bias
            if W.shape[1] == 4 and 'exclusivity' not in name:
                W = W[:, [0, 1, 3, 2]]
            row.update(w_the=ms(W[:, 0]), w_residual=ms(W[:, 1]), w_conflict=ms(W[:, 2]), w_bias=ms(W[:, 3]))
        rows.append(row)
    return dict(rows=rows, kept_apart={k: ms(v) for k, v in apart.items()}, same_content_explained=ms(one_column))


def table_q4(res):
    lines = ['| chooser | ordinary | "a dog . a dog" (2) | "a dog . the dog" (1) | conflict: "a black dog . the white dog" (2) | '
             'new info: "a dog . the black dog" (1) | "a black dog . a black dog" (2) | w the | w residual | w exclusivity | w bias |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in res['rows']:
        ws = ' | '.join(fmt(r[k]) if k in r else '-' for k in ('w_the', 'w_residual', 'w_conflict', 'w_bias'))
        lines.append(f"| {r['chooser']} | " + ' | '.join(fmt(r[k]) for k in Q4_TESTS) + f' | {ws} |')
    lines.append('')
    lines.append('| two dogs in STM, then "the ... dog V ." | bound to the right dog |')
    lines.append('|---|---|')
    for case, v in res['kept_apart'].items():
        lines.append(f'| {case} | {fmt(v)} |')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--q', type=int, nargs='*', default=[1, 2, 3, 4])
    args = parser.parse_args()
    seeds = list(range(args.seeds))
    here = os.path.dirname(os.path.abspath(__file__))
    out, md = dict(seeds=seeds, constants=dict(D=D, NOISE=NOISE, LAM=LAM, TAU=TAU, REC=REC, LR=LR,
                                               BATCH=BATCH, sklearn=HAVE_SKLEARN)), []
    started = time.time()
    if 1 in args.q:
        out['q1'] = run_q1(seeds)
        out['q1_ica_conditions'] = run_q1_ica_conditions(seeds)
        md += ['## Q1 factorial vs confounded', table_q1(out['q1']), '',
               'FastICA mean recovery: ' + ', '.join(f'{k} {fmt(v)}' for k, v in out['q1_ica_conditions'].items())]
    if 2 in args.q:
        out['q2'] = run_q2(seeds)
        out['q2_floor'] = residual_floor(seeds)
        out['q2_gap'] = run_q2_gap(seeds)
        md += ['', '## Q2 support curriculum (identification / recovery (columns))', table_q2(out['q2']), '',
               table_q2_extra(out['q2_floor'], out['q2_gap'])]
    if 3 in args.q:
        out['q3'] = run_q3(seeds)
        q3 = out['q3']
        md += ['', '## Q3 pronoun binding credited by prediction', table_q3(q3), '',
               f"dictionary: {fmt(q3['dictionary_columns'], 0)} columns, one-atom {fmt(q3['dictionary_clean'])}; "
               f"content margin antecedent - other: strong {fmt(q3['content_margin'])}, weak {fmt(q3['weak_margin'])}; "
               "reconstruction credit higher for the antecedent: " +
               ', '.join(f'{k} {fmt(v)}' for k, v in q3['recon_favours_antecedent'].items())]
    if 4 in args.q:
        out['q4'] = run_q4(seeds)
        md += ['', '## Q4 same-kind individuals and determiners (accuracy against the true identity)',
               table_q4(out['q4']), '',
               f"second-sentence row explained by the kind column (no mint possible): {fmt(out['q4']['same_content_explained'])}"]
    md.append(f'\n({len(seeds)} seeds, {time.time() - started:.0f} s)')
    text = '\n'.join(md)
    print(text)

    def plain(value):
        if isinstance(value, dict):
            return {str(k): plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value
    with open(os.path.join(here, 'results.json'), 'w') as f:
        json.dump(plain(out), f, indent=1)
    with open(os.path.join(here, 'results.md'), 'w') as f:
        f.write('# Identity-by-unmixing toy: full tables (written by sim.py)\n\n' + text + '\n')


if __name__ == '__main__':
    main()
