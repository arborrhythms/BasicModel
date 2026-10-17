#!/usr/bin/env python
"""Reference resolved at the close of each sentence: a numpy toy.

Run with ``../../../.venv/bin/python sim.py`` from this directory (about one
minute; ``--quick`` for a smoke run). Reads the identity corpus in
``data/identity_from_data``; writes ``results.json`` and ``results.md`` here.

The learner sees: sentence text, the mention heads and their grammatical
roles (the structure a kept derivation supplies), determiner forms and the
pronoun ``it`` (a word with no content). Entity labels, candidate lists,
targets and row-pair answers are read only by the grader.
"""
import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / 'data' / 'identity_from_data'
DETS = ('a', 'the')
PRONOUNS = ('it',)
D = 64
# weight vector layout
W_COMPAT, W_PRED, W_REC, W_LASTROLE, W_CURROLE, W_LOGALPHA, W_DET_A, W_DET_THE, W_DET_PRO = range(9)
WNAMES = ['compat', 'pred', 'recency', 'last_role', 'cur_role', 'log_alpha', 'det_a', 'det_the', 'det_pro']
SALIENCE = [W_REC, W_LASTROLE, W_CURROLE]
DETCUES = [W_DET_A, W_DET_THE, W_DET_PRO]

DEFAULTS = dict(
    rho_c=0.3,      # B1's EMA toward mention content (the code's blend)
    rho_p=0.5,      # weight a predicate word adds to the bound individual's signature
    pronoun_can_mint=False,  # a content-less mention has no residual to mint from
    lam_assoc=10.0,  # pseudo-count of the base rate in the association
    forget=0.9993,   # per predicate word, decay of the association's counts (window ~ one epoch)
    atom_floor=0.3,  # read weight below which an atom is absent from a signature
    nu=0.5,         # explanation level a fresh individual is credited with for its own mention
    exclusive=True,  # learn exclusive atoms (never share a mention) and count them against a candidate
    excl_min=8,      # mentions of each atom before a never-co-occurrence counts as exclusivity
    eta_w=0.2,      # policy weight rate
    beta=1.0,       # sharpness of the credit posterior q ~ exp(beta * R)
    lam_next=1.0,   # weight of the next-sentence term in the credit
    kappa=6.0,      # content likelihood exp(kappa * (explanation - 1))
    alpha_credit=1.0,  # CRP concentration used inside the next-sentence term
    explore=0.05,   # epsilon exploration while training
    soft_assoc=True,  # association counts weighted by the posterior over alternatives (EM)
    beta_assoc=3.0,   # sharpening of that posterior for the association's counts
    decay=1e-4,     # weight decay on policy weights
    epochs=2,
    recurrence=0,   # 0 = no gate; r >= 2: an individual is a row once mentioned r times
    close=True,     # decide at the close (predicate visible); False = during compose (B3)
    pred_in_credit=True,
    learn_salience=True,
    learn_detcue=True,
    fix_det_a=None,  # diagnostic: clamp the 'a' cue instead of learning it
    noise=0.0,      # sd of Gaussian noise added to every mention content
    det_flip=0.0,   # fraction of determiners swapped a<->the
)


# ----------------------------------------------------------------------------
# corpus
def parse(sentences, label):
    by_sentence = defaultdict(list)
    for m in label['mentions']:
        by_sentence[m['sentence']].append(m)
    doc = dict(id=label['id'], family=label['family'], probe=label['probe'], row_pair=label['row_pair'],
               mentions={m['id']: m for m in label['mentions']}, sentences=[])
    for sid, text in enumerate(sentences):
        words = text.split()
        ments = sorted(by_sentence[sid], key=lambda m: m['word'])
        spans, out = [], []
        prev_head = -1
        for m in ments:
            head = m['word']
            det, start = None, head
            for j in range(head - 1, max(prev_head, head - 4), -1):
                if words[j] in DETS:
                    det, start = words[j], j
                    break
            form = words[head]
            content = [] if form in PRONOUNS else [w for w in words[start:head + 1] if w not in DETS]
            out.append(dict(id=m['id'], role=m['role'], det=det, form=form, content=content,
                            entity=m['entity'], pronoun=form in PRONOUNS, head=head))
            spans.append((start, head))
            prev_head = head
        inside = set(i for s, e in spans for i in range(s, e + 1))
        outside = [w for i, w in enumerate(words) if i not in inside and w != '.']
        # the predicate is its head: the last word outside the mention spans
        # ('is white' -> white, 'and runs' -> runs); the copula/conjunction carry nothing
        pred = outside[-1:]
        doc['sentences'].append(dict(mentions=out, pred=pred, text=text))
    return doc


def load(split):
    texts = [json.loads(s) for s in (DATA / f'{split}.text.jsonl').read_text().splitlines()]
    labels = {r['id']: r for s in (DATA / f'{split}.labels.jsonl').read_text().splitlines() for r in [json.loads(s)]}
    return [parse(t['sentences'], labels[t['id']]) for t in texts]


class World:
    """Random meaning codes and the predicate vocabulary, per seed."""

    def __init__(self, docs_all, seed):
        rng = np.random.default_rng(1000 + seed)
        words, preds = set(), set()
        for d in docs_all:
            for s in d['sentences']:
                for m in s['mentions']:
                    words.update(m['content'])
                if s['mentions']:
                    preds.update(s['pred'])
                words.update(s['pred'])
        self.words = sorted(words)
        self.code = {}
        for w in self.words:
            v = rng.standard_normal(D)
            self.code[w] = v / np.linalg.norm(v)
        for p in PRONOUNS:
            self.code[p] = np.zeros(D)
        self.atom_words = [w for w in self.words if w not in PRONOUNS]
        self.atom_codes = np.array([self.code[w] for w in self.atom_words])
        # the dictionary read: coefficients of a vector over the atom codes (exact for
        # linear combinations, since the ~30 codes are independent in 64 dimensions)
        self.read = np.linalg.pinv(self.atom_codes).T          # (M, D): weights = sig @ read.T
        self.atom_index = {w: i for i, w in enumerate(self.atom_words)}
        self.pred_vocab = sorted(preds)
        self.pred_index = {w: i for i, w in enumerate(self.pred_vocab)}
        self.V = len(self.pred_vocab)

    def prepare(self, docs, rng, noise=0.0, det_flip=0.0):
        """Attach content vectors, predicate index lists and (perturbed) determiners."""
        out = []
        for d in docs:
            nd = dict(d)
            nd['sentences'] = []
            for s in d['sentences']:
                ns = dict(s)
                ns['pred_idx'] = np.array([self.pred_index[w] for w in s['pred'] if w in self.pred_index], dtype=int)
                ns['pred_mask'] = np.zeros(len(self.atom_words), dtype=bool)
                for w in s['pred']:
                    if w in self.atom_index:
                        ns['pred_mask'][self.atom_index[w]] = True
                ms = []
                for m in s['mentions']:
                    nm = dict(m)
                    atoms = np.array([self.code[w] for w in m['content']]).reshape(-1, D)
                    if noise > 0:
                        # noise with expected norm `noise` per atom (atom codes have unit norm)
                        atoms = atoms + noise * rng.standard_normal(atoms.shape) / math.sqrt(D)
                    nm['atoms'] = atoms
                    nm['vec'] = atoms.sum(0) if len(atoms) else np.zeros(D)
                    nm['has_content'] = len(atoms) > 0
                    # the mention's atoms as read from its (possibly noisy) vector
                    nm['mask'] = np.clip(nm['vec'] @ self.read.T, 0, 1) >= 0.5
                    if det_flip > 0 and nm['det'] in DETS and rng.random() < det_flip:
                        nm['det'] = 'the' if nm['det'] == 'a' else 'a'
                    ms.append(nm)
                ns['mentions'] = ms
                nd['sentences'].append(ns)
            out.append(nd)
        return out


# ----------------------------------------------------------------------------
# the learner
def log_softmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def explain(sig_weights, mention_mask, exclusive=None):
    """How well each signature explains a mention: mean over the mention's atoms of the
    atom's weight in the signature (asymmetric: a signature may hold more than the
    mention says), minus, when `exclusive` (M, M) is given, the weight of the strongest
    atom the signature holds that excludes it (black vs white). Range [-1, 1]."""
    idx = np.nonzero(mention_mask)[0]
    if len(idx) == 0:
        return np.zeros(len(sig_weights))
    w = sig_weights[:, idx]
    if exclusive is not None:
        contra = (sig_weights[:, None, :] * exclusive[idx][None]).max(2)
        w = w - contra
    return np.clip(w, -1, 1).mean(1)


def unit(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-9)


class Resolver:
    """Membership-by-evidence reference at the close of each sentence (the proposal).

    An individual's signature is a fuzzy set of atoms in code space: a content
    atom of a bound mention enters with weight 1 (saturating union), the
    predicate's head enters with weight ``rho_p`` per occurrence (saturating at
    1). Atom weights are read back through the dictionary's pseudo-inverse. The
    kind<->predicate association is a count table over (atom, predicate),
    updated with the posterior responsibilities over the alternatives (EM).
    """

    def __init__(self, world, cfg, rng):
        self.world, self.cfg, self.rng = world, cfg, rng
        M = len(world.atom_codes)
        self.n_wa = np.zeros((world.V, M)); self.n_a = np.zeros(M); self.n_w = np.zeros(world.V); self.N = 0
        # exclusivity learned from the text: content atoms that never share a mention
        self.cooc = np.zeros((M, M)); self.seen = np.zeros(M)
        self.exclusive = None
        self.w = np.array([1.0, 1.0, 0, 0, 0, 0, 0, 0, 0])
        self.mask = np.ones(9)
        if not cfg['close'] or not cfg['pred_in_credit']:
            self.w[W_PRED] = 0
            self.mask[W_PRED] = 0
        if not cfg['learn_salience']:
            self.mask[SALIENCE] = 0
        if not cfg['learn_detcue']:
            self.mask[DETCUES] = 0
        if cfg['fix_det_a'] is not None:
            self.w[W_DET_A] = cfg['fix_det_a']; self.mask[W_DET_A] = 0
        self.training = True
        self.credit_log = []

    # --- signature updates (saturating union in code space) ------------------
    def absorb(self, sig, mask, cap):
        """Raise the weight of each atom in `mask` toward 1 by at most cap (saturating union)."""
        if not mask.any():
            return sig
        w = sig @ self.world.read.T
        add = np.clip(1.0 - w, 0.0, cap) * mask
        return sig + add @ self.world.atom_codes

    def note_mention(self, mask):
        if not self.cfg['exclusive'] or not mask.any():
            return
        self.cooc += np.outer(mask, mask); self.seen += mask
        m = self.cfg['excl_min']
        self.exclusive = ((self.cooc == 0) & (self.seen[:, None] >= m) & (self.seen[None, :] >= m)).astype(float)

    # --- predicate model -----------------------------------------------------
    # A count-based kind<->predicate association over the signature's atoms:
    # log p(w | s) = log base(w) + mean_a weight_a(s) * (log p(w | a) - log base(w)),
    # with p(w | a) = (n(w,a) + lam * base(w)) / (n(a) + lam). Atom weights below
    # `atom_floor` read as absent. Unseen (atom, predicate) pairs predict the base
    # rate, never worse.
    def atom_weights(self, sigs):
        w = np.clip(sigs @ self.world.read.T, 0, 1)
        w[w < self.cfg['atom_floor']] = 0
        return w

    def logratio(self):
        base = (self.n_w + 1) / (self.N + self.world.V)
        lam = self.cfg['lam_assoc']
        p = (self.n_wa + lam * base[:, None]) / (self.n_a[None, :] + lam)
        lr = np.log(p) - np.log(base)[:, None]
        lr[self.n_w == 0] = 0.0          # a predicate word never seen carries no evidence
        return np.log(base), lr

    def llpred(self, sigs, pred_idx):
        """Sum over the predicate words of log p(w | sig), per row of sigs."""
        if len(pred_idx) == 0:
            return np.zeros(len(sigs))
        logbase, lr = self.logratio()
        W = self.atom_weights(sigs)
        # weighted average of the atoms' log-ratios (a geometric-mean expert), so the
        # evidence of a signature is bounded by one atom's, however many atoms it holds
        W = W / np.maximum(W.sum(1, keepdims=True), 1.0)
        return (logbase[pred_idx].sum() + W @ lr[pred_idx].sum(0))

    def assoc_update(self, sigs, pred_idx, weights=None):
        """Count (atom, predicate) co-occurrences; `weights` are the responsibilities of
        the alternative signatures (soft EM); hard assignment when None."""
        if len(pred_idx) == 0:
            return
        W = self.atom_weights(np.atleast_2d(sigs))
        if weights is None:
            weights = np.ones(len(W))
        Wsum = weights @ W
        f = self.cfg['forget'] ** len(pred_idx)
        self.n_wa *= f; self.n_a *= f; self.n_w *= f; self.N *= f
        for w in pred_idx:
            self.n_wa[w] += Wsum
            self.n_w[w] += 1
        self.n_a += len(pred_idx) * Wsum
        self.N += len(pred_idx)

    # --- document -------------------------------------------------------------
    def run_document(self, doc):
        cfg = self.cfg
        sig = np.zeros((0, D))
        count, last_pos, last_role = [], [], []
        refs = {}
        pos = 0
        sents = doc['sentences']
        for t, s in enumerate(sents):
            nxt = sents[t + 1] if t + 1 < len(sents) else None
            for m in s['mentions']:
                K = len(sig)
                c = m['vec']
                has_c = m['has_content']
                det = m['det']
                detkey = W_DET_A if det == 'a' else (W_DET_THE if det == 'the' else W_DET_PRO)
                currole = 0.5 if m['role'] == 'subject' else -0.5
                # candidates ordered most recent first (argmax tie-break = recency)
                order = np.argsort(-np.array(last_pos)) if K else np.array([], dtype=int)
                # post-content signatures of every alternative (K candidates + NEW)
                if self.training and has_c:
                    self.note_mention(m['mask'])
                if has_c and K:
                    compat = explain(self.atom_weights(sig[order]), m['mask'], self.exclusive)
                    post = self.absorb(sig[order], m['mask'], 1.0)
                else:
                    compat = np.zeros(K)
                    post = sig[order].copy()
                alts = np.vstack([post, c[None]])  # last row = NEW
                phi = np.zeros((K + 1, 9))
                phi[:K, W_COMPAT] = compat
                phi[:K, W_REC] = -np.arange(K)
                phi[:K, W_LASTROLE] = [0.5 if last_role[i] == 'subject' else -0.5 for i in order]
                phi[:K, W_CURROLE] = currole
                phi[K, W_LOGALPHA] = 1.0
                phi[K, detkey] = 1.0
                if cfg['close']:
                    ll = self.llpred(alts, s['pred_idx'])
                    ll0 = self.llpred(np.zeros((1, D)), s['pred_idx'])[0]
                    phi[:, W_PRED] = ll - ll0
                score = phi @ self.w
                if K and not has_c and not cfg['pronoun_can_mint']:
                    score[K] = -np.inf     # no content, no residual: nothing to mint
                choice = int(np.argmax(score))
                if self.training and cfg['explore'] > 0 and self.rng.random() < cfg['explore']:
                    allowed = np.nonzero(np.isfinite(score))[0]
                    choice = int(self.rng.choice(allowed))
                if self.training:
                    q = self.learn(phi, score, alts, compat, order, sig, count, s, nxt, has_c, c)
                    if cfg['soft_assoc']:
                        qa = q ** cfg['beta_assoc']; qa /= qa.sum()
                        self.assoc_update(alts, s['pred_idx'], qa)
                # apply the choice
                if choice == K:
                    sig = np.vstack([sig, c[None]])
                    count.append(1); last_pos.append(pos); last_role.append(m['role'])
                    idx = K
                else:
                    idx = int(order[choice])
                    sig[idx] = alts[choice]
                    count[idx] += 1; last_pos[idx] = pos; last_role[idx] = m['role']
                refs[m['id']] = idx
                if self.training and not cfg['soft_assoc']:
                    self.assoc_update(sig[idx], s['pred_idx'])
                if cfg['rho_p'] > 0:
                    sig[idx] = self.absorb(sig[idx][None], s['pred_mask'], cfg['rho_p'])[0]
                pos += 1
        r = cfg['recurrence']
        durable = [i for i in range(len(count)) if r < 2 or count[i] >= r]
        return refs, dict(minted=len(durable), individuals=len(count), durable=set(durable))

    # --- credit ---------------------------------------------------------------
    def learn(self, phi, score, alts, compat, order, sig, count, s, nxt, has_c, c):
        """Posterior responsibilities over the alternatives from the evidence
        (this mention's content, this sentence's predicate, the next sentence),
        then one M step on the policy weights."""
        cfg = self.cfg
        K = len(order)
        R = np.zeros(K + 1)
        if has_c:
            R[:K] += cfg['kappa'] * (compat - 1.0)
            R[K] += cfg['kappa'] * (cfg['nu'] - 1.0)
        if cfg['pred_in_credit']:
            R += self.llpred(alts, s['pred_idx'])
        if nxt is not None and nxt['mentions'] and cfg['lam_next'] > 0:
            for a in range(K + 1):
                # STM after alternative a
                if a == K:
                    sig_a = np.vstack([sig, c[None]])
                    n_a = np.array(count + [1], dtype=float)
                    j = K
                else:
                    sig_a = sig.copy()
                    j = int(order[a])
                    sig_a[j] = alts[a]
                    n_a = np.array(count, dtype=float); n_a[j] += 1
                if cfg['rho_p'] > 0:
                    sig_a[j] = self.absorb(sig_a[j][None], s['pred_mask'], cfg['rho_p'])[0]
                N = n_a.sum() + cfg['alpha_credit']
                for m2 in nxt['mentions']:
                    logw = np.log(np.concatenate([n_a, [cfg['alpha_credit']]]) / N)
                    if m2['has_content']:
                        comp = np.concatenate([explain(self.atom_weights(sig_a), m2['mask'], self.exclusive), [cfg['nu']]])
                        logw = logw + cfg['kappa'] * (comp - 1)
                    if cfg['pred_in_credit'] and len(nxt['pred_idx']):
                        logw = logw + self.llpred(np.vstack([sig_a, m2['vec'][None]]), nxt['pred_idx'])
                    mx = logw.max()
                    R[a] += cfg['lam_next'] * (mx + np.log(np.exp(logw - mx).sum()))
        if K and not has_c and not cfg['pronoun_can_mint']:
            R[K] = -np.inf
        q = np.exp(cfg['beta'] * (R - R.max())); q /= q.sum()
        pi = np.exp(score - score.max()); pi /= pi.sum()
        grad = (q - pi) @ phi
        self.w += cfg['eta_w'] * grad * self.mask
        self.w -= cfg['decay'] * self.w * self.mask
        self.credit_log.append(float(R[int(np.argmax(score))] - R.max()))
        return q


class RuleBaseline:
    """B1: the current code's rule (declared modes, cosine argmax, recency tie-break).
    B2: recency only (declared modes, most recent individual)."""

    def __init__(self, world, cfg, rng, kind):
        self.world, self.cfg, self.kind = world, cfg, kind
        self.training = False
        self.w = np.zeros(9)
        self.credit_log = []

    def run_document(self, doc):
        cfg = self.cfg
        sig = np.zeros((0, D)); count, last_pos = [], []
        refs = {}; pos = 0
        for s in doc['sentences']:
            for m in s['mentions']:
                K = len(sig); c = m['vec']
                order = np.argsort(-np.array(last_pos)) if K else np.array([], dtype=int)
                if m['det'] == 'a' or K == 0:
                    sig = np.vstack([sig, c[None]]); count.append(1); last_pos.append(pos); idx = K
                else:
                    if self.kind == 'B1' and m['has_content']:
                        cos = unit(sig[order]) @ unit(c)
                        idx = int(order[int(np.argmax(cos))])
                    else:
                        idx = int(order[0])
                    if m['has_content']:
                        sig[idx] = (1 - cfg['rho_c']) * sig[idx] + cfg['rho_c'] * c
                    count[idx] += 1; last_pos[idx] = pos
                refs[m['id']] = idx
                pos += 1
        r = cfg['recurrence']
        durable = [i for i in range(len(count)) if r < 2 or count[i] >= r]
        return refs, dict(minted=len(durable), individuals=len(count), durable=set(durable))


# ----------------------------------------------------------------------------
# grading (labels used only here)
def grade(doc, refs, info, strict):
    def ref(mid):
        v = refs.get(mid)
        if v is None or (strict and v not in info['durable']):
            return None
        return v
    out = dict(family=doc['family'], binding=None, rows=None, mentions=[], minted=info['minted'],
               individuals=info['individuals'])
    probe = doc['probe']
    if probe:
        cands = [ref(mid) for mid in probe['candidates']]
        sel = ref(probe['mention'])
        complete = all(v is not None for v in cands)
        distinct = complete and len(set(cands)) == len(cands)
        resolved = distinct and sel is not None and sel in cands
        target = ref(probe['target'])
        out['binding'] = dict(candidate_count=probe['candidate_count'], recency_rank=probe['recency_rank'],
                              role=probe['role'], informative=probe['informative'], resolved=resolved,
                              correct=bool(resolved and sel == target), collision=bool(complete and not distinct),
                              chance=1 / len(cands))
    pair = doc['row_pair']
    if pair:
        vals = [ref(mid) for mid in pair['mentions']]
        resolved = all(v is not None for v in vals)
        actual = len(set(vals)) if resolved else None
        out['rows'] = dict(expected=pair['expected'], actual=actual, resolved=resolved,
                           correct=actual == pair['expected'])
    # every mention after its first sentence: same individual as the entity's previous mention,
    # or a fresh one when the entity is new
    seen_entity = {}
    seen_refs = set()
    for s_i, s in enumerate(doc['sentences']):
        for m in s['mentions']:
            v = ref(m['id'])
            if s_i > 0:
                prev = seen_entity.get(m['entity'])
                if prev is None:
                    correct = v is not None and v not in seen_refs
                else:
                    correct = v is not None and v == prev
                out['mentions'].append(dict(correct=bool(correct), resolved=v is not None,
                                            pronoun=m['pronoun'], det=m['det']))
            if v is not None:
                seen_refs.add(v)
            if v is not None:
                seen_entity[m['entity']] = v
    return out


def summarize(graded):
    groups = defaultdict(list)
    for g in graded:
        b = g['binding']
        if b:
            for name in ('probes', f"family={g['family']}", f"candidates={b['candidate_count']}",
                         f"recency={b['recency_rank']}", f"role={b['role']}"):
                if name.startswith(('candidates', 'recency', 'role')) and not b['informative']:
                    continue
                groups[name].append(b)
        if g['rows']:
            groups[f"rows={g['family']}"].append(g['rows'])
            groups[f"rows={g['family']}/{g['rows']['expected']}"].append(g['rows'])
            groups['rows'].append(g['rows'])
        for mm in g['mentions']:
            groups[f"mentions={g['family']}"].append(mm)
            groups['mentions'].append(mm)
    summary = {}
    for name, vals in groups.items():
        n = len(vals)
        summary[name] = dict(n=n, accuracy=sum(v['correct'] for v in vals) / n,
                             coverage=sum(v['resolved'] for v in vals) / n)
        if 'collision' in vals[0]:
            summary[name]['collisions'] = sum(v['collision'] for v in vals) / n
            summary[name]['chance'] = sum(v['chance'] for v in vals) / n
    summary['minted_per_doc'] = float(np.mean([g['minted'] for g in graded]))
    summary['individuals_per_doc'] = float(np.mean([g['individuals'] for g in graded]))
    return summary


# ----------------------------------------------------------------------------
# experiments
SPLITS = {}
ALL_DOCS = []


def get_split(name):
    if name not in SPLITS:
        SPLITS[name] = load(name)
    return SPLITS[name]


def make_model(kind, world, cfg, rng):
    if kind in ('B1', 'B2'):
        return RuleBaseline(world, cfg, rng, kind)
    return Resolver(world, cfg, rng)


def train(model, docs, epochs, rng):
    model.training = True
    for _ in range(epochs):
        for i in rng.permutation(len(docs)):
            model.run_document(docs[i])


def evaluate(model, docs, strict):
    model.training = False
    graded = [grade(d, *model.run_document(d), strict=strict) for d in docs]
    return summarize(graded)


def run_one(kind, seed, cfg, train_plan=None, eval_splits=('eval', 'reversed_eval')):
    """train_plan: list of (split name or doc list, epochs); default = train split."""
    rng = np.random.default_rng(seed)
    world = World(ALL_DOCS, seed)
    model = make_model(kind, world, cfg, rng)
    prep = {}

    def prepared(name):
        if name not in prep:
            prep[name] = world.prepare(get_split(name), rng, cfg['noise'], cfg['det_flip'])
        return prep[name]

    if kind not in ('B1', 'B2'):
        plan = train_plan or [('train', cfg['epochs'])]
        for item, epochs in plan:
            docs = prepared(item) if isinstance(item, str) else item
            train(model, docs, epochs, rng)
    out = dict(weights=dict(zip(WNAMES, map(float, model.w))))
    for name in eval_splits:
        out[name] = evaluate(model, prepared(name), strict=cfg['recurrence'] >= 2)
        if cfg['recurrence'] >= 2:
            out[name + '_lenient'] = evaluate(model, prepared(name), strict=False)
    if model.credit_log:
        out['train_regret'] = float(np.mean(model.credit_log[-500:]))
    if getattr(model, 'exclusive', None) is not None:
        ex = model.exclusive
        pairs = [(world.atom_words[i], world.atom_words[j]) for i in range(len(ex)) for j in range(i + 1, len(ex)) if ex[i, j]]
        out['exclusive_pairs'] = pairs
    return out


def agg(runs, path):
    """mean ± sd of a nested numeric path across seeds."""
    vals = []
    for r in runs:
        v = r
        try:
            for p in path:
                v = v[p]
        except (KeyError, TypeError):
            continue
        if v is not None:
            vals.append(float(v))
    if not vals:
        return None
    return dict(mean=float(np.mean(vals)), sd=float(np.std(vals)), n=len(vals))


def fmt(a, digits=2):
    if a is None:
        return '—'
    if a['sd'] < 0.005:
        return f"{a['mean']:.{digits}f}"
    return f"{a['mean']:.{digits}f} ± {a['sd']:.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--seeds', type=int, default=5)
    args = ap.parse_args()
    t0 = time.time()
    for name in ('train', 'eval', 'biased_train', 'reversed_eval'):
        ALL_DOCS.extend(get_split(name))
    seeds = list(range(args.seeds if not args.quick else 2))
    results = dict(config=DEFAULTS, seeds=seeds, runs={})

    def suite(label, kind, overrides=None, train_plan=None, eval_splits=('eval', 'reversed_eval')):
        cfg = dict(DEFAULTS); cfg.update(overrides or {})
        if args.quick:
            cfg['epochs'] = 1
        runs = [run_one(kind, s, cfg, train_plan, eval_splits) for s in seeds]
        results['runs'][label] = dict(kind=kind, overrides=overrides or {}, runs=runs)
        print(f"{label:28s} {time.time() - t0:6.1f}s  probes {fmt(agg(runs, ['eval', 'probes', 'accuracy']))}"
              f"  rows {fmt(agg(runs, ['eval', 'rows', 'accuracy']))}"
              f"  rev {fmt(agg(runs, ['reversed_eval', 'probes', 'accuracy']))}", flush=True)
        return runs

    # main comparison
    suite('proposal', 'proposal')
    suite('B1 code rule', 'B1')
    suite('B2 recency', 'B2')
    suite('B3 during compose', 'proposal', dict(close=False))
    # recurrence gate
    suite('proposal gate r=2', 'proposal', dict(recurrence=2))
    suite('proposal gate r=3', 'proposal', dict(recurrence=3))
    suite('B1 gate r=2', 'B1', dict(recurrence=2))
    # epochs
    suite('proposal 1 epoch', 'proposal', dict(epochs=1))
    suite('proposal 3 epochs', 'proposal', dict(epochs=3))
    # ablations
    suite('abl no prediction', 'proposal', dict(pred_in_credit=False))
    suite('abl no salience', 'proposal', dict(learn_salience=False))
    suite('abl no det cue', 'proposal', dict(learn_detcue=False))
    suite('abl no next-sentence credit', 'proposal', dict(lam_next=0.0))
    suite('abl no predicate absorption', 'proposal', dict(rho_p=0.0))
    suite('abl no exclusivity', 'proposal', dict(exclusive=False))
    suite('abl hard association', 'proposal', dict(soft_assoc=False))
    suite('abl pronoun may mint', 'proposal', dict(pronoun_can_mint=True))
    suite('abl no exploration', 'proposal', dict(explore=0.0))
    suite('diag det_a fixed 3', 'proposal', dict(fix_det_a=3.0))
    suite('variant flat prior nu=1', 'proposal', dict(nu=1.0))
    suite('variant nu=1 no exclusivity', 'proposal', dict(nu=1.0, exclusive=False))
    suite('variant nu=1 no det cue', 'proposal', dict(nu=1.0, learn_detcue=False))
    suite('variant nu=1 no next-sentence credit', 'proposal', dict(nu=1.0, lam_next=0.0))
    suite('variant nu=1 det flip .2', 'proposal', dict(nu=1.0, det_flip=0.2))
    suite('variant nu=1 noise 1.0', 'proposal', dict(nu=1.0, noise=1.0))
    # robustness
    for sd in (0.3, 0.6, 1.0):
        suite(f'proposal noise {sd}', 'proposal', dict(noise=sd))
        suite(f'B1 noise {sd}', 'B1', dict(noise=sd))
    suite('proposal det flip .2', 'proposal', dict(det_flip=0.2))
    suite('B1 det flip .2', 'B1', dict(det_flip=0.2))
    suite('B3 det flip .2', 'proposal', dict(det_flip=0.2, close=False))
    # Q3: matched recency branches. warm-up = train without the two-candidate pronoun
    # family; then 32 balanced presentations (train's pronoun family) or 16 biased x 2.
    train_docs = get_split('train')
    warm = [d for d in train_docs if d['family'] != 'pronoun']
    balanced = [d for d in train_docs if d['family'] == 'pronoun']
    for label, branch, reps, over in (('branch balanced', balanced, 1, {}), ('branch biased', 'biased_train', 2, {}),
                                      ('branch biased x4', 'biased_train', 8, {}),
                                      ('branch balanced (B3)', balanced, 1, dict(close=False)),
                                      ('branch biased (B3)', 'biased_train', 2, dict(close=False))):
        cfg = dict(DEFAULTS); cfg.update(over)
        runs = []
        for s in seeds:
            rng = np.random.default_rng(s)
            world = World(ALL_DOCS, s)
            model = Resolver(world, cfg, rng)
            ep = 1 if args.quick else cfg['epochs']
            train(model, world.prepare(warm, rng), ep, rng)
            docs = world.prepare(balanced if isinstance(branch, list) else get_split(branch), rng)
            train(model, docs, ep * reps, rng)
            out = dict(weights=dict(zip(WNAMES, map(float, model.w))))
            for name in ('eval', 'reversed_eval'):
                out[name] = evaluate(model, world.prepare(get_split(name), rng), strict=False)
            runs.append(out)
        results['runs'][label] = dict(kind='proposal', overrides={'branch': label}, runs=runs)
        print(f"{label:28s} {time.time() - t0:6.1f}s  rev {fmt(agg(runs, ['reversed_eval', 'probes', 'accuracy']))}"
              f"  recency w {fmt(agg(runs, ['weights', 'recency']))}", flush=True)
    results['elapsed_s'] = time.time() - t0
    (HERE / 'results.json').write_text(json.dumps(results, separators=(',', ':')))
    (HERE / 'results.md').write_text(render(results))
    print(f'done in {results["elapsed_s"]:.0f}s')


# ----------------------------------------------------------------------------
# tables
FAMILIES = ['pronoun_one', 'pronoun', 'pronoun_three', 'document_context', 'neither_candidate',
            'determiner', 'determiner_conflict', 'same_kind']


def cell(results, label, path, digits=2):
    r = results['runs'].get(label)
    return fmt(agg(r['runs'], path), digits) if r else '—'


def render(results):
    R = results['runs']
    L = []
    L.append('# Results: reference at the close (toy), full tables\n')
    L.append(f"Seeds {results['seeds']}; elapsed {results.get('elapsed_s', 0):.0f}s; defaults `{json.dumps(results['config'])}`\n")
    main_cols = ['proposal', 'B1 code rule', 'B2 recency', 'B3 during compose']
    L.append('## Q1 Binding accuracy by family (eval split; probe for probe families, '
             'the final `it` for determiner families, the two `the colour kind` mentions for same_kind)\n')
    L.append('| family | n | ' + ' | '.join(main_cols) + ' |')
    L.append('|---|---|' + '---|' * len(main_cols))
    for fam in FAMILIES:
        if fam in ('determiner', 'determiner_conflict', 'same_kind'):
            path = lambda c: ['eval', f'mentions={fam}', 'accuracy']
            n = R['proposal']['runs'][0]['eval'].get(f'mentions={fam}', {}).get('n', 0)
            fam_label = fam + ' (mentions)'
        else:
            path = lambda c: ['eval', f'family={fam}', 'accuracy']
            n = R['proposal']['runs'][0]['eval'].get(f'family={fam}', {}).get('n', 0)
            fam_label = fam
        L.append(f'| {fam_label} | {n} | ' + ' | '.join(cell(results, c, path(c)) for c in main_cols) + ' |')
    L.append(f'| all probes | {R["proposal"]["runs"][0]["eval"]["probes"]["n"]} | ' +
             ' | '.join(cell(results, c, ['eval', 'probes', 'accuracy']) for c in main_cols) + ' |')
    L.append(f'| all graded mentions | {R["proposal"]["runs"][0]["eval"]["mentions"]["n"]} | ' +
             ' | '.join(cell(results, c, ['eval', 'mentions', 'accuracy']) for c in main_cols) + ' |')
    L.append('')
    L.append('By recency rank and role (informative probes only):\n')
    L.append('| group | n | ' + ' | '.join(main_cols) + ' |')
    L.append('|---|---|' + '---|' * len(main_cols))
    for grp in ('candidates=1', 'candidates=2', 'candidates=3', 'recency=0', 'recency=1', 'recency=2',
                'role=subject', 'role=object'):
        n = R['proposal']['runs'][0]['eval'].get(grp, {}).get('n', 0)
        L.append(f'| {grp} | {n} | ' + ' | '.join(cell(results, c, ['eval', grp, 'accuracy']) for c in main_cols) + ' |')
    L.append('')
    L.append('## Q2 Row pairs (one vs two individuals), eval split\n')
    q2_cols = main_cols + ['variant flat prior nu=1', 'abl no exclusivity', 'proposal det flip .2']
    L.append('| family / expected rows | n | ' + ' | '.join(q2_cols) + ' |')
    L.append('|---|---|' + '---|' * len(q2_cols))
    for fam in ('determiner', 'determiner_conflict', 'same_kind'):
        for exp in (1, 2):
            grp = f'rows={fam}/{exp}'
            n = R['proposal']['runs'][0]['eval'].get(grp, {}).get('n', 0)
            if n:
                L.append(f'| {fam} / {exp} | {n} | ' + ' | '.join(cell(results, c, ['eval', grp, 'accuracy']) for c in q2_cols) + ' |')
    L.append(f'| all row pairs | {R["proposal"]["runs"][0]["eval"]["rows"]["n"]} | ' +
             ' | '.join(cell(results, c, ['eval', 'rows', 'accuracy']) for c in q2_cols) + ' |')
    L.append(f'| individuals per document | | ' +
             ' | '.join(cell(results, c, ['eval', 'individuals_per_doc']) for c in q2_cols) + ' |')
    L.append('')
    L.append('## Q3 Reversed recency\n')
    L.append('| model | reversed_eval acc | eval pronoun acc | learned recency w | log α | det_pro |')
    L.append('|---|---|---|---|---|---|')
    for lab in ('proposal', 'branch balanced', 'branch biased', 'branch biased x4', 'branch balanced (B3)',
                'branch biased (B3)', 'B1 code rule', 'B2 recency', 'B3 during compose'):
        if lab in R:
            L.append(f'| {lab} | {cell(results, lab, ["reversed_eval", "probes", "accuracy"])} | '
                     f'{cell(results, lab, ["eval", "family=pronoun", "accuracy"])} | '
                     f'{cell(results, lab, ["weights", "recency"])} | {cell(results, lab, ["weights", "log_alpha"])} | '
                     f'{cell(results, lab, ["weights", "det_pro"])} |')
    L.append('')
    L.append('## Q4 Learned weights (mean ± sd over seeds) and the recurrence gate\n')
    L.append('| run | ' + ' | '.join(WNAMES) + ' |')
    L.append('|---|' + '---|' * len(WNAMES))
    for lab in ('proposal', 'proposal 1 epoch', 'proposal 3 epochs', 'B3 during compose', 'branch balanced',
                'branch biased', 'branch biased x4', 'abl no prediction', 'abl no salience', 'abl no det cue',
                'abl no next-sentence credit', 'abl no predicate absorption', 'abl no exclusivity',
                'abl hard association', 'abl pronoun may mint', 'abl no exploration', 'diag det_a fixed 3',
                'variant flat prior nu=1', 'variant nu=1 no exclusivity', 'variant nu=1 no det cue',
                'variant nu=1 no next-sentence credit', 'variant nu=1 det flip .2', 'variant nu=1 noise 1.0',
                'proposal det flip .2'):
        if lab in R:
            L.append(f'| {lab} | ' + ' | '.join(cell(results, lab, ['weights', w]) for w in WNAMES) + ' |')
    L.append('')
    L.append('| run | individuals/doc | rows (durable)/doc | probes strict | probes lenient | row pairs strict | row pairs lenient |')
    L.append('|---|---|---|---|---|---|---|')
    for lab in ('proposal', 'proposal gate r=2', 'proposal gate r=3', 'B1 code rule', 'B1 gate r=2'):
        gate = 'gate' in lab
        L.append(f'| {lab} | {cell(results, lab, ["eval", "individuals_per_doc"])} | {cell(results, lab, ["eval", "minted_per_doc"])} | '
                 f'{cell(results, lab, ["eval", "probes", "accuracy"])} | '
                 f'{cell(results, lab, ["eval_lenient", "probes", "accuracy"]) if gate else cell(results, lab, ["eval", "probes", "accuracy"])} | '
                 f'{cell(results, lab, ["eval", "rows", "accuracy"])} | '
                 f'{cell(results, lab, ["eval_lenient", "rows", "accuracy"]) if gate else cell(results, lab, ["eval", "rows", "accuracy"])} |')
    L.append('')
    L.append('## Q5 Robustness\n')
    L.append('| condition | model | probes | pronoun | pronoun_three | document_context | row pairs | det. mentions |')
    L.append('|---|---|---|---|---|---|---|---|')
    for lab in ('proposal', 'B1 code rule', 'proposal noise 0.3', 'B1 noise 0.3', 'proposal noise 0.6', 'B1 noise 0.6',
                'proposal noise 1.0', 'B1 noise 1.0', 'variant nu=1 noise 1.0', 'proposal det flip .2', 'B1 det flip .2',
                'B3 det flip .2', 'variant nu=1 det flip .2'):
        if lab in R:
            L.append(f'| {lab} | {R[lab]["kind"]} | {cell(results, lab, ["eval", "probes", "accuracy"])} | '
                     f'{cell(results, lab, ["eval", "family=pronoun", "accuracy"])} | '
                     f'{cell(results, lab, ["eval", "family=pronoun_three", "accuracy"])} | '
                     f'{cell(results, lab, ["eval", "family=document_context", "accuracy"])} | '
                     f'{cell(results, lab, ["eval", "rows", "accuracy"])} | '
                     f'{cell(results, lab, ["eval", "mentions=determiner", "accuracy"])} |')
    L.append('')
    L.append('## Q6 Ablations and epochs\n')
    L.append('| run | probes | pronoun | pronoun_three | document_context | neither | row pairs | same_kind rows | reversed_eval |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for lab in ('proposal', 'proposal 1 epoch', 'proposal 3 epochs', 'B3 during compose', 'abl no prediction',
                'abl no salience', 'abl no det cue', 'abl no next-sentence credit', 'abl no predicate absorption',
                'abl no exclusivity', 'abl hard association', 'abl pronoun may mint', 'abl no exploration',
                'diag det_a fixed 3', 'variant flat prior nu=1', 'variant nu=1 no exclusivity',
                'variant nu=1 no det cue', 'variant nu=1 no next-sentence credit', 'variant nu=1 det flip .2',
                'variant nu=1 noise 1.0'):
        if lab in R:
            L.append(f'| {lab} | {cell(results, lab, ["eval", "probes", "accuracy"])} | '
                     f'{cell(results, lab, ["eval", "family=pronoun", "accuracy"])} | '
                     f'{cell(results, lab, ["eval", "family=pronoun_three", "accuracy"])} | '
                     f'{cell(results, lab, ["eval", "family=document_context", "accuracy"])} | '
                     f'{cell(results, lab, ["eval", "family=neither_candidate", "accuracy"])} | '
                     f'{cell(results, lab, ["eval", "rows", "accuracy"])} | '
                     f'{cell(results, lab, ["eval", "rows=same_kind", "accuracy"])} | '
                     f'{cell(results, lab, ["reversed_eval", "probes", "accuracy"])} |')
    L.append('')
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    main()
