"""Truth-grounded reasoning over the grammar's introspection query ops.

All queries reduce to two judgments over concept extensions -- ``isTrue(A)`` and
``isPart(A, B)`` -- evaluated by the *hard* tools enumerated in the grammar's
``<queries>`` section (data/complete.grammar). A small NeuralToolUser (later
phases) sequences these tools as a soft policy while the judgments stay exact.

Grammar ``<queries>`` op  ->  tool method here:

    exist(X)        -> exist(X)        : X is an ultimate truth OR a single idea
                                         with positive trust  (the isTrue leaf).
    equal(X, Y)     -> equal(X, Y)     : isomorphic=True -> fraction of shared
                                         parts & wholes; False -> norm of diff.
    part(X, Y)      -> part(X, Y)      : graded parthood X ⊆ Y in [0, 1].
    query(X[, Y])   -> query(X[, Y])   : an LTM lookup (the .where-typed read;
                                         soft full-truth-space version = Phase 3).
    quantize(X)     -> quantize(X)     : snap X onto the nearest real idea.
    wholes(X)       -> wholes(X)       : the proximal containing wholes of X
                                         (transitivity gives the rest).
    parts(X)        -> parts(X)        : the proximal contained parts of X.

A chain of parthood reasoning is just repeated ``wholes()`` toward the goal --
the same modus-ponens climb ``ConceptualSpace.reason`` performs, expressed via
the canonical primitive. (``isomorphs`` was removed from the grammar; it is
subsumed by ``equal(isomorphic=True)``.)

Conceptual framing + phase map: doc/plans/2026-06-23-truth-grounded-reasoning.md.
The reasoner reads the unified reasoning store (TernaryTruthStore under
<ltmConsolidation>, else RelativeTruthStore) and the model's absolute
TruthLayer; it mutates the store only via ``materialize``. Unused by the forward
path until the <queryReasoning> consumer lands (Phase 5), so it is dark.
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

from Spaces import ConceptualSpace, GlobalAttention
from Layers import TernaryTruthStore


# -- Query framing (Phase 0) -------------------------------------------------

KIND_IS_TRUE = "isTrue"
KIND_IS_PART = "isPart"
KIND_IS_EQUAL = "isEqual"   # sugar: equal(A, B)

# Map every query surface form to one of the three reduction kinds. ``exist`` is
# the absolute-truth wrapper (ExistLayer); ``queryPart`` / ``queryEqual`` are the
# interrogative dispatch targets of isPart / isEqual; ``part`` / ``equal`` are
# the bare grammar query ops.
_SURFACE_TO_KIND = {
    "exist": KIND_IS_TRUE, "isTrue": KIND_IS_TRUE, "true": KIND_IS_TRUE,
    "isPart": KIND_IS_PART, "queryPart": KIND_IS_PART, "part": KIND_IS_PART,
    "isEqual": KIND_IS_EQUAL, "queryEqual": KIND_IS_EQUAL, "equal": KIND_IS_EQUAL,
}

# Posture of an answer.
TRUE = "TRUE"
FALSE = "FALSE"
UNKNOWN = "UNKNOWN"
BOTH = "BOTH"


@dataclass
class QuerySpec:
    """A query reduced to one of the two atomic predicates over ideas.

    ``left`` / ``right`` are idea vectors (the operands). ``variables`` names
    any open slot for a binding query (e.g. ``("left",)`` for "what is part of
    B?"). ``desired_polarity`` is the polarity asked about (False asks "is A NOT
    part of B?").
    """

    predicate: str                       # KIND_IS_TRUE | KIND_IS_PART | KIND_IS_EQUAL
    left: Any = None
    right: Any = None
    variables: tuple = ()
    desired_polarity: bool = True

    @classmethod
    def from_surface(cls, name, left=None, right=None, *,
                     variables=(), polarity=True):
        """Build a QuerySpec from a grammar/query surface name, normalizing
        ``exist``→isTrue, ``queryPart``/``part``→isPart, ``queryEqual``/
        ``equal``→isEqual. Raises ValueError on an unknown name."""
        kind = _SURFACE_TO_KIND.get(str(name))
        if kind is None:
            raise ValueError(
                f"QuerySpec.from_surface: unknown query surface '{name}' "
                f"(known: {sorted(set(_SURFACE_TO_KIND))})")
        return cls(kind, left, right, tuple(variables), bool(polarity))

    @property
    def is_open(self) -> bool:
        """True iff the query has an unbound variable (a binding query)."""
        return bool(self.variables)


# -- helpers -----------------------------------------------------------------

def _as_vec(x) -> torch.Tensor:
    """Coerce an idea operand to a finite 1-D float tensor."""
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32)
    return torch.nan_to_num(x.reshape(-1).float())


# The parthood relation kind. The reasoner climbs REL_PARTOF rows (the tag is
# ignored for the untagged RelativeTruthStore, which yields all rows). The row
# iteration + the climb primitive (wholes / parts / _chain_to_target) are the
# CANONICAL ones on ConceptualSpace -- shared with ConceptualSpace.reason, not
# re-implemented here.
_REL_PARTOF = TernaryTruthStore.REL_PARTOF


# -- the reasoner ------------------------------------------------------------

class TruthGroundedReasoner:
    """The exact (non-differentiable) truth tools = the grammar query ops, plus
    the ``isTrue`` / ``isPart`` reduction API built on them.

    ``theta`` is the parthood coverage bar (a hop is real iff coverage ≥ theta);
    ``tau_id`` is the idea-identity bar for matching a query operand to a stored
    row; ``trust_threshold`` is the bar a scalar trust must clear to count as
    positive support; ``materialize_floor`` is the bar a derived conclusion must
    clear to be written back (guards lemma drift; cf. TruthLayer.compact).
    """

    def __init__(self, model=None, *, store=None,
                 theta: float = 0.7, tau_id: float = 0.7,
                 trust_threshold: float = 0.0,
                 materialize_floor: float = 0.5):
        self.model = model
        self._store = store
        self.theta = float(theta)
        self.tau_id = float(tau_id)
        self.trust_threshold = float(trust_threshold)
        self.materialize_floor = float(materialize_floor)

    def reasoning_store(self):
        """The reasoning corpus: an explicit ``store=`` wins; else the model's
        ``_reasoning_store()`` (unified TernaryTruthStore under
        <ltmConsolidation>, else RelativeTruthStore); else None."""
        if self._store is not None:
            return self._store
        cs = getattr(self.model, "conceptualSpace", None)
        if cs is not None and hasattr(cs, "_reasoning_store"):
            try:
                return cs._reasoning_store()
            except Exception:
                return None
        return None

    # == grammar query ops (the hard tools) ==============================

    @staticmethod
    def part(x, y) -> float:
        """``part(X, Y)``: graded parthood ``X ⊆ Y`` in [0,1] (the fraction of
        X's signed energy Y also carries; ConceptualSpace._idea_parthood)."""
        return float(ConceptualSpace._idea_parthood(_as_vec(x), _as_vec(y)))

    @staticmethod
    def equal(x, y, *, isomorphic: bool = True) -> float:
        """``equal(X, Y)``. ``isomorphic=True`` (default): the fraction of
        shared parts & wholes in [0,1] (1 = identical; ConceptualSpace._idea_
        identity). ``isomorphic=False``: the L2 norm of the difference in
        [0,∞) (0 = identical)."""
        if isomorphic:
            return float(ConceptualSpace._idea_identity(_as_vec(x), _as_vec(y)))
        a = _as_vec(x)
        b = _as_vec(y)
        n = min(a.numel(), b.numel())
        return float(torch.linalg.vector_norm(a[:n] - b[:n]))

    def exist(self, X) -> float:
        """``exist(X)``: the isTrue leaf -- signed Degree-of-Truth in [-1,1]
        (>0 true, <0 false, 0 unknown). X is true if it is a single idea with
        positive trust (a REL_NONE row of the unified store whose identity to X
        clears ``tau_id``, trust > 0), OR an ultimate truth (the model's
        absolute TruthLayer grounds it with positive DoT). No chaining."""
        X = _as_vec(X)
        best = 0.0
        store = self.reasoning_store()
        if store is not None and hasattr(store, "ideas") and hasattr(store, "row"):
            idxs = store.ideas()
            idxs = idxs.tolist() if hasattr(idxs, "tolist") else list(idxs)
            for i in idxs:
                row = store.row(int(i))
                if self.equal(X, row["np1"]) >= self.tau_id:
                    t = float(row["trust"])
                    if abs(t) > abs(best):
                        best = t
        if best > self.trust_threshold:
            return float(best)
        m = self.model
        if m is not None and hasattr(m, "isTrue"):
            try:
                dot = float(m.isTrue(X))
            except Exception:
                dot = 0.0
            if dot != 0.0:
                return dot
        return float(best) if best < 0 else 0.0

    def wholes(self, X) -> list:
        """``wholes(X)``: the proximal containing wholes of X -- the canonical
        ConceptualSpace.wholes over the REL_PARTOF rows. Returns
        ``[{idea, trust, row}, ...]``."""
        return ConceptualSpace.wholes(
            _as_vec(X), self.reasoning_store(), theta=self.theta,
            trust_threshold=self.trust_threshold, rel_type=_REL_PARTOF)

    def parts(self, X) -> list:
        """``parts(X)``: the proximal contained parts of X -- the canonical
        ConceptualSpace.parts (inverse of ``wholes``). Returns
        ``[{idea, trust, row}, ...]``."""
        return ConceptualSpace.parts(
            _as_vec(X), self.reasoning_store(), theta=self.theta,
            trust_threshold=self.trust_threshold, rel_type=_REL_PARTOF)

    def query(self, X, Y=None) -> Optional[dict]:
        """``query(X[, Y])``: an LTM lookup. ``query(X)`` returns the best
        matching stored ABSOLUTE idea ``{idea, trust, row, kind:'idea'}``;
        ``query(X, Y)`` returns the best matching stored relation
        ``{np1, np2, trust, row, kind:'relation'}`` (joint identity to X, Y).
        None when nothing matches.

        This is the HARD retrieval. The soft, ``.where``-typed read over the
        full truth-space (input / codebooks / STM / LTM via GlobalAttention)
        that the intervening-idea generator conditions on is Phase 3.
        """
        store = self.reasoning_store()
        if store is None:
            return None
        X = _as_vec(X)
        if Y is None:
            best = None
            if hasattr(store, "ideas") and hasattr(store, "row"):
                idxs = store.ideas()
                idxs = idxs.tolist() if hasattr(idxs, "tolist") else list(idxs)
                for i in idxs:
                    row = store.row(int(i))
                    s = self.equal(X, row["np1"])
                    if best is None or s > best["match"]:
                        best = {"idea": row["np1"], "trust": float(row["trust"]),
                                "row": int(i), "kind": "idea", "match": s}
            return best
        Y = _as_vec(Y)
        best = None
        for (idx, np1, vp, np2, t1) in ConceptualSpace._iter_relation_rows(
                store, _REL_PARTOF):
            s = min(self.equal(X, np1), self.equal(Y, np2))
            if best is None or s > best["match"]:
                best = {"np1": np1, "np2": np2, "trust": float(t1),
                        "row": int(idx), "kind": "relation", "match": s}
        return best

    def quantize(self, X):
        """``quantize(X)``: snap X onto the nearest real idea -- the best
        matching stored ABSOLUTE idea by ``equal`` (the grounding step that
        keeps a proposed bridge on the manifold of known ideas). Returns the
        snapped idea vector, or X unchanged when no store / no idea is
        reachable. (A model codebook is the richer basis; Phase 3.)"""
        hit = self.query(X)
        return hit["idea"] if hit is not None else _as_vec(X)

    # == soft .where-read over the FULL truth-space (Phase 3) ============

    @staticmethod
    def _valid_space(s):
        k = s.get("keys")
        return (k is not None and torch.is_tensor(k) and k.dim() in (2, 3)
                and int(k.shape[-2]) > 0)

    @staticmethod
    def _topk_candidates(spaces, obs, top_k, b=0):
        """Reconstruct GlobalAttention's concatenated key layout (same usable-
        space order) and return the top-``top_k`` attended keys as candidate
        ideas: ``[{idea, space, alpha}]`` -- each a REAL stored key, so the
        proposal stays on the manifold of known ideas."""
        Dc = int(obs["content"].shape[-1])
        alpha = obs["alpha"][b].detach()
        cands = []
        off = 0
        for s in spaces:
            if not TruthGroundedReasoner._valid_space(s):
                continue
            keys = s["keys"]
            shared = keys.dim() == 2
            M = int(keys.shape[0] if shared else keys.shape[1])
            for m in range(M):
                key = (keys[m] if shared else keys[b, m])[:Dc]
                cands.append({"idea": key.detach(), "space": int(s["id"]),
                              "alpha": float(alpha[off + m])})
            off += M
        cands.sort(key=lambda c: -c["alpha"])
        return cands[:int(top_k)]

    @staticmethod
    def where_read(concept_q, spaces, *, ga, symbol_q=None,
                   temperature=0.0, top_k=8):
        """The soft ``.where``-typed read over the full truth-space (input /
        codebooks / STM / LTM) the intervening-idea generator conditions on
        (§4.3a). Runs ``ga`` (a GlobalAttention) over the typed ``spaces`` and
        returns ``{idea, where, space_id, candidates, alpha}`` where ``idea`` =
        the soft-read ``Σ αₖ·keyₖ`` (a blend of REAL keys, grounded by
        construction) and ``space_id`` is the typed provenance. Gradient flows
        through ``α`` (and any query head) only -- the keys are detached, so the
        recalled ideas are never softened. None when no space has candidates."""
        cq = _as_vec(concept_q).unsqueeze(0)
        sq = None if symbol_q is None else _as_vec(symbol_q).unsqueeze(0)
        obs = ga(concept_q=cq, symbol_q=sq, spaces=spaces,
                 temperature=temperature)
        if obs is None:
            return None
        return {"idea": obs["content"][0], "where": obs["where"][0],
                "space_id": int(obs["space_id"][0]),
                "candidates": TruthGroundedReasoner._topk_candidates(
                    spaces, obs, top_k, b=0),
                "alpha": obs["alpha"][0]}

    # == reduction API (isTrue / isPart over the tools) =================

    def is_true(self, A) -> float:
        """``isTrue(A)`` -- alias of the ``exist`` leaf tool."""
        return self.exist(A)

    def is_part_direct(self, A, B) -> Optional[tuple]:
        """Direct parthood ``A ⊑ B`` without a chain: ``(score, how)`` in [0,1]
        or None. ``how`` is "geometric" (``part(A,B) ≥ theta``) or "stored" (a
        positive-trust REL_PARTOF row whose endpoints match A, B by ``equal``).
        """
        A = _as_vec(A)
        B = _as_vec(B)
        p = self.part(A, B)
        if p >= self.theta:
            return (float(p), "geometric")
        best = None
        for (idx, np1, vp, np2, t1) in ConceptualSpace._iter_relation_rows(
                self.reasoning_store(), _REL_PARTOF):
            if (t1 > self.trust_threshold
                    and self.equal(A, np1) >= self.tau_id
                    and self.equal(B, np2) >= self.tau_id):
                if best is None or t1 > best:
                    best = t1
        if best is not None:
            return (float(best), "stored")
        return None

    def is_part(self, A, B, *, max_steps: int = 8, beam: int = 8,
                materialize: bool = False) -> list:
        """Candidate chains supporting ``A ⊑ B``, ranked by score, never a bare
        boolean.

        Direct candidate first (§4.2), then a beam-limited climb via
        ``wholes()`` (§4.3): from the frontier concept fire each proximal whole,
        and succeed when a whole reaches ``B`` (``part(whole, B) ≥ theta``). A
        chain's score is the MIN over its hop trusts -- only as true as its
        weakest intervening idea. The frontier is pruned to the top-``beam``
        running scores each step; ``max_steps`` bounds depth; each row fires at
        most once per chain. When ``materialize`` is set, a verified chain's
        conclusion is written back as a direct edge (§4.4).
        """
        A = _as_vec(A)
        B = _as_vec(B)
        results = []
        d = self.is_part_direct(A, B)
        if d is not None:
            score, how = d
            results.append({"score": float(score), "how": how, "chain": [],
                            "trust": float(score), "steps": 0})
        # The single canonical chain search (shared with ConceptualSpace.reason):
        # a beam-limited MIN-trust climb via wholes() toward B.
        for c in ConceptualSpace._chain_to_target(
                A, B, self.reasoning_store(), parthood_threshold=self.theta,
                max_steps=int(max_steps), beam=int(beam), trust_combine="min",
                rel_type=_REL_PARTOF, trust_threshold=self.trust_threshold):
            results.append({"score": c["score"], "how": "chain",
                            "chain": c["chain"], "trust": c["trust"],
                            "steps": c["steps"]})
        results.sort(key=lambda r: -r["score"])
        results = results[:int(beam)]
        if materialize and results and results[0]["how"] == "chain":
            row = self.materialize(A, B, results[0]["score"])
            if row >= 0:
                results[0]["materialized"] = row
        return results

    def materialize(self, A, B, score, *, store=None) -> int:
        """Write a verified ``isPart(A, B)`` conclusion back as a REL_PARTOF
        lemma carrying the chain's MIN-composed ``score`` as trust, so a later
        identical query is a DIRECT hit and future ``wholes()`` reach it (§4.4).
        No-op (returns -1) below ``materialize_floor`` or with no writable store.
        """
        store = store if store is not None else self.reasoning_store()
        if store is None or float(score) < self.materialize_floor:
            return -1
        if self._creates_cycle(A, B):
            return -1
        A = _as_vec(A)
        B = _as_vec(B)
        if isinstance(store, TernaryTruthStore):
            return int(store.append_relation(
                A, torch.zeros_like(A), B,
                rel_type=store.REL_PARTOF, trust=float(score)))
        if hasattr(store, "record_triple"):
            return int(store.record_triple(
                A, torch.zeros_like(A), B, degree=float(score)))
        return -1

    def _creates_cycle(self, A, B) -> bool:
        """True iff writing ``A ⊑ B`` would violate the parthood partial order
        (antisymmetry): A and B are already the same idea, or B already reaches
        A by parthood (``B ⊑* A``). Enforced at edge insertion so the climb
        cannot loop (Phase 6); read-only."""
        if self.equal(A, B) >= self.tau_id:
            return True
        return bool(self.is_part(B, A))      # any direct/chain B -> A

    def _refuting_direct(self, A, B) -> float:
        """Best refuting evidence for ``isPart(A, B)`` in [0,1]: a stored
        REL_PARTOF (A→B) row asserted with NEGATIVE trust (¬isPart). 0 when
        none. (Chain-based refutation is Phase 5.)"""
        A = _as_vec(A)
        B = _as_vec(B)
        best = 0.0
        for (idx, np1, vp, np2, t1) in ConceptualSpace._iter_relation_rows(
                self.reasoning_store(), _REL_PARTOF):
            if (t1 < -self.trust_threshold
                    and self.equal(A, np1) >= self.tau_id
                    and self.equal(B, np2) >= self.tau_id):
                if -t1 > best:
                    best = -t1
        return float(best)

    # == trace + posture =================================================

    @staticmethod
    def render_chain(candidate: dict) -> str:
        """A one-line explanation of a candidate chain / direct hit."""
        how = candidate.get("how")
        if how in ("geometric", "stored"):
            return f"isPart direct ({how}), trust {candidate['score']:.2f}"
        hops = candidate.get("chain", [])
        steps = " → ".join(f"row{idx}(t={t:.2f})" for idx, t in hops)
        return (f"isPart via {steps} ⇒ trust {candidate['score']:.2f} "
                f"(min hop), {candidate.get('steps', len(hops))} steps")

    def _posture(self, support_true: float, support_false: float,
                 tau: float) -> dict:
        st = float(support_true)
        sf = float(support_false)
        if st >= tau and sf >= tau:
            posture = BOTH
        elif st >= tau:
            posture = TRUE
        elif sf >= tau:
            posture = FALSE
        else:
            posture = UNKNOWN
        return {"posture": posture, "confidence": max(st, sf),
                "support_true": st, "support_false": sf}

    def evaluate(self, q: QuerySpec, *, tau: float = None,
                 max_steps: int = 8, beam: int = 8) -> dict:
        """Evaluate a QuerySpec to a posture + confidence + candidate chains.

        isTrue uses the signed DoT from ``exist`` (sign splits support); isPart
        uses the best candidate-chain score as positive support and a refuting
        direct edge as negative support; isEqual uses ``equal(isomorphic=True)``
        (the fraction of shared parts & wholes).
        """
        tau = self.theta if tau is None else float(tau)
        if q.predicate == KIND_IS_TRUE:
            dot = self.exist(q.left)
            res = self._posture(max(0.0, dot), max(0.0, -dot), tau)
            res["kind"] = KIND_IS_TRUE
            res["candidates"] = []
            res["trace"] = None
            return res
        if q.predicate == KIND_IS_EQUAL:
            score = self.equal(q.left, q.right, isomorphic=True)
            res = self._posture(score, 0.0, tau)
            res["kind"] = KIND_IS_EQUAL
            res["candidates"] = []
            res["trace"] = f"equal (shared parts & wholes) = {score:.2f}"
            return res
        # KIND_IS_PART
        cands = self.is_part(q.left, q.right, max_steps=max_steps, beam=beam)
        score = cands[0]["score"] if cands else 0.0
        refute = self._refuting_direct(q.left, q.right)
        res = self._posture(score, refute, tau)
        res["kind"] = KIND_IS_PART
        res["candidates"] = cands
        res["trace"] = self.render_chain(cands[0]) if cands else None
        return res


# -- The intervening-idea generator (Phase 3, the SOFT policy) ----------------

def _fit_dim(v, d):
    """Pad/truncate a 1-D tensor to width d."""
    v = _as_vec(v)
    if v.numel() == d:
        return v
    out = v.new_zeros(d)
    k = min(d, v.numel())
    out[:k] = v[:k]
    return out


class InterveningIdeaGenerator(nn.Module):
    """Creates a candidate intervening idea ``M`` to bridge ``A → B`` (§4.3a).

    The guess is SOFT: an MLP query head over ``[A ; B ; prev_r]`` (the full
    solution space + the previous queried idea) produces the ``concept_q`` that
    GlobalAttention (:meth:`TruthGroundedReasoner.where_read`) indexes the
    truth-space with. The soft-read content IS ``M`` -- a blend of REAL stored
    ideas, grounded on the manifold by construction; the top-``k`` ``α`` are the
    candidate beam. Gradient flows through the head + ``α`` only (keys
    detached), so the deduction stays hard. ``M`` recurs as ``prev_r`` to the
    next reasoning iteration.

    Untrained here; the Phase-5 answer loss (:func:`answer_loss`) shapes which
    guesses lead to verifiable chains. Gated dark by the consumer.
    """

    def __init__(self, dim, hidden=None):
        super().__init__()
        self.dim = int(dim)
        h = int(hidden) if hidden else max(8, self.dim)
        self.head = nn.Sequential(
            nn.Linear(3 * self.dim, h), nn.ReLU(), nn.Linear(h, self.dim))

    def concept_q(self, A, B, prev_r=None):
        """The learned query: ``MLP([A ; B ; prev_r])`` at full width."""
        r = (torch.zeros(self.dim) if prev_r is None else _fit_dim(prev_r,
                                                                   self.dim))
        x = torch.cat([_fit_dim(A, self.dim), _fit_dim(B, self.dim), r])
        return self.head(x)

    def propose(self, A, B, spaces, *, ga, prev_r=None, top_k=8,
                temperature=0.0):
        """Propose ``M`` and the candidate beam by grounding the learned query
        in the truth-space. Returns the :meth:`where_read` dict (``idea`` = M;
        recur ``prev_r=result['idea']``), or None when no space has candidates.
        """
        q = self.concept_q(A, B, prev_r)
        return TruthGroundedReasoner.where_read(
            q, spaces, ga=ga, top_k=top_k, temperature=temperature)


# -- Answer (policy) loss (Phase 5) ------------------------------------------

def proof_score(signed_trust):
    """Map a signed trust / DoT in [-1,1] to a [0,1] proof-success score via the
    documented monotonic map ``(t+1)/2`` -- so a negative (refuting) score does
    NOT silently read as 'unknown' (0.5 = unknown, 0 = false, 1 = true)."""
    if torch.is_tensor(signed_trust):
        return ((signed_trust.float() + 1.0) / 2.0).clamp(0.0, 1.0)
    return max(0.0, min(1.0, (float(signed_trust) + 1.0) / 2.0))


def answer_loss(predicted_signed, gold_label):
    """The differentiable policy signal (Phase 5): NLL on the [0,1] proof score
    against a gold boolean (1 = true, 0 = false). Trains the SOFT route that
    produced ``predicted_signed`` (the generator's query head + the attention
    α); the hard deduction is never differentiated. ``predicted_signed`` should
    be a tensor that carries gradient from the policy."""
    p = predicted_signed if torch.is_tensor(predicted_signed) else \
        torch.as_tensor(float(predicted_signed))
    p01 = proof_score(p.float()).clamp(1e-6, 1.0 - 1e-6)
    y = torch.as_tensor(float(gold_label))
    return -(y * torch.log(p01) + (1.0 - y) * torch.log(1.0 - p01))
