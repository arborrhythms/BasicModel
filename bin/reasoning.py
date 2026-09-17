"""Truth-grounded reasoning tools and differentiable policy losses.

The hard tools reduce query ops to truth/parthood/equality checks over stored
ideas; the soft policy ranks candidate intervening ideas. See ``doc/Reasoning.md``.
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

from Spaces import ConceptualSpace, GlobalAttention
from Layers import TernaryTruthStore
from Meaning import ConceptualMeaning
from Taxonomy import capture_taxonomy, concept_reference


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
    "isWhole": KIND_IS_PART, "whole": KIND_IS_PART, "PartOf": KIND_IS_PART,
    "isEqual": KIND_IS_EQUAL, "queryEqual": KIND_IS_EQUAL, "equal": KIND_IS_EQUAL,
}

# Posture of an answer.
TRUE = "TRUE"
FALSE = "FALSE"
UNKNOWN = "UNKNOWN"
BOTH = "BOTH"


@dataclass
class QuerySpec:
    """Legacy query interface; Exist accepts one complete ConceptualMeaning.

    The checked grammatical-VP registry is a separate migration. This wrapper
    never flattens a structured description into an apparent unary operand.
    """

    predicate: str                       # KIND_IS_TRUE | KIND_IS_PART | KIND_IS_EQUAL
    left: Any = None
    right: Any = None
    variables: tuple = ()
    desired_polarity: bool = True
    domain: Optional[str] = None

    @classmethod
    def from_surface(cls, name, left=None, right=None, *,
                     variables=(), polarity=True, domain=None):
        """Build a QuerySpec from a grammar/query surface name, normalizing
        ``exist``→isTrue, ``queryPart``/``part``→isPart, ``queryEqual``/
        ``equal``→isEqual. Raises ValueError on an unknown name."""
        kind = _SURFACE_TO_KIND.get(str(name))
        if kind is None:
            raise ValueError(
                f"QuerySpec.from_surface: unknown query surface '{name}' "
                f"(known: {sorted(set(_SURFACE_TO_KIND))})")
        if name in ("isWhole", "whole"):
            left, right = right, left
        if kind == KIND_IS_PART:
            domain = "conceptual-taxonomy" if domain is None else domain
            if domain != "conceptual-taxonomy":
                raise ValueError(f"PartOf does not support relation domain {domain!r}")
        return cls(kind, left, right, tuple(variables), bool(polarity), domain)

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
    """Fact existence and conceptual-taxonomy query evidence.

    The older geometric/world-row helpers below remain explicit compatibility
    utilities. Public query evaluation does not use those helpers as PartOf
    evidence; typed concept references address the conceptual taxonomy.
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
    def legacy_part(x, y) -> float:
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

    def existence_evidence(self, description):
        """Ground the full description in accepted LTM facts, without chaining.

        Match each occupied role at the same width and preserve bindings,
        semantic scope and referenced constituents. Signed identity attenuates
        the stored degree; the least-matching role bounds that degree. Keep
        positive and negative evidence independently and never sum repeated
        evidence. Concept activation is not evidence that a referent exists.
        This is a hard lookup, with no derivative through its matching choices.
        """
        requested = ConceptualMeaning.from_description(description)
        candidates, incomplete = [], []
        support_true = support_false = 0.0
        store = self.reasoning_store()
        if isinstance(store, TernaryTruthStore):
            indexes = (store.record_kind[:len(store)] == store.KINDS.index("fact")).nonzero(as_tuple=True)[0]
            for i in indexes.tolist():
                row = store.row(i)
                fact = row["meaning"]
                if fact is None or not row["metadata_complete"]:
                    incomplete.append(row["occurrence"])
                    continue
                if fact.mode != "assertive" or fact.roles.shape != requested.roles.shape:
                    continue
                if not torch.equal(fact.role_mask.cpu(), requested.role_mask.cpu()):
                    continue
                if (fact.bindings != requested.bindings or fact.scope != requested.scope
                        or fact.role_refs != requested.role_refs):
                    continue
                strengths = [self.equal(requested.roles[role].to(fact.roles), fact.roles[role])
                             for role in requested.role_mask.nonzero(as_tuple=True)[0].tolist()]
                match = min(strengths)
                if match < self.tau_id:
                    continue
                signed = float(row["trust"]) * match
                if fact.polarity != requested.polarity:
                    signed = -signed
                support_true = max(support_true, signed)
                support_false = max(support_false, -signed)
                candidates.append({"row": i, "occurrence": row["occurrence"],
                                   "origin": row["origin"], "text": row["text"],
                                   "kind": "fact", "match": match,
                                   "signed_support": signed, "trust": float(row["trust"]),
                                   "bindings": fact.bindings, "scope": fact.scope,
                                   "role_refs": fact.role_refs})
        return {"support_true": support_true, "support_false": support_false,
                "candidates": candidates, "incomplete_evidence": incomplete,
                "meaning": requested}

    def exist(self, X) -> float:
        """Lossy legacy scalar view: positive minus negative fact support.

        Checked evaluation and the thinking kernel consume existence_evidence
        instead, since a scalar cannot preserve contradictory support.
        """
        evidence = self.existence_evidence(X)
        return evidence["support_true"] - evidence["support_false"]

    def legacy_wholes(self, X) -> list:
        """``wholes(X)``: the proximal containing wholes of X -- the canonical
        ConceptualSpace.wholes over the REL_PARTOF rows. Returns
        ``[{idea, trust, row}, ...]``."""
        return ConceptualSpace.wholes(
            _as_vec(X), self.reasoning_store(), theta=self.theta,
            trust_threshold=self.trust_threshold, rel_type=_REL_PARTOF)

    def legacy_parts(self, X) -> list:
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

    def arma(self, X=None):
        """``arma(X)``: the ARMA next-step prediction in conceptual space -- the
        ``InterSentenceLayer``'s predicted next idea (the statistical discourse
        trajectory). ``X`` is the current trajectory point (nominal; the ARMA
        reads its OWN observed end-state chain, the autoregressive history).
        Returns the predicted next-idea vector, or ``None`` when no warm
        discourse predictor is configured (no model / no ``_inter_predictor`` /
        a cold AR ring). A tool the reasoner can fold into a chain alongside the
        hard deduction -- the policy learns when the trajectory momentum, vs
        truth-space retrieval/deduction, is the relevant signal for the next
        idea (this is the soft/hard split applied to next-sentence prediction)."""
        m = self.model
        disc = (getattr(getattr(m, "symbolSpace", None), "discourse", None)
                if m is not None else None)
        if disc is None or getattr(disc, "_inter_predictor", None) is None:
            return None
        if not (hasattr(disc, "predict_next_end_state")
                and hasattr(disc, "get_stm_chain") and disc.get_stm_chain(n=1)):
            return None                    # cold AR ring -> no real prediction
        try:
            shape = disc.predict_next_end_state()
        except Exception:
            return None
        if shape is None:
            return None
        _depth, payload = shape
        if (payload is None or not torch.is_tensor(payload)
                or payload.numel() == 0 or not torch.isfinite(payload).all()):
            return None
        return payload.reshape(-1, int(payload.shape[-1]))[0]   # predicted root idea

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

    def taxonomy_evidence(self, part, whole, *, max_steps=8,
                          max_nodes=256, max_records=1024, max_expansions=1024):
        """Read bounded structural inclusion, with native record provenance.

        Legacy vector-only operands have no concept handle and remain unknown.
        Their geometry is never used to guess a referent. Checked grammatical
        callers supply typed references; the VP registry is a later adapter.
        """
        try:
            part, whole = concept_reference(part), concept_reference(whole)
        except TypeError:
            return {"domain": "conceptual-taxonomy", "support_true": 0.0,
                    "support_false": 0.0, "path": (),
                    "incomplete": ("unbound_concept_reference",),
                    "nodes_scanned": 0, "records_scanned": 0, "edges_expanded": 0}
        cs = getattr(self.model, "conceptualSpace", None)
        view = capture_taxonomy(cs, max_nodes=max_nodes, max_records=max_records,
                                focus=(part, whole))
        return view.part_of(part, whole, max_steps=max_steps,
                            max_expansions=max_expansions)

    def taxonomy_neighbors(self, reference, *, direction="up",
                           max_nodes=256, max_records=1024):
        """Return grounded reference neighbors, never numeric concept-id features."""
        if direction not in ("up", "down"):
            raise ValueError("taxonomy direction must be up or down")
        try:
            reference = concept_reference(reference)
        except TypeError:
            return []
        view = capture_taxonomy(getattr(self.model, "conceptualSpace", None),
                                max_nodes=max_nodes, max_records=max_records,
                                focus=(reference,))
        result = []
        for edge in view.neighbors(reference, direction=direction):
            target = edge.whole if direction == "up" else edge.part
            result.append({"reference": target, "idea": target, "trust": 1.0,
                           "domain": "conceptual-taxonomy", "source": edge,
                           "source_key": (edge.owner, edge.role, edge.part, edge.whole),
                           "incomplete": view.incomplete,
                           "nodes_scanned": view.nodes_scanned,
                           "records_scanned": view.records_scanned})
        return result

    def part(self, part, whole):
        """Compatibility scalar view of direct conceptual inclusion."""
        return self.taxonomy_evidence(part, whole, max_steps=1)["support_true"]

    def is_part_direct(self, part, whole):
        """One native taxonomy link; no vector/world-row fallback."""
        evidence = self.taxonomy_evidence(part, whole, max_steps=1)
        return (evidence["support_true"], "taxonomy") if evidence["support_true"] else None

    def is_part(self, part, whole, *, max_steps=8, beam=8, materialize=False):
        """Bounded taxonomy candidates. The legacy write flag grants no world fact."""
        return self.evaluate(QuerySpec.from_surface("isPart", part, whole),
                             max_steps=max_steps, beam=beam)["candidates"]

    def wholes(self, reference):
        """Proximal conceptual wholes as typed references and record sources."""
        return self.taxonomy_neighbors(reference, direction="up")

    def parts(self, reference):
        """Proximal conceptual parts as typed references and record sources."""
        return self.taxonomy_neighbors(reference, direction="down")

    def materialize(self, *args, **kwargs):
        """Query evidence cannot edit conceptual definitions or assert world facts."""
        raise ValueError("PartOf is read-only; materialization is not a supported query effect")

    def legacy_is_part_direct(self, A, B) -> Optional[tuple]:
        """Direct parthood ``A ⊑ B`` without a chain: ``(score, how)`` in [0,1]
        or None. ``how`` is "geometric" (``part(A,B) ≥ theta``) or "stored" (a
        positive-trust REL_PARTOF row whose endpoints match A, B by ``equal``).
        """
        A = _as_vec(A)
        B = _as_vec(B)
        p = self.legacy_part(A, B)
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

    def legacy_is_part(self, A, B, *, max_steps: int = 8, beam: int = 8,
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
        d = self.legacy_is_part_direct(A, B)
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
            row = self.legacy_materialize(A, B, results[0]["score"])
            if row >= 0:
                results[0]["materialized"] = row
        return results

    def legacy_materialize(self, A, B, score, *, store=None) -> int:
        """Write a verified ``isPart(A, B)`` conclusion back as a REL_PARTOF
        lemma carrying the chain's MIN-composed ``score`` as trust, so a later
        identical query is a DIRECT hit and future ``wholes()`` reach it (§4.4).
        No-op (returns -1) below ``materialize_floor`` or with no writable store.
        """
        store = store if store is not None else self.reasoning_store()
        if store is None or float(score) < self.materialize_floor:
            return -1
        if self._legacy_creates_cycle(A, B):
            return -1
        A = _as_vec(A)
        B = _as_vec(B)
        if isinstance(store, TernaryTruthStore):
            # This legacy adapter has no grammatical VP. Its relation tag may
            # serve legacy Part readers, but cannot certify a full Exist fact.
            return int(store.append_relation(
                A, torch.zeros_like(A), B,
                rel_type=store.REL_PARTOF, trust=float(score), kind="unverified"))
        if hasattr(store, "record_triple"):
            return int(store.record_triple(
                A, torch.zeros_like(A), B, degree=float(score)))
        return -1

    def _legacy_creates_cycle(self, A, B) -> bool:
        """True iff writing ``A ⊑ B`` would violate the parthood partial order
        (antisymmetry): A and B are already the same idea, or B already reaches
        A by parthood (``B ⊑* A``). Enforced at edge insertion so the climb
        cannot loop (Phase 6); read-only."""
        if self.equal(A, B) >= self.tau_id:
            return True
        return bool(self.legacy_is_part(B, A))      # any direct/chain B -> A

    def _legacy_refuting_direct(self, A, B) -> float:
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
        positive = st > 0.0 and st >= tau
        negative = sf > 0.0 and sf >= tau
        if positive and negative:
            posture = BOTH
        elif positive:
            posture = TRUE
        elif negative:
            posture = FALSE
        else:
            posture = UNKNOWN
        return {"posture": posture, "confidence": max(st, sf),
                "support_true": st, "support_false": sf}

    def evaluate(self, q: QuerySpec, *, tau: float = None,
                 max_steps: int = 8, beam: int = 8) -> dict:
        """Evaluate a QuerySpec to a posture + confidence + candidate chains.

        isTrue preserves independent positive/negative LTM fact support.
        PartOf reads only typed conceptual-taxonomy references; a known path
        supports the positive proposition or refutes its negation. A missing path is
        unknown. isEqual retains the legacy geometric identity adapter until
        its checked grammatical registry entry is wired.
        """
        tau = self.theta if tau is None else float(tau)
        if q.predicate == KIND_IS_TRUE:
            evidence = self.existence_evidence(q.left)
            res = self._posture(evidence["support_true"], evidence["support_false"], tau)
            res.update(evidence)
            res["kind"] = KIND_IS_TRUE
            res["trace"] = (f"Exist: {len(evidence['candidates'])} matching facts; "
                            f"support +{res['support_true']:.3f}/-{res['support_false']:.3f}")
            return res
        if q.predicate == KIND_IS_EQUAL:
            score = self.equal(q.left, q.right, isomorphic=True)
            res = self._posture(score, 0.0, tau)
            res["kind"] = KIND_IS_EQUAL
            res["candidates"] = []
            res["trace"] = f"equal (shared parts & wholes) = {score:.2f}"
            return res
        if q.predicate != KIND_IS_PART:
            raise ValueError(f"unknown query predicate {q.predicate!r}")
        if q.domain not in (None, "conceptual-taxonomy"):
            raise ValueError(f"PartOf does not support relation domain {q.domain!r}")
        evidence = self.taxonomy_evidence(q.left, q.right, max_steps=max_steps,
                                          max_expansions=max(0, int(beam)) * max(0, int(max_steps)))
        if not q.desired_polarity:
            evidence = dict(evidence, support_true=evidence["support_false"],
                             support_false=evidence["support_true"])
        res = self._posture(evidence["support_true"], evidence["support_false"], tau)
        res.update(evidence)
        res["kind"] = KIND_IS_PART
        strength = max(evidence["support_true"], evidence["support_false"])
        res["candidates"] = ([dict(evidence, score=strength,
                                   trust=strength, how="taxonomy",
                                   steps=len(evidence["path"]))]
                              if strength else [])
        res["trace"] = (f"PartOf conceptual taxonomy: {len(evidence['path'])} links; "
                        f"{evidence['edges_expanded']} expansions; "
                        f"support +{evidence['support_true']:.3f}/-{evidence['support_false']:.3f}")
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

class NextIdeaScorer(nn.Module):
    """Per-candidate logit head for the next-idea blend (Step 2): scores
    ``[q ; cand_detached] -> scalar``, ADDED to the cosine logit so a learned
    per-tool prior can ride the state_dict. Optional (cosine-only also works)."""

    def __init__(self, dim, hidden=None):
        super().__init__()
        self.dim = int(dim)
        h = int(hidden) if hidden else max(8, self.dim)
        self.head = nn.Sequential(
            nn.Linear(2 * self.dim, h), nn.ReLU(), nn.Linear(h, 1))

    def logit(self, q, cand_detached):
        x = torch.cat([_fit_dim(q, self.dim), _fit_dim(cand_detached, self.dim)])
        return self.head(x).reshape(())


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


# -- The NeuralToolUser: the recurrent policy over the hard tools (Phase B) ---

@dataclass
class ReasoningResult:
    """Structured result for one query-reasoning run."""

    posture: str
    confidence: float
    support_true: float
    support_false: float
    ideas: list
    chain: list
    trace: Optional[str] = None
    iterations: int = 0
    evidence: Optional[dict] = None


class NeuralToolUser:
    """Soft-propose / hard-verify policy around ``TruthGroundedReasoner``."""

    def __init__(self, reasoner, *, generator=None, ga=None, spaces=None,
                 iterations: int = 10, beam: int = 8, top_k: int = 8,
                 materialize: bool = False):
        self.reasoner = reasoner
        self.generator = generator
        self.ga = ga
        self.spaces = spaces
        self.iterations = int(iterations)
        self.beam = int(beam)
        self.top_k = int(top_k)
        self.materialize = bool(materialize)

    def run(self, q: QuerySpec, *, spaces=None) -> ReasoningResult:
        """Public query route; generated vectors cannot certify taxonomy links."""
        result = self.reasoner.evaluate(q, max_steps=max(0, self.iterations),
                                         beam=self.beam)
        return ReasoningResult(
            posture=result["posture"], confidence=result["confidence"],
            support_true=result["support_true"], support_false=result["support_false"],
            ideas=[], chain=result.get("candidates", []), trace=result.get("trace"),
            iterations=int(result.get("edges_expanded", 0)), evidence=result)

    def run_legacy_world(self, q: QuerySpec, *, spaces=None) -> ReasoningResult:
        """Explicit legacy vector/world-row experiment, outside public dispatch.

        Its geometric conclusions are not conceptual-taxonomy evidence. Kept
        for the older proposal/loss experiments; model query entry uses run().
        """
        spaces = self.spaces if spaces is None else spaces
        r = self.reasoner
        N = max(0, self.iterations)
        # Leaf judgments need no chain -- delegate to the exact evaluate().
        if q.predicate in (KIND_IS_TRUE, KIND_IS_EQUAL):
            res = r.evaluate(q, max_steps=max(1, N), beam=self.beam)
            return ReasoningResult(
                posture=res["posture"], confidence=res["confidence"],
                support_true=res["support_true"],
                support_false=res["support_false"],
                ideas=[], chain=res.get("candidates", []),
                trace=res.get("trace"), iterations=0)
        # isPart: the stored hard chain, AUGMENTED by the soft generator climb.
        A = _as_vec(q.left)
        B = _as_vec(q.right)
        chains = list(r.legacy_is_part(A, B, max_steps=max(1, N), beam=self.beam,
                                materialize=self.materialize))
        ideas, steps = self._generate_chain(A, B, spaces, N)
        for it in ideas:
            if it["verified"]:
                chains.append({"score": it["trust"], "how": "generated",
                               "chain": [], "trust": it["trust"],
                               "steps": it["step"] + 1, "idea": it["idea"]})
        chains.sort(key=lambda c: -c["score"])
        chains = chains[:self.beam]
        score = chains[0]["score"] if chains else 0.0
        refute = r._legacy_refuting_direct(A, B)
        post = r._posture(score, refute, r.theta)
        # The N ideas, ranked by subsymbolic relevance (the N-sentence material).
        ideas.sort(key=lambda i: -i["relevance"])
        ideas = ideas[:max(1, N)]
        trace = r.render_chain(chains[0]) if chains else None
        return ReasoningResult(
            posture=post["posture"], confidence=post["confidence"],
            support_true=post["support_true"],
            support_false=post["support_false"],
            ideas=ideas, chain=chains, trace=trace, iterations=steps)

    def _generate_chain(self, A, B, spaces, N):
        """The recurrent soft-propose / hard-verify loop. Returns
        ``(ideas, steps)``: ``ideas`` are the candidate intervening ideas
        surfaced across the steps (each ``{idea, relevance, space_id, cov_in,
        cov_out, verified, trust, step}``); ``steps`` is how many iterations ran
        (early-exit when a candidate bridges to ``B``). Inert (``[], 0``) when
        the soft half is absent."""
        ideas = []
        if (self.generator is None or self.ga is None or not spaces or N <= 0):
            return ideas, 0
        r = self.reasoner
        cur = A
        prev_r = None
        steps = 0
        for t in range(N):
            out = self.generator.propose(cur, B, spaces, ga=self.ga,
                                         prev_r=prev_r, top_k=self.top_k)
            if out is None:
                break
            steps = t + 1
            prev_r = out["idea"]                       # recur the soft read
            best_adv, best_adv_rel = None, -1.0
            reached = False
            for cand in out.get("candidates", []):
                M = cand["idea"]
                rel = float(cand.get("alpha", 0.0))
                in_ok = r.legacy_is_part_direct(cur, M)       # cur ⊑ M ?
                out_ok = r.legacy_is_part_direct(M, B)        # M ⊑ B ?
                verified = in_ok is not None and out_ok is not None
                trust = (min(in_ok[0], out_ok[0]) if verified else 0.0)
                ideas.append({
                    "idea": M, "relevance": rel,
                    "space_id": cand.get("space"),
                    "cov_in": None if in_ok is None else float(in_ok[0]),
                    "cov_out": None if out_ok is None else float(out_ok[0]),
                    "verified": verified, "trust": float(trust), "step": t})
                if verified:
                    if self.materialize:
                        r.legacy_materialize(A, B, trust)
                    reached = True
                elif in_ok is not None and rel > best_adv_rel:
                    best_adv, best_adv_rel = M, rel
            if reached:
                break
            if best_adv is not None:
                cur = best_adv                         # advance the frontier
        return ideas, steps

    def reason_predict_next(self, state_idea, *, spaces=None, scorer=None):
        """Blend ``{arma, retrieval, deduction}`` into ONE differentiable
        next-idea ``e_hat`` (Step 2). The candidate next-ideas are DETACHED
        (hard/grounded); the gradient rides ONLY the generator query head (+ the
        optional scorer) through the softmax blend weights -- the same in-graph
        cosine-softmax pattern as ``policy_answer_loss`` (NOT ``GlobalAttention``,
        which detaches its query). Tool order ``[arma, retrieval, deduction]``;
        an absent tool gets a ``-inf`` logit so its weight is exactly 0. Returns
        ``(e_hat [D], weights [C])``, or ``(None, None)`` when no candidate or no
        generator."""
        if self.generator is None:
            return None, None
        spaces = self.spaces if spaces is None else spaces
        r = self.reasoner
        D = int(self.generator.dim)
        s = _fit_dim(state_idea, D)
        zeros = torch.zeros(D, device=s.device, dtype=s.dtype)

        cands, present = [], []
        # (1) arma -- the statistical trajectory (a single [D] or None).
        av = r.arma(s)
        if av is not None and torch.is_tensor(av) and torch.isfinite(av).all():
            cands.append(_fit_dim(av, D).detach())
            present.append(True)
        else:
            cands.append(zeros)
            present.append(False)
        # (2) retrieval -- the soft .where read (GA already detaches keys+query).
        rv = None
        if self.ga is not None and spaces:
            out = r.where_read(self.generator.concept_q(s, zeros, None),
                               spaces, ga=self.ga, top_k=self.top_k)
            rv = None if out is None else out.get("idea")
        if rv is not None and torch.is_tensor(rv) and torch.isfinite(rv).all():
            cands.append(_fit_dim(rv, D).detach())
            present.append(True)
        else:
            cands.append(zeros)
            present.append(False)
        # (3) deduction -- the top containing whole (a stored, detached idea).
        ws = r.legacy_wholes(s)
        dv = ws[0]["idea"] if ws else None
        if dv is not None and torch.isfinite(_as_vec(dv)).all():
            cands.append(_fit_dim(dv, D).detach())
            present.append(True)
        else:
            cands.append(zeros)
            present.append(False)

        if not any(present):
            return None, None
        C = torch.stack(cands, dim=0)                  # [3, D] detached
        Cn = C / C.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        q = self.generator.concept_q(s, zeros, None)   # [D] grad-bearing
        qn = q / q.norm().clamp_min(1e-12)
        logits = Cn @ qn                               # [3] grad via q
        if scorer is not None:
            logits = logits + torch.stack(
                [scorer.logit(q, C[i]) for i in range(int(C.shape[0]))])
        # Absent tools -> -inf so the softmax assigns them exactly 0 weight.
        mask = torch.tensor([0.0 if p else float("-inf") for p in present],
                            device=logits.device, dtype=logits.dtype)
        weights = torch.softmax(logits + mask, dim=0)  # [3]
        e_hat = (weights.unsqueeze(-1) * C).sum(0)      # [D] grad via weights
        return e_hat, weights


# -- The answer (policy) loss: train the soft route (Phase C) -----------------

def policy_examples_from_store(reasoner, *, max_examples: int = 8) -> list:
    """Build ``(A, B, gold)`` training examples from the reasoning store,
    self-supervised from the provisioned truthSet -- no labelled QA set needed.
    Each stored 2-hop transitive parthood ``a ⊑ b ⊑ c`` yields a POSITIVE
    ``(a, c, 1.0)`` whose bridge is ``b``; its reverse ``(c, a, 0.0)`` is a
    NEGATIVE. The positives carry the gradient (the policy learns to attend to a
    real bridge); negatives have an empty bridge mask, so they only assert the
    no-hallucination floor. Returns ``[]`` when no store / no 2-hop chains."""
    store = reasoner.reasoning_store()
    if store is None:
        return []
    edges = []
    for (idx, np1, vp, np2, t1) in ConceptualSpace._iter_relation_rows(
            store, _REL_PARTOF):
        if t1 > reasoner.trust_threshold:
            edges.append((_as_vec(np1), _as_vec(np2)))
    examples = []
    for (a, b) in edges:
        for (b2, c) in edges:
            if (reasoner.equal(b, b2) >= reasoner.tau_id
                    and reasoner.equal(a, c) < reasoner.tau_id):
                examples.append((a, c, 1.0))          # a ⊑ b ⊑ c  (true)
                examples.append((c, a, 0.0))          # reversed   (false)
                if len(examples) >= int(max_examples):
                    return examples
    return examples


def _fit_keys(block, d):
    """Pad/truncate a key matrix ``[M, W]`` to width ``d``."""
    M, W = int(block.shape[0]), int(block.shape[1])
    if W == d:
        return block
    out = block.new_zeros(M, d)
    k = min(d, W)
    out[:, :k] = block[:, :k]
    return out


def policy_answer_loss(generator, spaces, reasoner, examples, *,
                       max_keys: int = 512):
    """Train the soft query head against detached hard bridge masks."""
    if generator is None or not spaces or not examples:
        return None
    D = int(generator.dim)
    blocks = []
    for s in spaces:
        k = s.get("keys")
        if not (k is not None and torch.is_tensor(k) and k.dim() in (2, 3)
                and int(k.shape[-2]) > 0):
            continue
        block = (k if k.dim() == 2 else k[0])
        if int(block.shape[0]) > int(max_keys):
            continue                                  # skip a huge percept codebook
        blocks.append(_fit_keys(block.detach().float(), D))
    if not blocks:
        return None
    K = torch.cat(blocks, dim=0)                       # [Mtot, D], detached
    Kn = K / K.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    losses = []
    for (A, B, gold) in examples:
        Av = _fit_dim(A, D)
        Bv = _fit_dim(B, D)
        q = generator.concept_q(Av, Bv)               # [D], grad-bearing
        qn = q / q.norm().clamp_min(1e-12)
        alpha = torch.softmax(Kn @ qn, dim=0)         # [Mtot], grad through q
        mask = torch.zeros(int(K.shape[0]))
        for m in range(int(K.shape[0])):
            key = K[m]
            if (reasoner.legacy_is_part_direct(Av, key) is not None
                    and reasoner.legacy_is_part_direct(key, Bv) is not None):
                mask[m] = 1.0
        support = (alpha * mask.detach()).sum()        # bridge-attention mass (grad)
        predicted_signed = torch.tanh(support)         # (-1,1) via α; mask detached
        losses.append(answer_loss(predicted_signed, gold))
    if not losses:
        return None
    return torch.stack(losses).mean()
