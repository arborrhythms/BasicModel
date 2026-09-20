"""Checked existence, equality and conceptual-taxonomy readers.

Grammatical thought selection and its sole policy loss belong to BasicModel.
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch

from Spaces import ConceptualSpace
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

    Public model entry points adapt this value into the checked grammatical-VP
    registry. It never flattens a structured description into a unary operand.
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

    Typed concept references address the conceptual taxonomy. World facts
    and geometric similarity cannot establish a taxonomy edge.
    """

    def __init__(self, model=None, *, store=None,
                 theta: float = 0.7, tau_id: float = 0.7,
                 trust_threshold: float = 0.0):
        self.model = model
        self._store = store
        self.theta = float(theta)
        self.tau_id = float(tau_id)
        self.trust_threshold = float(trust_threshold)

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

    def existence_evidence(self, description, *, max_records=None, work=None):
        """Ground the full description in accepted LTM facts, without chaining.

        Match each occupied role at the same width and preserve bindings,
        semantic scope and referenced constituents. Signed identity attenuates
        the stored degree; the least-matching role bounds that degree. Keep
        positive and negative evidence independently and never sum repeated
        evidence. Concept activation is not evidence that a referent exists.
        This is a hard lookup, with no derivative through its matching choices.
        """
        if max_records is not None and (type(max_records) is not int or max_records < 0):
            raise ValueError("existence max_records must be a non-negative integer")
        requested = ConceptualMeaning.from_description(description)
        candidates, incomplete = [], []
        diagnostics, scanned = [], 0
        support_true = support_false = 0.0
        store = self.reasoning_store()
        if isinstance(store, TernaryTruthStore):
            count = len(store) if max_records is None else min(len(store), max_records)
            if count < len(store):
                diagnostics.append("capture_limit")
            for i in range(count):
                if work is not None and not work.consume("record"):
                    diagnostics.append("work_budget")
                    break
                scanned += 1
                if int(store.record_kind[i]) != store.KINDS.index("fact"):
                    continue
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
                "incomplete": tuple(diagnostics), "records_scanned": scanned,
                "meaning": requested}

    def exist(self, X) -> float:
        """Lossy legacy scalar view: positive minus negative fact support.

        Checked grammatical execution and evaluation consume existence_evidence
        instead, since a scalar cannot preserve contradictory support.
        """
        evidence = self.existence_evidence(X)
        return evidence["support_true"] - evidence["support_false"]


    def query(self, X, Y=None) -> Optional[dict]:
        """``query(X[, Y])``: an LTM lookup. ``query(X)`` returns the best
        matching stored ABSOLUTE idea ``{idea, trust, row, kind:'idea'}``;
        ``query(X, Y)`` returns the best matching stored relation
        ``{np1, np2, trust, row, kind:'relation'}`` (joint identity to X, Y).
        None when nothing matches.

        This numerical reader is retained independently of the normal thought
        controller. It neither proposes actions nor receives policy credit.
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

    # == retained numerical .where-read ================================

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
        """A numerical ``.where``-typed read over caller-supplied spaces.

        Runs ``ga`` (a GlobalAttention) over the typed ``spaces`` and
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
                          max_nodes=256, max_records=1024, max_expansions=1024,
                          work=None):
        """Read bounded structural inclusion, with native record provenance.

        Legacy vector-only operands have no concept handle and remain unknown.
        Their geometry is never used to guess a referent. Checked grammatical
        callers supply typed references through the current VP registry.
        """
        try:
            part, whole = concept_reference(part), concept_reference(whole)
        except TypeError:
            return {"domain": "conceptual-taxonomy", "support_true": 0.0,
                    "support_false": 0.0, "path": (),
                    "incomplete": ("unbound_concept_reference",),
                    "nodes_scanned": 0, "records_scanned": 0, "edges_expanded": 0}
        cs = getattr(self.model, "conceptualSpace", None)
        from QueryWork import capture_limits
        max_nodes, max_records = capture_limits(work, max_nodes, max_records)
        view = capture_taxonomy(cs, max_nodes=max_nodes, max_records=max_records,
                                focus=(part, whole), work=work)
        return view.part_of(part, whole, max_steps=max_steps,
                            max_expansions=max_expansions, work=work)

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
