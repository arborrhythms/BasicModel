"""The Thinking Kernel: lookup/part/think/query/answer over truth intervals.

The runtime-enforced execution loop of ``doc/plans/thinking_kernel_spec.md``
(execution notes: ``doc/plans/2026-07-12-thinking-kernel-execution.md``). The
kernel drives the HARD tools of :class:`reasoning.TruthGroundedReasoner` from a
stack of STM frames; the policy proposes operations, the runtime executes and
validates them, and only the runtime's two grounded paths (materialize /
incorporate) may write LTM.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from reasoning import (QuerySpec, TruthGroundedReasoner, _as_vec,
                       KIND_IS_TRUE, KIND_IS_PART, KIND_IS_EQUAL)
from Layers import TernaryTruthStore

# Frame / answer values (spec §9.1). ``true``/``false``/``unknown``/``mixed``/
# ``conflicting`` mirror the interval statuses; ``bounded_unknown`` is the
# budget-exhaustion terminal (§9.2 rule 4).
TRUE = "true"
FALSE = "false"
UNKNOWN = "unknown"
MIXED = "mixed"
CONFLICTING = "conflicting"
BOUNDED_UNKNOWN = "bounded_unknown"

_OPS = ("lookup", "part", "think", "query", "answer")


# -- Truth intervals (spec §1.3/§5.2) -----------------------------------------

@dataclass
class TruthInterval:
    """Signed truth interval ``[lower, upper] ⊆ [-1, 1]`` + trust + provenance."""

    lower: float = 0.0
    upper: float = 0.0
    trust: float = 0.0
    provenance: list = field(default_factory=list)

    @property
    def luminosity(self) -> float:
        """Distance from unknownness (§1.2; the §15 sketch's max-abs)."""
        return max(abs(self.lower), abs(self.upper))

    def status(self, tau: float = 0.5) -> str:
        """Classify against a determination bar ``tau``: one-sided luminous ⇒
        true/false; two-sided-strong ⇒ conflicting; a luminous straddle ⇒
        mixed; else unknown."""
        if self.lower <= -tau and self.upper >= tau:
            return CONFLICTING
        if self.luminosity < tau:
            return UNKNOWN
        if self.lower > 0.0:
            return TRUE
        if self.upper < 0.0:
            return FALSE
        return MIXED

    @classmethod
    def from_evidence(cls, evidence):
        """Build from ``[(signed_value, trust, provenance), …]``; empty ⇒
        ``[0, 0]`` at trust 0 (effectively unknown)."""
        ev = [e for e in (evidence or []) if e is not None]
        if not ev:
            return cls()
        vals = [float(v) for (v, _t, _p) in ev]
        return cls(lower=min(vals), upper=max(vals),
                   trust=max(abs(float(t)) for (_v, t, _p) in ev),
                   provenance=[p for (_v, _t, p) in ev])


# -- Testimony (spec §8.3) -----------------------------------------------------

@dataclass
class Testimony:
    """An external answer: evidence, never truth (§14.3)."""

    __test__ = False               # not a test class despite the Test* name

    proposition: Any
    value: Any
    source: str
    source_trust: float = 0.0
    channel_trust: float = 1.0
    provenance: list = field(default_factory=list)

    @property
    def effective_trust(self) -> float:
        return float(self.source_trust) * float(self.channel_trust)


# -- STM frames (spec §2.2/§9.3) -----------------------------------------------

@dataclass
class ChildResult:
    """What a closing frame returns to its parent — scratch is discarded."""

    target: Any
    value: str
    interval: TruthInterval
    trust: float
    trace: list
    provenance: list = field(default_factory=list)
    relevance_to_parent: float = 1.0


@dataclass
class Frame:
    """One STM frame: target + purpose + local scratch + trace + status."""

    target: QuerySpec
    purpose: str = ""
    depth: int = 0
    bindings: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)
    budget_at_push: int = 0
    status: str = "open"
    result: Optional[ChildResult] = None

    @property
    def closed(self) -> bool:
        return self.status != "open"


def as_spec(target) -> QuerySpec:
    """Normalize a location to a QuerySpec: a bare idea vector denotes its own
    existence question (isTrue); a QuerySpec passes through."""
    if isinstance(target, QuerySpec):
        return target
    return QuerySpec(KIND_IS_TRUE, _as_vec(target))


# -- The kernel ----------------------------------------------------------------

class ThinkingKernel:
    """Runtime for the five kernel operations over one budgeted frame stack.

    The policy (``next_operation``) proposes; :meth:`execute` validates the op,
    charges the shared budget pool, runs the semantics, and appends the trace.
    LTM writes happen only inside the runtime: :meth:`_materialize_close`
    (trusted derivation) and :meth:`incorporate` (testimony above the floor).
    """

    # Policy constants (test seams, class-attribute convention).
    tau = 0.5                     # determination bar for status/closure
    trust_bar = 0.0               # minimum trust to close on rule 1/2
    step_cost = 0.01              # per-op reward charge (§12.2)
    terminal_reward = 1.0         # rule-1/2 close bonus (§12.3)
    incorporate_floor = 0.5       # testimony write floor (source×channel)
    max_depth = 8                 # frame-stack depth cap
    consult_addressees = True     # leaf policy may query() before unknown
    default_source_trust = 0.5    # reliability when none given at registration

    def __init__(self, reasoner, *, budget: int = 16, policy=None,
                 generator=None, ga=None, spaces=None, materialize=False):
        self.reasoner = reasoner
        self.budget = int(budget)
        self.policy = policy or KernelPolicy()
        self.generator = generator
        self.ga = ga
        self.spaces = spaces or []
        self.materialize = bool(materialize)
        self.addressees: dict[str, tuple] = {}      # name -> (fn, reliability)
        if getattr(reasoner, "model", None) is not None:
            # The discourse predictor lives outside the kernel's memory --
            # reachable only through query(), like any other external tool.
            self.register_addressee(
                "arma", lambda target: reasoner.arma(as_spec(target).left))
        self._pool = 0
        self.stack: list[Frame] = []

    # == the execution loop (spec §10.1) ====================================

    def run(self, target, *, purpose="answer") -> ChildResult:
        """``think(target)`` at the top level: push the main frame and drive
        ``policy.next_operation`` until it closes. Returns the ChildResult."""
        self._pool = int(self.budget)
        self.stack = []
        return self._think(as_spec(target), purpose=purpose)

    def _think(self, spec: QuerySpec, *, purpose="") -> ChildResult:
        """Push a frame, loop the policy over it, pop it (spec §7.1)."""
        frame = Frame(target=as_spec(spec), purpose=purpose,
                      depth=len(self.stack), budget_at_push=self._pool)
        self.stack.append(frame)
        try:
            while not frame.closed:
                if self._pool <= 0 or frame.depth >= self.max_depth:
                    # Budget exhausted (§9.2 rule 4) -- but rule 1 outranks it:
                    # evidence already gathered may close the frame grounded.
                    iv = self._frame_interval(frame)
                    st = iv.status(self.tau)
                    if st in (TRUE, FALSE) and iv.trust > self.trust_bar:
                        self._close(frame, st, iv, how="grounded")
                    else:
                        self._close(frame, BOUNDED_UNKNOWN, iv, how="budget")
                    break
                op = self.policy.next_operation(frame, self)
                self.execute(frame, op)
        finally:
            self.stack.pop()
        return frame.result

    def execute(self, frame: Frame, op: dict):
        """Validate + run ONE proposed operation against ``frame``."""
        name = (op or {}).get("op")
        if name not in _OPS:
            raise ValueError(f"ThinkingKernel: unknown operation {name!r}")
        self._pool -= 1
        lum_before = self._frame_interval(frame).luminosity
        entry = {"op": name, "depth": frame.depth, "lum_before": lum_before}
        if name == "lookup":
            iv = self.lookup(op.get("target", frame.target))
            frame.bindings["interval"] = iv
            entry["interval"] = iv
        elif name == "part":
            rels = self.part(op["location"], mode=op.get("mode", "meronomy"),
                             direction=op.get("direction", "up"))
            frame.bindings.setdefault("candidates", []).extend(rels)
            entry["n"] = len(rels)
        elif name == "think":
            child = self._think(as_spec(op["target"]),
                                purpose=op.get("purpose", "subgoal"))
            frame.bindings.setdefault("children", []).append(
                {"result": child, "hop_trust": float(op.get("hop_trust", 1.0)),
                 "hop_row": op.get("hop_row")})
            entry["child_value"] = child.value
        elif name == "query":
            t = self.query(op["addressee"], op.get("target", frame.target))
            frame.bindings.setdefault("testimony", []).append(t)
            entry["testimony"] = t
        else:                                          # answer
            self._answer(frame, op)
        entry["lum_after"] = self._frame_interval(frame).luminosity
        frame.trace.append(entry)

    # == lookup (spec §5): LTM-direct evidence, no chaining ==================

    def lookup(self, target) -> TruthInterval:
        spec = as_spec(target)
        r = self.reasoner
        ev = []
        if spec.predicate != KIND_IS_TRUE and spec.right is None:
            return TruthInterval()         # an open binary query has no direct row
        if spec.predicate == KIND_IS_TRUE:
            dot = r.exist(spec.left)
            if dot != 0.0:
                ev.append((dot, dot, {"how": "exist"}))
        elif spec.predicate == KIND_IS_EQUAL:
            s = r.equal(spec.left, spec.right, isomorphic=True)
            if s > 0.0:
                ev.append((s, s, {"how": "equal"}))
        else:                                          # KIND_IS_PART
            d = r.is_part_direct(spec.left, spec.right)
            if d is not None:
                ev.append((d[0], d[0], {"how": d[1]}))
            refute = r._refuting_direct(spec.left, spec.right)
            if refute > 0.0:
                ev.append((-refute, refute, {"how": "refuting"}))
        return TruthInterval.from_evidence(ev)

    # == part (spec §6): structural traversal ================================

    def part(self, location, *, mode="meronomy", direction="up") -> list:
        """Proximal structure of ``location``. Both modes read the REL_PARTOF
        rows today (subsumption is stored as parthood; the mode rides the
        provenance for when the stores split — see the execution notes §0)."""
        if mode not in ("meronomy", "taxonomy"):
            raise ValueError(f"part: unknown mode {mode!r}")
        if direction not in ("up", "down"):
            raise ValueError(f"part: unknown direction {direction!r}")
        fn = self.reasoner.wholes if direction == "up" else self.reasoner.parts
        out = []
        for rel in fn(_as_vec(location)):
            rel = dict(rel)
            rel["mode"] = mode
            rel["direction"] = direction
            out.append(rel)
        return out

    # == query (spec §8): outward, testimony only ============================

    def register_addressee(self, name: str, fn: Callable, *,
                           source_trust: float = None):
        """Register an external addressee with its reliability (§1.4 source
        trust; default :attr:`default_source_trust`)."""
        t = (self.default_source_trust if source_trust is None
             else float(source_trust))
        self.addressees[str(name)] = (fn, t)

    def query(self, addressee: str, target) -> Testimony:
        entry = self.addressees.get(str(addressee))
        if entry is None:
            return Testimony(proposition=target, value=None,
                             source=str(addressee), source_trust=0.0,
                             provenance=[{"error": "unknown addressee"}])
        fn, source_trust = entry
        try:
            value = fn(target)
        except Exception as e:
            return Testimony(proposition=target, value=None,
                             source=str(addressee), source_trust=0.0,
                             provenance=[{"error": repr(e)}])
        return Testimony(proposition=target, value=value,
                         source=str(addressee), source_trust=source_trust,
                         provenance=[{"how": "query"}])

    def incorporate(self, testimony: Testimony) -> int:
        """The §14.2 external-evidence grounding path: write testimony to LTM
        with its effective trust — only above the floor, only by the runtime.
        Returns the row index or -1 (no store / below floor / bad shape)."""
        reliability = float(testimony.effective_trust)
        store = self.reasoner.reasoning_store()
        if (store is None or testimony.value is None
                or reliability < self.incorporate_floor):
            return -1
        # A scalar testimony value is the asserted signed truth; a reliable
        # source asserting false writes a NEGATIVE-trust row.
        try:
            signed = max(-1.0, min(1.0, float(testimony.value)))
        except (TypeError, ValueError):
            signed = 1.0
        t = reliability * signed
        spec = as_spec(testimony.proposition)
        if not isinstance(store, TernaryTruthStore):
            return -1
        if spec.predicate == KIND_IS_PART and spec.right is not None:
            A = _as_vec(spec.left)
            return int(store.append_relation(
                A, torch.zeros_like(A), _as_vec(spec.right),
                rel_type=store.REL_PARTOF, trust=t))
        return int(store.append_idea(_as_vec(spec.left), trust=t))

    # == answer (spec §9): closure rules ======================================

    def _frame_interval(self, frame: Frame) -> TruthInterval:
        """The frame's current best interval: its own lookup, WIDENED by any
        verified child conclusion (a child's true at trust t is evidence
        ``(+t)`` for the parent target it was opened to support) and by
        NUMERIC testimony (§14.2 external query with source trust: the
        asserted signed value SCALED by the source's reliability, so flimsy
        testimony cannot manufacture luminosity)."""
        iv = frame.bindings.get("interval") or TruthInterval()
        ev = [(iv.lower, iv.trust, {"how": "lookup"}),
              (iv.upper, iv.trust, {"how": "lookup"})] if iv.provenance else []
        for ch in frame.bindings.get("children", []):
            res = ch["result"]
            if res is not None and res.value in (TRUE, FALSE):
                t = min(float(res.trust), float(ch.get("hop_trust", 1.0)))
                ev.append((t if res.value == TRUE else -t, t,
                           {"how": "child", "target": res.target}))
        for t in frame.bindings.get("testimony", []):
            rel = float(t.effective_trust)
            if t.value is None or rel <= 0.0 or torch.is_tensor(t.value):
                continue                   # content, not a truth assertion
            try:
                signed = max(-1.0, min(1.0, float(t.value)))
            except (TypeError, ValueError):
                continue
            ev.append((signed * rel, rel,
                       {"how": "testimony", "source": t.source}))
        return TruthInterval.from_evidence(ev) if ev else iv

    def _answer(self, frame: Frame, op: dict):
        """Close ``frame`` under the §9.2 rules. An asserted true/false that
        the frame's evidence does not support is an UNSUPPORTED ASSERTION
        (§12.3): the runtime refuses it and closes unknown instead."""
        value = str(op.get("value", UNKNOWN))
        iv = self._frame_interval(frame)
        status = iv.status(self.tau)
        if value in (TRUE, FALSE):
            supported = (status == value and iv.trust > self.trust_bar)
            if not supported:
                self._close(frame, UNKNOWN, iv, how="unsupported_assertion")
                return
            self._close(frame, value, iv, how="grounded",
                        trust=float(op.get("trust", iv.trust)))
            return
        if value in (MIXED, CONFLICTING) and status in (MIXED, CONFLICTING):
            self._close(frame, status, iv, how="two_sided")
            return
        how = "budget" if value == BOUNDED_UNKNOWN else "search_exhausted"
        self._close(frame, value if value == BOUNDED_UNKNOWN else UNKNOWN,
                    iv, how=how)

    def _close(self, frame: Frame, value: str, iv: TruthInterval, *,
               how: str, trust: float = None):
        frame.status = "closed"
        frame.result = ChildResult(
            target=frame.target, value=value, interval=iv,
            trust=float(iv.trust if trust is None else trust),
            trace=list(frame.trace) + [{"op": "answer", "value": value,
                                        "how": how}],
            provenance=list(iv.provenance))
        if value == TRUE and how == "grounded":
            self._materialize_close(frame)

    def _materialize_close(self, frame: Frame):
        """Trusted-derivation LTM write (§14.2): a frame that closed TRUE via
        child subgoals writes the derived isPart edge back as a lemma, exactly
        the reasoner's materialize path (cycle-guarded, floor-gated)."""
        spec = frame.target
        if (not self.materialize or spec.predicate != KIND_IS_PART
                or not frame.bindings.get("children")):
            return
        row = self.reasoner.materialize(spec.left, spec.right,
                                        frame.result.trust)
        if row >= 0:
            frame.result.provenance.append({"materialized_row": int(row)})

    # == soft candidate ordering (execution notes §2.4) =======================

    def order_candidates(self, frame: Frame, cands: list) -> list:
        """Rank part() candidates for exploration: by the soft α when the
        generator half is present (soft-propose / hard-verify — the α only
        ORDERS, it never asserts), else by stored hop trust."""
        spec = frame.target
        if (self.generator is not None and self.ga is not None and self.spaces
                and spec.right is not None):
            try:
                out = self.generator.propose(
                    _as_vec(spec.left), _as_vec(spec.right), self.spaces,
                    ga=self.ga, top_k=max(8, len(cands)))
            except Exception:
                out = None
            if out is not None:
                def alpha(c):
                    best = 0.0
                    for k in out.get("candidates", []):
                        if self.reasoner.equal(c["idea"], k["idea"]) >= \
                                self.reasoner.tau_id:
                            best = max(best, float(k.get("alpha", 0.0)))
                    return best
                return sorted(cands, key=lambda c: -alpha(c))
        return sorted(cands, key=lambda c: -float(c.get("trust", 0.0)))

    # == rewards + traces (spec §12) ==========================================

    def compile_rewards(self, frame_result: ChildResult) -> dict:
        """Compile the frame-local rewards FROM the trace (§12.4 — the mind
        never writes reward): per-op ``Δluminosity − step_cost``, plus the
        terminal (§12.3): grounded true/false ⇒ +terminal_reward; a valid
        unknown/bounded_unknown ⇒ 0; an unsupported assertion ⇒ −terminal."""
        steps = []
        terminal = 0.0
        for e in frame_result.trace:
            if e.get("op") == "answer":
                how = e.get("how")
                if how == "grounded":
                    terminal = self.terminal_reward
                elif how == "unsupported_assertion":
                    terminal = -self.terminal_reward
                continue
            dl = float(e.get("lum_after", 0.0)) - float(e.get("lum_before", 0.0))
            steps.append(dl - self.step_cost)
        return {"steps": steps, "terminal": terminal,
                "total": sum(steps) + terminal}

    @staticmethod
    def trace_examples(frame_result: ChildResult) -> list:
        """Successful-trace supervision pairs (§12.6): ``(state, op)`` for
        next-operation prediction. Empty unless the close was grounded."""
        closed = frame_result.trace[-1] if frame_result.trace else {}
        if closed.get("how") != "grounded":
            return []
        examples = []
        n = 0
        for e in frame_result.trace:
            state = {"kind": frame_result.target.predicate,
                     "depth": int(e.get("depth", 0)), "n_ops": n,
                     "lum": float(e.get("lum_before", 0.0))}
            examples.append((state, e["op"]))
            n += 1
        return examples


# -- Next-operation prediction (spec §12.6/12.7) --------------------------------

_KIND_INDEX = {KIND_IS_TRUE: 0, KIND_IS_PART: 1, KIND_IS_EQUAL: 2}


class NextOpPolicy(nn.Module):
    """Learned next-operation head, behavior-cloned on successful traces.

    Featurizes a ``trace_examples`` state descriptor (predicate kind one-hot +
    depth + op count + target luminosity) and scores the five kernel ops.
    Trained by :func:`next_op_loss`; consulted by :class:`KernelPolicy` ONLY
    at explore-vs-stop choice points — the runtime still validates every op,
    so a bad head can waste budget but never assert."""

    N_FEATURES = 6

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(self.N_FEATURES, int(hidden)), nn.ReLU(),
            nn.Linear(int(hidden), len(_OPS)))
        # NEUTRAL AT INIT: zero the output layer so an UNTRAINED head scores
        # every op identically -- choose() then breaks the tie toward the
        # explore option (listed first), i.e. exactly the deterministic
        # baseline. Random init made an untrained head randomly prefer
        # "answer", killing climbs at inference before any training.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def featurize(state: dict) -> torch.Tensor:
        f = torch.zeros(NextOpPolicy.N_FEATURES)
        k = _KIND_INDEX.get(state.get("kind"))
        if k is not None:
            f[k] = 1.0
        f[3] = float(state.get("depth", 0))
        f[4] = float(state.get("n_ops", 0))
        f[5] = float(state.get("lum", 0.0))
        return f

    def logits(self, state: dict) -> torch.Tensor:
        return self.head(self.featurize(state))

    def choose(self, state: dict, options) -> str:
        """The highest-scoring op among ``options`` (restricted argmax; ties
        keep the FIRST option, so a neutral head defers to the caller's
        preferred order)."""
        with torch.no_grad():
            lg = self.logits(state)
        return max(options, key=lambda o: float(lg[_OPS.index(o)]))


def next_op_loss(head, examples):
    """Cross-entropy next-op prediction loss over ``(state, op)`` pairs (the
    §12.6 supervised objective on successful traces). ``None`` when there is
    nothing to train on."""
    if head is None or not examples:
        return None
    feats = torch.stack([NextOpPolicy.featurize(s) for (s, _op) in examples])
    targets = torch.tensor([_OPS.index(op) for (_s, op) in examples])
    return nn.functional.cross_entropy(head.head(feats), targets)


def traces_from_store(kernel, *, max_targets: int = 4) -> list:
    """Self-supervised trace generation (§12.6): derive 2-hop transitive
    isPart targets from the reasoning store (the ``policy_examples_from_
    store`` positives), run the kernel on each, and keep the ``(state, op)``
    pairs of the GROUNDED traces. Run the teacher with ``materialize=False``
    so trace generation never writes LTM."""
    from reasoning import policy_examples_from_store
    targets = [(a, c) for (a, c, gold) in policy_examples_from_store(
        kernel.reasoner, max_examples=2 * int(max_targets)) if gold >= 0.5]
    examples = []
    for (a, c) in targets[:int(max_targets)]:
        res = kernel.run(QuerySpec(KIND_IS_PART, a, c))
        examples.extend(ThinkingKernel.trace_examples(res))
    return examples


# -- The deterministic baseline policy (execution notes §2.4) ------------------

class KernelPolicy:
    """lookup → close if luminous → climb taxonomy-up via think() subgoals.

    ``next_op``: an optional :class:`NextOpPolicy` consulted at the two
    explore-vs-stop choice points (query another addressee vs answer; open
    another think() subgoal vs answer). The head can only pick among the
    LEGAL options the deterministic policy offers."""

    def __init__(self, next_op=None):
        self.next_op = next_op

    def _prefer_stop(self, frame: Frame, kernel: ThinkingKernel,
                     explore_op: str) -> bool:
        """True iff the learned head prefers answering now over exploring."""
        if self.next_op is None:
            return False
        iv = kernel._frame_interval(frame)
        state = {"kind": frame.target.predicate, "depth": frame.depth,
                 "n_ops": len(frame.trace), "lum": iv.luminosity}
        try:
            return self.next_op.choose(state, (explore_op, "answer")) \
                == "answer"
        except Exception:
            return False

    def next_operation(self, frame: Frame, kernel: ThinkingKernel) -> dict:
        spec = frame.target
        b = frame.bindings
        # 1. Always inspect memory first (depth-0 curriculum shape).
        if "interval" not in b:
            return {"op": "lookup", "target": spec}
        iv = kernel._frame_interval(frame)
        status = iv.status(kernel.tau)
        if status in (TRUE, FALSE) and iv.trust > kernel.trust_bar:
            return {"op": "answer", "value": status, "trust": iv.trust}
        if status in (MIXED, CONFLICTING):
            return {"op": "answer", "value": status}
        # 2. Leaves (isTrue/isEqual) have no structural climb: consult the
        #    registered external addressees once each (§7.1 "possibly
        #    query()"; numeric testimony folds into the frame interval at
        #    source trust, §14.2), then unknown is the valid close (§14.6).
        if spec.predicate != KIND_IS_PART or spec.right is None:
            if kernel.consult_addressees:
                asked = b.setdefault("queried_addressees", set())
                for name in sorted(kernel.addressees):
                    if name not in asked:
                        if self._prefer_stop(frame, kernel, "query"):
                            break
                        asked.add(name)
                        return {"op": "query", "addressee": name,
                                "target": spec}
            return {"op": "answer", "value": UNKNOWN}
        # 3. isPart(A, B): expose A's wholes once...
        if "candidates" not in b:
            return {"op": "part", "location": spec.left,
                    "mode": "taxonomy", "direction": "up"}
        # 4. ...then open one subgoal per unvisited whole, best-ranked first.
        visited = b.setdefault("visited", set())
        fresh = [c for c in b.get("candidates", [])
                 if int(c.get("row", -1)) not in visited]
        for cand in kernel.order_candidates(frame, fresh):
            if self._prefer_stop(frame, kernel, "think"):
                break
            visited.add(int(cand.get("row", -1)))
            sub = QuerySpec(KIND_IS_PART, cand["idea"], spec.right)
            return {"op": "think", "target": sub, "purpose": "climb",
                    "hop_trust": float(cand.get("trust", 0.0)),
                    "hop_row": cand.get("row")}
        # 5. Bounded search failed (§9.2 rule 3).
        return {"op": "answer", "value": UNKNOWN}
