"""PartSpace meronymic analyzer (PS forward analysis / PS reverse
synthesis).

doc/plans/2026-05-30-subsymbolic-analyzer-terminal-emitter.md. This module
holds the PS-side machinery that mirrors the symbolic parser in the
meronymic direction:

  * :class:`EndpointSumWhere` -- the invertible endpoint-sum span key
    ``where = phase(start) + phase(end)`` ("Where Encoding And Spans").
  * :class:`MeronymicAnalyzer` -- analyzes a surface whole into perceptual
    parts (terminals) and writes durable span state to an ``ObjectSubSpace``
    ("Forward PS Analysis"); reverse-synthesizes surface from route metadata
    ("Reverse PS Synthesis"). The first analyzer mode is a *compatibility
    mode* (stop / whitespace ``boundary`` / ``uniform`` + byte fallback)
    that reproduces the current word-lexer terminal sequence.

Durable analysis state lives on ``ObjectSubSpace`` (bin/Language.py); the
trainable routing modules live on the LanguageLayer-like router. This
module is the orchestration + PS-meronymic-operation layer between them.
"""

import math

import torch


class EndpointSumWhere:
    """Invertible endpoint-sum span key ``where = phase(start)+phase(end)``.

    ``phase(p) = [sin(p*div_term), cos(p*div_term)]``; summing the two
    endpoints gives, by the sum-to-product identity,
    ``2*cos((end-start)*div_term/2) * [sin(center*div_term), cos(center*
    div_term)]`` with ``center=(start+end)/2``. Therefore the ANGLE decodes
    the span center and the MAGNITUDE decodes the span length -- two spans
    with the same center but different extent differ by radius.

    The key is recoverable only while centers stay within one period and
    span lengths stay below half the period; :meth:`is_recoverable` checks
    this. ``div_term`` defaults to ``pi / (2*namespace)`` so both the center
    and the length (each at most ``namespace``) sit below the half-period.
    Do NOT normalise the key -- the magnitude carries the length.
    """

    def __init__(self, namespace=256, div_term=None):
        """``namespace`` is the max recoverable byte/atom position; pass an
        explicit ``div_term`` to share one period across spaces."""
        self.namespace = int(namespace)
        if div_term is None:
            div_term = math.pi / (2.0 * max(1, self.namespace))
        self.div_term = float(div_term)
        # Centers / lengths must stay strictly below this to round-trip.
        self.half_period = math.pi / self.div_term

    def encode(self, start, end):
        """Encode span ``[start, end)`` into a 2-vector key (no batching)."""
        dt = self.div_term
        ds, de = float(start) * dt, float(end) * dt
        return torch.tensor(
            [math.sin(ds) + math.sin(de), math.cos(ds) + math.cos(de)],
            dtype=torch.float32)

    def decode(self, where):
        """Decode a key (``[2]`` or batched ``[..., 2]``) to (start, end).

        Returns python ints for a single key, or a pair of ``LongTensor``s
        for a batched input. Decoded endpoints are snapped to the integer
        boundary grid.
        """
        where = torch.as_tensor(where, dtype=torch.float32)
        w0, w1 = where[..., 0], where[..., 1]
        dt = self.div_term
        center = torch.atan2(w0, w1) / dt
        radius = torch.sqrt(w0 * w0 + w1 * w1)
        length = 2.0 * torch.arccos(torch.clamp(radius / 2.0, -1.0, 1.0)) / dt
        start = torch.round(center - length / 2.0).long()
        end = torch.round(center + length / 2.0).long()
        if where.dim() == 1:
            return int(start.item()), int(end.item())
        return start, end

    def is_recoverable(self, start, end):
        """True iff ``[start, end)`` round-trips under this namespace/period
        (center and length both inside the recoverable half-period)."""
        center = (float(start) + float(end)) / 2.0
        length = float(end) - float(start)
        return (0.0 <= center < self.half_period
                and 0.0 <= length < self.half_period)


def byte_fallback_vector(text, dim):
    """Deterministic ``[dim]`` percept vector for a surface span from its
    bytes -- the analyzer's stand-in for the percept store's
    ``BytesFallbackEncoder`` (sum of byte-codebook rows / sqrt(len)) when
    running standalone. Total: every byte yields a vector, so unknown
    surface is always covered.
    """
    raw = text.encode("utf-8") if isinstance(text, str) else bytes(text)
    if not raw:
        return torch.zeros(int(dim), dtype=torch.float32)
    j = torch.arange(int(dim), dtype=torch.float32)
    acc = torch.zeros(int(dim), dtype=torch.float32)
    for k, byte in enumerate(raw):
        acc = acc + torch.sin((byte + 1) * (j + 1) * 0.13 + k * 0.07)
    return acc / math.sqrt(len(raw))


class MeronymicRouter:
    """Learned meronymic router: ONE hard route + soft marginals over a
    surface's atom sequence, reusing the SHARED inverse-routing primitive.

    ``binary_tiling_viterbi`` / ``binary_tiling_soft_dp`` (bin/Language.py)
    are the direction-neutral routing core the symbolic
    ``BinaryStructuredReductionLayer`` already uses for WS reduce / unreduce;
    the meronymic analyzer reuses the very same DP in the analysis
    direction, so PS analysis and WS unreduce share one primitive (Phase R3,
    "Factor the shared inverse routing primitive out of unreduce /
    reverse_stack").

    Boundary evidence is SIGNED-NEIGHBORHOOD: the merge (reduce) score
    between adjacent atoms ``t, t+1`` is their cosine similarity -- bound
    neighbours (bytes inside a word) score high, a boundary scores low --
    minus a ``depth_penalty`` that thresholds merging and so controls
    terminal granularity. Because a merge fires iff the signed evidence
    beats the keep cost, the route is the exact Viterbi optimum, not a
    greedy / beam pick. The soft marginals from the same scores are returned
    alongside the one hard route.
    """

    def __init__(self, depth_penalty=0.0, keep_bias=0.0, max_rounds=64):
        self.depth_penalty = float(depth_penalty)
        self.keep_bias = float(keep_bias)
        self.max_rounds = int(max_rounds)

    def scores(self, atoms, bonus=None):
        """Signed-neighborhood ``(copy_score [1,N,1], reduce_score [1,N-1,1])``.

        ``bonus`` (optional ``[N-1]``) adds known-percept evidence so a
        recognized chunk's atoms cohere more strongly than its raw cosine.
        """
        atoms = torch.as_tensor(atoms, dtype=torch.float32)
        N = int(atoms.shape[0])
        copy = torch.full((1, max(N, 0), 1), self.keep_bias, dtype=torch.float32)
        if N < 2:
            return copy, torch.zeros(1, 0, 1, dtype=torch.float32)
        x = atoms / (atoms.norm(dim=-1, keepdim=True) + 1e-8)
        cos = (x[:-1] * x[1:]).sum(-1)                       # [N-1]
        reduce = cos - self.depth_penalty
        if bonus is not None:
            reduce = reduce + torch.as_tensor(bonus, dtype=torch.float32).view(-1)
        return copy, reduce.view(1, N - 1, 1)

    def route_once(self, copy_score, reduce_score):
        """One DP level: exact Viterbi hard route + soft marginals.

        Returns ``dict(merges, reduce_mask, reduce_marginal, score)`` where
        ``merges`` is the sorted list of positions ``t`` at which the pair
        ``(t, t+1)`` is merged (non-overlapping by construction).
        """
        from Language import binary_tiling_viterbi, binary_tiling_soft_dp
        copy_score = torch.as_tensor(copy_score, dtype=torch.float32)
        reduce_score = torch.as_tensor(reduce_score, dtype=torch.float32)
        M = int(reduce_score.shape[1]) if reduce_score.dim() == 3 else 0
        if M == 0:
            return {"merges": [], "reduce_mask": [], "reduce_marginal": [],
                    "score": float(copy_score.sum())}
        hard = binary_tiling_viterbi(copy_score, reduce_score)
        soft = binary_tiling_soft_dp(copy_score, reduce_score)
        mask = hard["reduce_mask"].squeeze(0).sum(-1)        # [M] 0/1
        merges = [t for t in range(M) if float(mask[t]) > 0.5]
        return {"merges": merges,
                "reduce_mask": [int(float(mask[t]) > 0.5) for t in range(M)],
                "reduce_marginal": soft["reduce_marginal"].squeeze(0).tolist(),
                "score": float(hard["score"])}

    def route(self, atoms, bonus=None, bonus_fn=None):
        """Iterated agglomerative meronymic route over ``atoms`` ``[N, D]``.

        Each round scores the current sequence by signed-neighborhood
        evidence, runs the exact DP, and merges the chosen non-overlapping
        adjacent pairs (sum-combining their vectors and unioning their
        ORIGINAL-index spans). Converges when no pair clears the penalty --
        the byte/atom cover is total, so a valid route always exists.

        ``bonus`` is a static leaf-level per-pair bonus (applied on the
        first round only). ``bonus_fn(segments) -> [len(segments)-1]`` is
        re-evaluated every round against the CURRENT segment list, so
        cross-level evidence (e.g. "this merged span is still inside a known
        word") keeps cohering a chunk that one non-overlapping DP level
        cannot fully merge. Returns ``dict(segments=[(start, end), ...],
        n_merges, rounds)`` (``rounds`` = per-round reduce marginals).
        """
        atoms = torch.as_tensor(atoms, dtype=torch.float32)
        N = int(atoms.shape[0])
        if N == 0:
            return {"segments": [], "n_merges": 0, "rounds": []}
        segs = [(i, i + 1) for i in range(N)]
        vecs = atoms.clone()
        n_merges, rounds = 0, []
        for _ in range(self.max_rounds):
            b = bonus_fn(segs) if bonus_fn is not None else bonus
            copy_score, reduce_score = self.scores(vecs, bonus=b)
            out = self.route_once(copy_score, reduce_score)
            rounds.append(out["reduce_marginal"])
            merges = set(out["merges"])
            if not merges:
                break
            n_merges += len(merges)
            new_segs, new_vecs, i, M = [], [], 0, len(segs)
            while i < M:
                if i in merges:                              # merge seg i, i+1
                    new_segs.append((segs[i][0], segs[i + 1][1]))
                    new_vecs.append(vecs[i] + vecs[i + 1])
                    i += 2
                else:
                    new_segs.append(segs[i])
                    new_vecs.append(vecs[i])
                    i += 1
            segs, vecs = new_segs, torch.stack(new_vecs)
            if bonus_fn is None:
                bonus = None    # static bonus applies to the leaf level only
        return {"segments": segs, "n_merges": n_merges, "rounds": rounds}


class MeronymicAnalyzer:
    """Analyze a surface whole into perceptual parts (terminals).

    Compatibility mode reuses the existing word/byte tokenizer
    (``util.parse``) as the ``boundary`` meronymic operation, so its
    terminal sequence matches the current lexer by construction. Routing
    per span: a known percept is accepted whole (``stop``); an unknown word
    falls back to byte terminals; with no percept store every word is one
    ``boundary`` terminal. Durable spans are written to an
    :class:`ObjectSubSpace`; :meth:`terminal_view` exposes the fixed-
    capacity terminal stream (``what`` / ``where`` / ``ids`` / ``mask`` /
    ``len``) the PS-to-WS binding consumes. ``.where`` is the endpoint-sum
    span key.

    ``percept_lookup`` (optional) maps a span's text to ``(vector, part_id)``
    or ``None`` (unknown). It is the seam to the real percept store /
    RadixLayer; standalone, byte fallback covers everything.
    """

    # Meronymic route ids written to ObjectSubSpace._route_id.
    STOP, BOUNDARY, UNIFORM, BYTE = 0, 1, 2, 3

    def __init__(self, percept_lookup=None, namespace=256, where_enc=None):
        """``percept_lookup``: callable(text) -> (vec, part_id) | None."""
        self.percept_lookup = percept_lookup
        self.where = where_enc or EndpointSumWhere(namespace=namespace)

    def _resolve(self, oss, text):
        """Return ``(vec, part_id)``: a percept-store hit, or byte fallback
        with part_id -1."""
        if self.percept_lookup is not None:
            hit = self.percept_lookup(text)
            if hit is not None:
                vec, pid = hit
                return vec, int(pid)
        return byte_fallback_vector(text, oss.percept_dim), -1

    def _emit(self, oss, b, text, start, end, vec, part_id, route_id, record,
              raw=None):
        """Push one terminal span to the ObjectSubSpace with endpoint-sum
        ``.where`` and append its host-side replay record. When ``raw`` (the
        terminal's exact source bytes) is given it is stored so reverse
        synthesis can reconstruct the surface byte-exactly (no mojibake for a
        multi-byte glyph split across byte terminals)."""
        oss.push(b, vec, part_id=int(part_id), span_start=int(start),
                 span_end=int(end), span_where=self.where.encode(start, end),
                 route_id=int(route_id))
        entry = {"text": text, "start": int(start), "end": int(end),
                 "part_id": int(part_id), "route": int(route_id)}
        if raw is not None:
            entry["raw"] = bytes(raw)
        record.append(entry)

    def analyze(self, surface, oss, b=0, granularity="auto"):
        """Analyze ``surface`` into terminals on row ``b`` of ``oss``.

        ``granularity``: ``"word"`` = whitespace/word boundary (matches the
        word lexer); ``"byte"`` = one terminal per byte; ``"auto"`` = word
        boundary with byte fallback for unknown words (the compatibility
        analyzer mode). Durable span state is written to ``oss``; a
        host-side replay record (one dict per terminal with the canonical
        ``text`` + span + route) is returned so reverse synthesis can do
        exact replay (the standalone analyzer's stand-in for the percept
        store's canonical bytes).
        """
        from util import parse
        record = []
        if granularity == "byte":
            for ch, off in parse(surface, lex="bytes"):
                vec, pid = self._resolve(oss, ch)
                self._emit(oss, b, ch, off, off + len(ch.encode("utf-8")),
                           vec, pid, self.BYTE, record)
            return record
        for tok, start in parse(surface, lex="words"):
            end = start + len(tok.encode("utf-8"))
            hit = (self.percept_lookup(tok)
                   if self.percept_lookup is not None else None)
            if hit is not None:
                vec, pid = hit
                self._emit(oss, b, tok, start, end, vec, int(pid),
                           self.STOP, record)
            elif granularity == "auto" and self.percept_lookup is not None:
                # Unknown word: cover it with byte terminals (total fallback).
                for k, byte in enumerate(tok.encode("utf-8")):
                    vec, pid = self._resolve(oss, chr(byte))
                    self._emit(oss, b, chr(byte), start + k, start + k + 1,
                               vec, pid, self.BYTE, record)
            else:
                vec, pid = self._resolve(oss, tok)
                self._emit(oss, b, tok, start, end, vec, pid,
                           self.BOUNDARY, record)
        return record

    def analyze_routed(self, surface, oss, b=0, router=None, known_bonus=10.0):
        """Learned-route PS analysis (Phase R3): segment ``surface`` with the
        meronymic Viterbi router (:class:`MeronymicRouter`) instead of the
        ``util.parse`` heuristic, then resolve each routed segment to a
        terminal (known percept -> ``stop``; unknown -> byte fallback) and
        write it to ``oss``.

        The router's signed-neighborhood evidence is boosted inside spans
        the percept store recognizes (``known_bonus`` on every byte-pair
        whose merged span stays inside a known word), so a known word's
        bytes cohere into one ``stop`` terminal while unknown surface stays
        byte terminals -- the same known-vs-byte cover the compatibility
        analyzer produces, now selected by the SHARED DP primitive
        (``binary_tiling_viterbi``) rather than a tokenizer. Returns the
        host-side replay record (so :meth:`synthesize` round-trips).
        """
        raw = (surface.encode("utf-8")
               if isinstance(surface, str) else bytes(surface))
        record = []
        if not raw:
            return record
        if router is None:
            # Penalty above the raw byte-cosine ceiling so ONLY known-word
            # evidence (the bonus) drives merges; unknown surface stays byte.
            router = MeronymicRouter(depth_penalty=2.0)
        dim = oss.percept_dim
        atoms = torch.stack(
            [byte_fallback_vector(bytes([byte_]), dim) for byte_ in raw])
        # Known-word byte spans (the standalone stand-in for the percept
        # store's "where do recognized percepts live" evidence).
        known_spans = []
        if self.percept_lookup is not None:
            from util import parse
            for tok, start in parse(surface, lex="words"):
                if self.percept_lookup(tok) is not None:
                    known_spans.append((start, start + len(tok.encode("utf-8"))))

        def bonus_fn(segs):
            out = torch.zeros(max(len(segs) - 1, 0))
            for i in range(len(segs) - 1):
                s0, e1 = segs[i][0], segs[i + 1][1]
                if any(ss <= s0 and e1 <= we for (ss, we) in known_spans):
                    out[i] = known_bonus
            return out

        segments = router.route(atoms, bonus_fn=bonus_fn)["segments"]
        for (s, e) in segments:
            seg = bytes(raw[s:e])                 # the terminal's EXACT bytes
            # Percept lookup only on a cleanly-decodable span; a partial
            # UTF-8 byte terminal is never a known percept -> byte fallback.
            hit = None
            if self.percept_lookup is not None:
                try:
                    token = seg.decode("utf-8")
                except UnicodeDecodeError:
                    token = None
                if token is not None:
                    hit = self.percept_lookup(token)
            if hit is not None:
                vec, pid = hit
                self._emit(oss, b, token, s, e, vec, int(pid),
                           self.STOP, record, raw=seg)
            else:
                # Byte fallback: vector from the RAW bytes (never the lossy
                # decode), exact bytes stored for byte-exact reverse synthesis.
                vec = byte_fallback_vector(seg, oss.percept_dim)
                text = seg.decode("utf-8", errors="surrogateescape")
                route = self.BYTE if (e - s) == 1 else self.BOUNDARY
                self._emit(oss, b, text, s, e, vec, -1, route, record, raw=seg)
        return record

    def terminal_view(self, oss, b=0):
        """Fixed-capacity terminal-stream view over ``oss`` row ``b`` leaves.

        Returns a dict with ``what`` ``[1, cap, D]``, ``where`` ``[1, cap, 2]``
        (endpoint-sum keys), ``ids`` ``[1, cap]`` (PS part ids, -1 = byte
        fallback), ``mask`` ``[1, cap]`` bool (live slots), and ``len``
        ``[1]`` (live terminal count). This is a source-agnostic view, not a
        new durable object.
        """
        sl = slice(b, b + 1)
        return {
            "what": oss._buffer[sl].clone(),
            "where": oss._span_where[sl].clone(),
            "ids": oss._part_id[sl].clone(),
            "mask": oss.live_mask()[sl].clone(),
            "len": oss._depth[sl].clone(),
        }

    # -- Reverse PS synthesis (Phase 8) --------------------------------

    def synthesize(self, record):
        """Exact-replay reverse synthesis: reconstruct the surface from an
        analysis ``record`` (route metadata + canonical span text) by
        concatenating the terminals in span order. This is the
        ``boundary``/``stop`` synthesis direction -- spaces and affixes are
        replayed as the terminals they were analyzed into, not from any
        tokenizer state.

        When the terminals carry their exact source bytes (``"raw"`` -- the
        routed analyzer), reconstruction joins the BYTES and decodes once, so
        a multi-byte glyph split across byte terminals is reassembled exactly
        rather than interpolated as ``U+FFFD`` mojibake. Records without
        ``"raw"`` (the compatibility analyzer) fall back to the text join."""
        rec = sorted(record, key=lambda r: r["start"])
        if rec and all("raw" in r for r in rec):
            return b"".join(r["raw"] for r in rec).decode(
                "utf-8", errors="surrogateescape")
        return "".join(r["text"] for r in rec)

    def synthesize_tree(self, node, marker_resolver=None):
        """Generative reverse synthesis from an operator-prefixed tree.

        ``node`` is either ``("leaf", surface_text)`` or
        ``("op", grammar_layer, child, ...)``. The operator's surface marker
        is realized by :meth:`GrammarLayer.emit` and placed per its
        :class:`SurfaceSchema`: T1 affix = ``"marker arg"``; T2/T3 with a
        bound marker = ``"left marker right"`` (the marker is LEARNED, not a
        grammar token); T4 / no-marker = bare juxtaposition ``"left right"``.

        ``emit`` returns a marker PS codebook id, so replay resolves it to
        canonical surface via ``marker_resolver(marker_id) -> str`` (the PS
        store / marker codebook) and NEVER interpolates an opaque id as
        literal output (spec §9 carry-forward concern). A marker that is
        already surface text is placed directly.
        """
        if not isinstance(node, tuple) or node[0] == "leaf":
            return node[1] if isinstance(node, tuple) else str(node)
        _, op, *children = node
        parts = [self.synthesize_tree(c, marker_resolver=marker_resolver)
                 for c in children]
        schema = op.surface_schema
        marker = self._resolve_marker(op.emit(), marker_resolver)
        if schema.template_id == "T1":
            return f"{marker} {parts[0]}".strip() if marker else parts[0]
        if schema.has_marker and marker and len(parts) == 2:
            return f"{parts[0]} {marker} {parts[1]}"
        return " ".join(parts)

    @staticmethod
    def _resolve_marker(marker, marker_resolver):
        """Resolve an ``emit()``ed marker to canonical surface text.

        A string marker is already surface and is used directly. An opaque
        PS codebook id is mapped through ``marker_resolver`` (the PS store /
        marker codebook). Absent a resolver, an opaque id degrades to no
        marker (bare juxtaposition) rather than leaking the id into the
        surface -- replay never interpolates an opaque marker id (spec §9).
        """
        if marker is None:
            return None
        if isinstance(marker, str):
            return marker
        if marker_resolver is not None:
            resolved = marker_resolver(marker)
            return resolved if isinstance(resolved, str) else None
        return None


def soft_operator_compose(dist, left, right=None, *, classes=None):
    """Operator-superposition composition (Phase 9): the weighted sum of
    each operation's ``compose`` of the operands under the distribution
    ``dist`` (``{op_name: weight}``).

    A one-hot distribution reduces to that operation's hard compose, so the
    typed grammar is preserved; a spread distribution superposes operators
    -- the mechanism that lets STM discriminate ``A AND B`` from ``A OR B``
    by a soft slot-0 operator superposition rather than a hard
    part-of-speech grammar. ``classes`` defaults to ``GRAMMAR_LAYER_CLASSES``;
    unary operators ignore ``right``.
    """
    if classes is None:
        from Language import GRAMMAR_LAYER_CLASSES as classes
    total = float(sum(dist.values()))
    if total <= 0.0:
        raise ValueError("soft_operator_compose: distribution sums to 0")
    out = None
    for name, weight in dist.items():
        w = float(weight)
        if w == 0.0:
            continue
        cls = classes.get(name)
        if cls is None:
            continue
        op = cls()
        y = op.compose(left) if op.arity == 1 else op.compose(left, right)
        contrib = (w / total) * y
        out = contrib if out is None else out + contrib
    return out
