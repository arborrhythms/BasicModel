"""PerceptualSpace meronymic analyzer (PS forward analysis / PS reverse
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
    ``len``) the PS-to-SS binding consumes. ``.where`` is the endpoint-sum
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

    def _emit(self, oss, b, text, start, end, vec, part_id, route_id, record):
        """Push one terminal span to the ObjectSubSpace with endpoint-sum
        ``.where`` and append its host-side replay record."""
        oss.push(b, vec, part_id=int(part_id), span_start=int(start),
                 span_end=int(end), span_where=self.where.encode(start, end),
                 route_id=int(route_id))
        record.append({"text": text, "start": int(start), "end": int(end),
                       "part_id": int(part_id), "route": int(route_id)})

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
        concatenating the terminal surfaces in span order. This is the
        ``boundary``/``stop`` synthesis direction -- spaces and affixes are
        replayed as the terminals they were analyzed into, not from any
        tokenizer state."""
        return "".join(r["text"]
                       for r in sorted(record, key=lambda r: r["start"]))

    def synthesize_tree(self, node):
        """Generative reverse synthesis from an operator-prefixed tree.

        ``node`` is either ``("leaf", surface_text)`` or
        ``("op", grammar_layer, child, ...)``. The operator's surface marker
        is realized by :meth:`GrammarLayer.emit` and placed per its
        :class:`SurfaceSchema`: T1 affix = ``"marker arg"``; T2/T3 with a
        bound marker = ``"left marker right"`` (the marker is LEARNED, not a
        grammar token); T4 / no-marker = bare juxtaposition ``"left right"``.
        """
        if not isinstance(node, tuple) or node[0] == "leaf":
            return node[1] if isinstance(node, tuple) else str(node)
        _, op, *children = node
        parts = [self.synthesize_tree(c) for c in children]
        schema = op.surface_schema
        marker = op.emit()
        if schema.template_id == "T1":
            return (f"{marker} {parts[0]}".strip()
                    if marker is not None else parts[0])
        if schema.has_marker and marker is not None and len(parts) == 2:
            return f"{parts[0]} {marker} {parts[1]}"
        return " ".join(parts)


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
