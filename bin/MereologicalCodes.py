"""One index pairs a perceptual form with a conceptual meaning.

The cached representation is [form | meaning]. Perception reconstruction alone
writes native prototypes and evidence; the sentence read detaches them. Order-zero
meaning is the detached occurrence-context mean. Letters need no conceptual row.
Bootstrap from conceptual wholes is deferred to the operators update.
"""
import torch
from torch import nn
from torch.nn import functional as F


class MereologicalCodes(nn.Module):
    def __init__(self, owner, initial, *, percept_width,
                 percept_event_width):
        super().__init__()
        object.__setattr__(self, 'owner', owner)
        self.percept_width = int(percept_width)
        self.percept_event_width = int(percept_event_width)
        self.code_width = int(initial.shape[-1])
        if not 0 < self.percept_width <= self.percept_event_width <= self.code_width:
            raise ValueError('paired representation must contain the native form block')
        self.context_width = self.code_width - self.percept_event_width
        self._context = None

    @torch.no_grad()
    def occurrence_terms(self):
        """Occurrence membership supplies n and detached conceptual meaning M.

        A reference witnesses membership even in an unverified sentence.
        Its truth poles state a different fact and are never changed here.
        """
        store = self.owner._closed_clause_store()
        if store is None or not len(store):
            return {}
        terms = {}
        now = float(store._next_ts) - 1
        terms = occurrence_memberships(self.owner, store)
        result = {}
        width = self.code_width
        for code, rows in terms.items():
            rows = [r for r in sorted(rows) if int(store.rel_type[r]) == store.REL_NONE]
            if not rows:
                continue
            idx = torch.tensor(rows, dtype=torch.long, device=store.slots.device)
            weight = 1 / (1 + (now-store.timestamp[idx]).clamp_min(0))
            roots = store.slots[idx, 0, :width].detach()
            if roots.shape[-1] < width:
                roots = F.pad(roots, (0, width-roots.shape[-1]))
            roots = roots[:, self.percept_event_width:]
            result[code] = (len(rows), ((roots * weight[:, None]).sum(0) / weight.sum()).detach().clone())
        return result

    def begin_forward(self):
        # Both walk paths use the same pre-forward durable context; the
        # current closing becomes available on the next forward only.
        self._context = self.occurrence_terms()
        from Spaces import _concept_alloc_of
        cb = self.owner.similarity_codebook
        rows = sorted(_concept_alloc_of(self.owner).layer()._tensor_row_keys)
        with torch.no_grad():
            cb.W.zero_()
            if rows:
                cb.lookup_rows(torch.tensor(rows, device=cb.W.device))

    def context_audit(self):
        context = self.occurrence_terms() if self._context is None else self._context
        return [dict(row=row, occurrences=n, context_width=self.context_width,
                     context_norm=float(mean.norm()), bootstrap_learning=False)
                for row, (n, mean) in sorted(context.items())]

    @torch.no_grad()
    def support_audit(self, rows):
        rows = list(map(int, rows))
        codes = self.derive(rows)[..., :self.percept_width].abs()
        records = []
        for row, code in zip(rows, codes):
            nonzero = code != 0
            text = self.owner.word_surface_for_row(row)
            records.append(dict(row=row, word=None if text is None else text.decode('utf8'),
                dimension=self.percept_width, percept_event_width=self.percept_event_width,
                nonzero_coordinates=int(nonzero.sum()),
                nonzero_fraction=int(nonzero.sum())/self.percept_width,
                minimum_absolute_value=float(code.min()),
                minimum_nonzero_absolute_value=(float(code[nonzero].min()) if nonzero.any() else None)))
        return records

    def _native(self, tower):
        model = self.owner._model
        space = model.perceptualSpace if tower == 0 else model.wholeSpace
        return space.subspace.what

    @torch.no_grad()
    def _definitions(self, rows):
        """Net 11b evidence per native part/property, never the both corner."""
        from Spaces import _concept_alloc_of
        layer = _concept_alloc_of(self.owner).layer()
        definitions = {int(row): {} for row in rows}
        for (row, column), pos in layer.features._index.items():
            if row not in definitions:
                continue
            tower, pole = (column // 2) % 2, column % 2
            group = layer.feature_groups.get((row, column), (column // 4,))
            for pid in dict.fromkeys(group):
                poles = definitions[row].setdefault((tower, pid), [0., 0.])
                poles[pole] += float(layer.features.values[pos].detach())
        return {row: [(tower, pid, max(0., poles[0]-poles[1]))
                       for (tower, pid), poles in sorted(edges.items())]
                for row, edges in definitions.items()}

    @torch.no_grad()
    def _interval(self, edges):
        like = self.owner.similarity_codebook.W
        bounds, weights, extrema, ids = [], [], [], []
        for tower in (0, 1):
            selected = [(pid, d) for t, pid, d in edges if t == tower and d > 0]
            ids.append([pid for pid, _ in selected])
            if not selected:
                bounds.append(like.new_full((self.percept_width,), float(tower)))
                weights.append(0.)
                extrema.append(None)
                continue
            codes = self._native(tower).lookup_rows(torch.tensor(ids[-1], device=like.device)).detach()
            if codes.shape[-1] != self.percept_width:
                raise RuntimeError('native towers must share the form coordinates')
            d = like.new_tensor([d for _, d in selected])
            values = codes if tower == 0 else 1-d[:, None]*(1-codes)
            bound, extreme = values.max(0) if tower == 0 else values.min(0)
            bounds.append(bound); weights.append(float(d.sum())); extrema.append(extreme)
        return bounds, weights, extrema, ids

    @torch.no_grad()
    def room_report(self, margin=0.):
        from Spaces import _concept_alloc_of
        rows = sorted(_concept_alloc_of(self.owner).layer()._tensor_row_keys)
        violations = []
        for row, edges in self._definitions(rows).items():
            (lower, upper), _, _, ids = self._interval(edges)
            if ids[0] or ids[1]:
                violations.append((lower-upper+float(margin)).relu())
        values = torch.cat(violations) if violations else self.owner.similarity_codebook.W.new_zeros(0)
        return dict(count=int((values > 0).sum()),
                    largest=float(values.max()) if values.numel() else 0.)

    @torch.no_grad()
    def project_room(self, margin=0.):
        """One deterministic pass over concept/coordinate; no loss or gradient.

        Only the minimal whole moves, by the full violation. Parts never
        shrink to fit their types. Report any residual after evidence/clipping.
        """
        from Spaces import _concept_alloc_of
        before = self.room_report(margin)
        rows = sorted(_concept_alloc_of(self.owner).layer()._tensor_row_keys)
        for row, edges in self._definitions(rows).items():
            (lower, upper), _, extrema, ids = self._interval(edges)
            if not (ids[0] or ids[1]):
                continue
            violation = (lower-upper+float(margin)).relu()
            for coordinate in (violation > 0).nonzero().flatten().tolist():
                if not ids[1]:
                    continue  # the absent whole is the fixed upper boundary 1
                pid = ids[1][int(extrema[1][coordinate])]
                value = self._native(1).W[pid, coordinate]
                value.copy_((value.clamp(0, 1)+violation[coordinate]).clamp(0, 1))
        report = dict(margin=float(margin), before=before, after=self.room_report(margin))
        self.last_room_projection = report
        return report

    @torch.no_grad()
    def derive(self, indices):
        owner = self.owner
        cached = owner.similarity_codebook.W
        indices = torch.as_tensor(indices, dtype=torch.long, device=cached.device)
        shape = indices.shape
        requested = indices.flatten().cpu().tolist()
        unique = list(dict.fromkeys(requested))
        row_index = {row: i for i, row in enumerate(unique)}
        out = cached.new_zeros(len(unique), self.code_width)
        for row, edges in self._definitions(unique).items():
            if not edges:
                continue  # an unallocated cache row has neither form nor meaning
            (lower, _upper), _, _, _ = self._interval(edges)
            out[row_index[row], :self.percept_width] = lower
        context = self.occurrence_terms() if self._context is None else self._context
        for row in unique:
            if row in context and self.context_width:
                out[row_index[row], self.percept_event_width:] = context[row][1].detach().to(out)
        result = out[torch.tensor([row_index[r] for r in requested], device=out.device, dtype=torch.long)]
        return result.reshape(*shape, self.code_width)


@torch.no_grad()
def occurrence_memberships(owner, store):
    """Use both native slot references and the durable leaf-code postings."""
    terms = {}
    for (code, _role), rows in store._leaf_postings.items():
        terms.setdefault(int(code), set()).update(rows)
    for row, references in enumerate(store.refs[:len(store)].tolist()):
        if int(store.rel_type[row]) == store.REL_DEF:
            continue
        for cid in set(references):
            code = owner._csw_row_of(cid) if cid > 0 else None
            if code is None and cid > 0:
                definitions = owner._definition_index()
                objects = () if definitions is None else definitions.objects(cid)
                code = owner._csw_row_of(objects[0]) if len(objects) == 1 else None
            if code is not None:
                terms.setdefault(int(code), set()).add(row)
    return terms


@torch.no_grad()
def conduct_occurrences(owner, surface, *, active_rows=None):
    """Two hops, word → existing row → constituents, conserving energy."""
    store = owner._closed_clause_store()
    spread = float(getattr(owner, '_priming_spread', 0.) or 0.)
    if store is None or not len(store) or spread <= 0:
        return surface
    B, V = surface.shape
    active = torch.ones(B, dtype=torch.bool, device=surface.device) if active_rows is None else active_rows.to(surface.device)
    pairs = sorted({(int(code), int(row)) for code, rows in occurrence_memberships(owner, store).items()
                    if 0 <= code < V for row in rows if int(store.rel_type[row]) != store.REL_DEF})
    if not pairs:
        return surface
    words, rows = torch.tensor(pairs, device=surface.device, dtype=torch.long).T
    recency = 1 / (1 + (float(store._next_ts)-1-store.timestamp[rows]).clamp_min(0)).to(surface)
    degree = surface.new_zeros(V).index_add_(0, words, recency)
    flow = spread * (surface[:, words]-1) * recency / degree[words]
    flow *= active[:, None]
    row_energy = surface.new_zeros(B, len(store)).index_add_(1, rows, flow)
    # A row's constituents inherit the energy it received. Using the same
    # incidence set makes both hops explicit and avoids a word-code attraction.
    row_degree = surface.new_zeros(len(store)).index_add_(0, rows, recency)
    returning = row_energy[:, rows] * recency / row_degree[rows]
    before = surface.clone()
    surface.index_add_(1, words, -flow)
    surface.index_add_(1, words, returning)
    surface.clamp_min_(0.)
    object.__setattr__(owner, '_last_occurrence_priming', dict(
        rows=len(set(rows.tolist())), edges=len(pairs),
        activated_competitors=int(((before <= 1) & (surface > 1)).sum()),
        row_energy=row_energy.detach()))
    return surface
