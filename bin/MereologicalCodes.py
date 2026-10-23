"""One index pairs a perceptual form with a conceptual meaning.

The cached representation is [form | meaning]. Perception reconstruction alone
writes native prototypes and evidence; the sentence read detaches them. Order-zero
meaning is the detached occurrence-context mean of fixed sentence identities,
on the for pole, with no against evidence. Letters need no conceptual row.
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
        from util import TheXMLConfig
        available = self.code_width - self.percept_event_width
        self.reserved_pairs = available // 2
        requested = int(TheXMLConfig.space('ConceptualSpace', 'meaningWidth', available))
        self.context_width = min(requested, available)
        if requested < 0 or self.context_width % 2:
            raise ValueError('the meaning complement holds complete pole pairs')
        self.meaning_pairs = self.context_width // 2
        self.meaning_ones = min(self.meaning_pairs, 6 if self.meaning_pairs > 64 else 3)
        self._context = None
        self._centroid_cache = None

    @torch.no_grad()
    def centroid_forms(self):
        from Spaces import _concept_alloc_of
        from SymbolCentroid import place
        model = getattr(self.owner, '_model', None)
        identity = getattr(getattr(getattr(model, 'perceptualSpace', None), 'percept_store', None), 'identity', None)
        if identity is None:
            return {}
        rows = sorted(_concept_alloc_of(self.owner).layer()._tensor_row_keys)
        words = {row: raw for row in rows if (raw := self.owner.word_surface_for_row(row)) in identity.words}
        revision = (identity.revision, tuple(words.items()))
        if self._centroid_cache is None or revision != self._centroid_cache[0]:
            forms = {row: identity.form(raw) for row, raw in words.items()}
            parts = {row: identity.part_atoms(raw) for row, raw in words.items()}
            placed, report = place(forms, parts)
            self._centroid_cache = revision, placed, report
        return self._centroid_cache[1]

    def centroid_audit(self):
        self.centroid_forms()
        return {} if self._centroid_cache is None else self._centroid_cache[2]

    @torch.no_grad()
    def occurrence_terms(self, b=0):
        """Occurrence membership supplies n and detached conceptual meaning M.

        A reference witnesses membership even in an unverified sentence.
        Its truth poles state a different fact and are never changed here.
        """
        store = self.owner._closed_clause_store()
        if store is None or not self.context_width:
            return {}
        from PartRead import occurrence_roots, occurrence_codes
        records = occurrence_roots(self.owner, store, b)
        terms = occurrence_memberships(self.owner, store, records=records)
        result = {}
        from MeaningCodes import defined_code
        codes = occurrence_codes(records, self.meaning_pairs, self.meaning_ones, store.slots)
        for code, rows in terms.items():
            rows = sorted(rows)
            if not rows:
                continue
            weight = torch.stack([records[row]['weight'] for row in rows])
            roots = torch.stack([codes[row] for row in rows])
            positive = (roots * weight[:, None]).sum(0) / weight.sum()
            positive = defined_code(positive, len(rows))
            result[code] = (len(rows), torch.cat((positive, torch.zeros_like(positive))).detach())
        return result

    def begin_forward(self):
        # Both walk paths use the same pre-forward durable context; the
        # current closing becomes available on the next forward only.
        store = self.owner._closed_clause_store()
        self._contexts = {b: self.occurrence_terms(b) for b in range(max(1, 0 if store is None else store.stream.shape[1]))}
        self._context = self._contexts[0]
        self._unseen_context = self.occurrence_terms(len(self._contexts))
        self._form_context = self.centroid_forms()
        from Spaces import _concept_alloc_of
        cb = self.owner.similarity_codebook
        rows = sorted(_concept_alloc_of(self.owner).layer()._tensor_row_keys)
        with torch.no_grad():
            from IdentityRows import cache_rows
            retained = cache_rows(self.owner)
            clear = torch.ones(len(cb.W), dtype=torch.bool, device=cb.W.device)
            clear[list(retained)] = False
            cb.W[clear] = 0
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
    def containment_audit(self, *, before=None, after=None):
        """Audit order-zero form containment through positive-part postings.

        Optional row-to-form snapshots expose both sides of the future centroid
        placement pass. This observer does not project or change any code.
        Equal part sets are checked in both directions; self pairs are omitted.
        """
        from Spaces import _concept_alloc_of
        rows = sorted(row for row in _concept_alloc_of(self.owner).layer()._tensor_row_keys
                      if self.owner._order0_inventory_row(row))
        parts = {row: {pid for tower, pid, net in edges if tower == 0 and net > 0}
                 for row, edges in self._definitions(rows).items()}
        postings = {}
        for row, members in parts.items():
            for pid in members:
                postings.setdefault(pid, set()).add(row)
        pairs = []
        for row, members in parts.items():
            extents = sorted((postings[pid] for pid in members), key=len)
            containers = set.intersection(*extents) if extents else set(rows)
            pairs.extend((row, other) for other in sorted(containers) if row != other)
        values = dict(zip(rows, self.derive(rows)[..., :self.percept_width]))
        def report(snapshot):
            snapshot = values if snapshot is None else snapshot
            count = coordinates = 0
            largest = 0.
            for contained, container in pairs:
                violation = (snapshot[contained][:self.percept_width]-
                             snapshot[container][:self.percept_width]).clamp_min(0)
                positive = int(violation.gt(0).sum())
                count += int(positive > 0)
                coordinates += positive
                largest = max(largest, float(violation.max()))
            return dict(pairs=len(pairs), violating_pairs=count,
                        coordinates=coordinates, largest=largest)
        return dict(order=0, rows=len(rows), postings=len(postings),
                    self_pairs=False, before=report(before), after=report(after))

    @torch.no_grad()
    def derive(self, indices, b=0):
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
        model = getattr(owner, '_model', None)
        identity = getattr(getattr(getattr(model, 'perceptualSpace', None), 'percept_store', None), 'identity', None)
        if identity is not None:
            # New lexical witnesses can add nested parts during the stem.
            # Form identity must follow that vocabulary revision immediately;
            # occurrence meaning and adjacency stay frozen across paired walks.
            placed = self.centroid_forms()
            for row in unique:
                raw = owner.word_surface_for_row(row)
                if raw in identity.words:
                    # Identification is immutable. Learned support changes a
                    # reading's evidence, never the named word's sparse key.
                    out[row_index[row], :identity.width] = placed.get(row, identity.form(raw)).to(out)
        contexts = getattr(self, "_contexts", {0:self._context})
        context = contexts.get(b, getattr(self, '_unseen_context', None))
        if context is None:
            context = self.occurrence_terms(b)
        for row in unique:
            if row in context and self.context_width:
                mean = context[row][1].detach().to(out)
                start, k = self.percept_event_width, self.meaning_pairs
                out[row_index[row], start:start+k] = mean[:k]
                out[row_index[row], start+self.reserved_pairs:start+self.reserved_pairs+k] = mean[k:]
        from IdentityRows import cache_rows
        retained = set(cache_rows(owner))
        for row in unique:
            if row in retained:
                out[row_index[row]] = cached[row]
        result = out[torch.tensor([row_index[r] for r in requested], device=out.device, dtype=torch.long)]
        return result.reshape(*shape, self.code_width)


@torch.no_grad()
def occurrence_memberships(owner, store, *, records=None):
    """Map lexical seats to transient context rows rebuilt from Part edges."""
    from PartRead import occurrence_roots, word_rows
    if records is None:
        records = occurrence_roots(owner, store)
    terms = {}
    for row, record in enumerate(records):
        for code in word_rows(owner, store.parts, record['root']):
            terms.setdefault(code, set()).add(row)
    return terms
