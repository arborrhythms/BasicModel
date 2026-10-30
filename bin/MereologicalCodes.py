"""One index pairs a perceptual form with a conceptual meaning.

The cached representation is [form | meaning]. Perception reconstruction alone
writes native prototypes and evidence; the sentence read detaches them. Order-zero
meaning is placed at kept closes from the registered priming surface,
on the for pole, with no against evidence. Letters need no conceptual row.
"""
import torch
from torch import nn
from torch.nn import functional as F


def placement_step(q, context, weights, negatives, *, rate=.1, negative_weight=1.):
    """One meet-ratio subgradient step, in the positive conceptual coordinates."""
    gradient = torch.zeros_like(q)
    if len(context):
        gradient += ((q < context).to(q) * (weights / context.sum(-1).clamp_min(1e-12))[:, None]).sum(0)
    if len(negatives):
        gradient -= negative_weight * ((q < negatives).to(q) / negatives.sum(-1).clamp_min(1e-12)[:, None]).sum(0)
    return F.normalize((q + rate * gradient).clamp_min(0), dim=-1)


class MereologicalCodes(nn.Module):
    def __init__(self, owner, initial, *, percept_width,
                 percept_event_width, identity_width=None, property_width=0):
        super().__init__()
        object.__setattr__(self, 'owner', owner)
        self.percept_width = int(percept_width)
        self.property_width = int(property_width)
        self.identity_width = (self.percept_width - self.property_width
                               if identity_width is None else int(identity_width))
        self.percept_event_width = int(percept_event_width)
        self.code_width = int(initial.shape[-1])
        if (self.identity_width < 0 or self.property_width < 0
                or self.identity_width + self.property_width != self.percept_width):
            raise ValueError('percept_width must equal identity_width + property_width')
        if not 0 < self.percept_width <= self.percept_event_width <= self.code_width:
            raise ValueError('paired representation must contain the native form block')
        if self.property_width:
            self.register_buffer('_form_layout', torch.tensor(
                [self.identity_width, self.property_width, self.percept_event_width],
                dtype=torch.long, device=initial.device))
        from util import TheXMLConfig
        available = self.code_width - self.percept_event_width
        self.reserved_pairs = available // 2
        requested = int(TheXMLConfig.space('ConceptualSpace', 'meaningWidth', available))
        self.context_width = min(requested, available)
        if requested < 0 or self.context_width % 2:
            raise ValueError('the meaning complement holds complete pole pairs')
        self.meaning_pairs = self.context_width // 2
        self.meaning_ones = min(self.meaning_pairs, 6 if self.meaning_pairs > 64 else 3)
        # One direction, count, dirty flag and immutable minted-content key per
        # constructed codebook seat; meanings reserve half their bipolar width.
        count = len(initial)
        self.register_buffer('placement_version', torch.tensor(1, dtype=torch.long, device=initial.device))
        self.register_buffer('placement', initial.new_zeros(count, self.meaning_pairs))
        self.register_buffer('placement_witnesses', torch.zeros(count, dtype=torch.long, device=initial.device))
        self.register_buffer('placement_derived', torch.zeros(count, dtype=torch.bool, device=initial.device))
        self.register_buffer('placement_valid', torch.zeros(count, dtype=torch.bool, device=initial.device))
        self.register_buffer('placement_dirty', torch.zeros(count, dtype=torch.bool, device=initial.device), persistent=False)
        self.register_buffer('placement_keys', torch.zeros(count, 32, dtype=torch.uint8, device=initial.device))
        self.register_buffer('placement_keyed', torch.zeros(count, dtype=torch.bool, device=initial.device))
        # A cached form and two validity bits per seat replace unchanged-row scans.
        self.register_buffer('_forms', initial.new_zeros(count, self.percept_width), persistent=False)
        self.register_buffer('_form_valid', torch.zeros(count, dtype=torch.bool, device=initial.device), persistent=False)
        self.register_buffer('_allocated', torch.zeros(count, dtype=torch.bool, device=initial.device), persistent=False)
        # One tenth of a meet-ratio subgradient, balanced by five unit-weight
        # negatives: bounded close work, with negatives as the collapse brake.
        self.placement_rate = float(TheXMLConfig.space('ConceptualSpace', 'placementRate', .1))
        self.negative_count = int(TheXMLConfig.space('ConceptualSpace', 'placementNegatives', 5))
        self.negative_weight = float(TheXMLConfig.space('ConceptualSpace', 'placementNegativeWeight', 1.))
        import math
        if (not math.isfinite(self.placement_rate) or self.placement_rate < 0
                or self.negative_count < 0 or not math.isfinite(self.negative_weight) or self.negative_weight < 0):
            raise ValueError('placement rate and negative controls must be finite and nonnegative')
        self._form_revision = self._inventory_revision = None
        self.last_rederived_rows = self.form_rows_derived = 0

    @torch.no_grad()
    def remember_content(self, row, value):
        """Pin a minted concept's content key before its meaning cache is cleared."""
        import hashlib
        value = value.detach().clone().to(device='cpu', dtype=torch.float32)
        value[self.percept_width:self.percept_event_width] = 0
        key = hashlib.sha256(b'placed-concept-v1\0' + value.contiguous().numpy().astype('<f4').tobytes()).digest()
        self.placement_keys[row].copy_(self.placement_keys.new_tensor(list(key)))
        self.placement_keyed[row] = True
        return key

    def content_key(self, row):
        raw = self.owner.word_surface_for_row(row)
        if isinstance(raw, bytes):
            return raw
        if not bool(self.placement_keyed[row]):
            self.remember_content(row, self.owner.similarity_codebook.W[row])
        return bytes(self.placement_keys[row].cpu().tolist())

    def reading_key(self, reading):
        from Occurrence import sentence_key
        rows = reading.word_rows.detach().reshape(-1).tolist()
        identities = getattr(reading, 'concept_ids', None)
        if identities is not None:
            rows = [row if row >= 0 else self.owner._csw_row_of(int(cid))
                    for row, cid in zip(rows, identities.reshape(-1).tolist())]
        return sentence_key([self.content_key(int(row)) for row in rows if row is not None and row >= 0])

    def placement_step(self, q, context, weights, negatives):
        """One subgradient step of the meet-ratio objective; negatives brake collapse."""
        return placement_step(q, context, weights, negatives,
                              rate=self.placement_rate, negative_weight=self.negative_weight)

    def _negatives(self, row, excess, placed, key):
        import hashlib
        eligible = ((excess == 0) & placed).nonzero().flatten()
        eligible = eligible[eligible != row]
        generator = torch.Generator(device='cpu')
        generator.manual_seed(int.from_bytes(hashlib.sha256(key).digest()[:8], 'little'))
        order = torch.randperm(len(eligible), generator=generator)[:self.negative_count].to(eligible.device)
        return eligible[order]

    @torch.no_grad()
    def place_rows(self, rows, context, key):
        from MeaningCodes import identity_code
        if not self.meaning_pairs:
            return
        rows = list(dict.fromkeys(map(int, rows)))
        if any(not 0 <= row < len(self.placement) for row in rows):
            raise RuntimeError('placement row exceeds the constructed inventory')
        if context.shape != self.placement_witnesses.shape:
            raise ValueError('placement context differs from the constructed surface')
        # One frozen context for the close: result is independent of row order.
        codes = self.placement.detach().clone()
        placed = self.placement_witnesses > 0
        for row in rows:
            if bool(self.placement_derived[row]):
                continue
            q = codes[row] if placed[row] else F.normalize(identity_code(
                self.content_key(row), self.meaning_pairs, self.meaning_ones, like=codes), dim=0)
            selected = (context > 0) & placed
            selected[row] = False
            negatives = self._negatives(row, context, placed, key)
            self.placement[row].copy_(self.placement_step(q, codes[selected], context[selected], codes[negatives]))
            self.placement_witnesses[row] += 1
            self.placement_dirty[row] = True

    @torch.no_grad()
    def derive_class_placements(self):
        """Place raised classes at their members' evidence-weighted centroids."""
        from Priming import part_edges
        edges = part_edges(self.owner)
        rows = self.placement_derived.nonzero().flatten().tolist()
        allocator = self.owner._concept_allocator
        # Members may themselves be classes. Lower orders are derived first.
        rows.sort(key=lambda row: allocator.order_of(self.owner.concept_id_at_row(row)))
        threshold = self.owner._geometric_radius.cos()
        for row in rows:
            members = weights = None
            if edges is not None:
                src, dst, weight = edges
                selected = (dst == row) & (weight > 0)
                members, weights = src[selected], weight[selected]
                # A deferred/unplaced member cannot supply a location yet.
                if not bool((self.placement_witnesses[members] > 0).all()):
                    members, weights = members[:0], weights[:0]
            count = 0 if members is None else len(members)
            self.placement_witnesses[row] = count
            if count:
                codes = self.placement[members]
                centroid = F.normalize((codes * weights[:, None]).sum(0), dim=0)
                self.placement[row].copy_(centroid)
                self.placement_valid[row] = ((codes @ centroid) >= threshold).all()
            else:
                self.placement[row].zero_()
                self.placement_valid[row] = False
            self.placement_dirty[row] = True

    @torch.no_grad()
    def observe_close(self, reading, batch=0):
        """Place the registered sentence once, on its kept close only."""
        if reading is None:
            return None
        context = self.owner.placement_context(batch)
        rows = self.owner._sentence_seen_rows[int(batch)]
        self.place_rows(rows[rows >= 0].tolist(), context, self.reading_key(reading))
        return context

    def placement_extras(self):
        return dict(version=1, placement=self.placement.detach().cpu().clone(),
                    witnesses=self.placement_witnesses.detach().cpu().clone(),
                    derived=self.placement_derived.detach().cpu().clone(),
                    valid=self.placement_valid.detach().cpu().clone(),
                    keys=self.placement_keys.detach().cpu().clone(),
                    keyed=self.placement_keyed.detach().cpu().clone())

    @torch.no_grad()
    def load_placement_extras(self, saved):
        if not isinstance(saved, dict) or saved.get('version') != 1:
            raise ValueError('item 3.0 placement checkpoint version 1 is required')
        pairs = (('placement', self.placement), ('witnesses', self.placement_witnesses),
                 ('keys', self.placement_keys), ('keyed', self.placement_keyed))
        # Earlier version-1 checkpoints predate class placement. Their native
        # raise records supply the new flags; no constructed column is resized.
        flags = (('derived', self.placement_derived), ('valid', self.placement_valid))
        pairs += tuple((key, target) for key, target in flags if key in saved)
        for key, target in pairs:
            value = saved.get(key)
            if not torch.is_tensor(value) or value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError('placement checkpoint differs from constructed column: '+key)
        self._validate_placement(saved['placement'], saved['witnesses'])
        for key, target in pairs:
            target.copy_(saved[key])
        if 'derived' not in saved or 'valid' not in saved:
            self._restore_class_placements()
        self.placement_dirty.fill_(True)
        self._form_valid.zero_()

    @staticmethod
    def _validate_placement(codes, witnesses):
        norms = codes.norm(dim=-1)
        tolerance = max(1e-5, 8 * torch.finfo(codes.dtype).eps)
        if (not bool(torch.isfinite(codes).all()) or bool((codes < 0).any())
                or bool((witnesses < 0).any())
                or bool(((norms > 1e-6) & ((norms - 1).abs() > tolerance)).any())
                or bool((codes[witnesses == 0] != 0).any())):
            raise ValueError('invalid placement directions or witness counts')

    def _load_from_state_dict(self, state, prefix, *args, **kwargs):
        if self.property_width:
            layout = state.get(prefix+'_form_layout')
            if layout is None or not torch.equal(layout.to(self._form_layout), self._form_layout):
                raise ValueError('checkpoint form subspaces differ from the constructed layout')
        if int(state.get(prefix+'placement_version', -1)) != 1:
            raise ValueError('item 3.0 placement checkpoint version 1 is required')
        for name in ('placement', 'placement_witnesses', 'placement_keys', 'placement_keyed'):
            value = state.get(prefix+name)
            expected = getattr(self, name)
            if value is None or value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError('placement checkpoint differs from constructed column: '+name)
        self._validate_placement(state[prefix+'placement'], state[prefix+'placement_witnesses'])
        legacy = any(prefix+name not in state for name in ('placement_derived', 'placement_valid'))
        for name in ('placement_derived', 'placement_valid'):
            value = state.get(prefix+name)
            if value is None:
                state[prefix+name] = torch.zeros_like(getattr(self, name))
            elif value.shape != self.placement_witnesses.shape or value.dtype != torch.bool:
                raise ValueError('placement checkpoint differs from constructed column: '+name)
        super()._load_from_state_dict(state, prefix, *args, **kwargs)
        if legacy:
            self._restore_class_placements()
        self.placement_dirty.fill_(True)
        self._form_valid.zero_()
        self._form_revision = None

    @torch.no_grad()
    def _restore_class_placements(self):
        self.placement_derived.zero_()
        self.placement_valid.zero_()
        from Priming import part_edges
        self.owner._priming_edge_revision = None
        edges = part_edges(self.owner)
        if edges is not None:
            self.placement_derived[edges[1]] = True
        self.derive_class_placements()

    def _form_sources(self):
        from Spaces import _concept_alloc_of
        layer = _concept_alloc_of(self.owner).layer()
        model = getattr(self.owner, '_model', None)
        native = []
        for name in ('perceptualSpace', 'wholeSpace'):
            space = getattr(model, name, None)
            cb = getattr(getattr(space, 'subspace', None), 'what', None)
            W = getattr(cb, 'W', None)
            native.append(None if W is None else (id(W), W._version))
        identity = getattr(getattr(getattr(model, 'perceptualSpace', None), 'percept_store', None), 'identity', None)
        return (id(layer.features.values), None if layer.features.values is None else layer.features.values._version, len(layer.features._index),
                tuple(native), None if identity is None else identity.revision)

    def _invalidate_forms(self):
        revision = self._form_sources()
        if revision != self._form_revision:
            self._form_valid.zero_()
            self._form_revision = revision

    def begin_forward(self):
        """Refresh changed forms and placement only; paired trials share the cache."""
        self._invalidate_forms()
        from Spaces import _concept_alloc_of
        layer = _concept_alloc_of(self.owner).layer()
        revision = (id(layer._tensor_row_keys), len(layer._tensor_row_keys), getattr(layer, '_record_revision', 0))
        if revision != self._inventory_revision:
            self._allocated.zero_()
            rows = list(layer._tensor_row_keys)
            if rows:
                self._allocated[rows] = True
            self._inventory_revision = revision
        dirty = self._allocated & (self.placement_dirty | ~self._form_valid)
        rows = dirty.nonzero().flatten()
        self.last_rederived_rows = len(rows)
        if len(rows):
            self.owner.similarity_codebook.lookup_rows(rows)
        self.placement_dirty.zero_()

    def context_audit(self):
        return [dict(row=row, placement_witnesses=int(self.placement_witnesses[row]),
                     context_width=self.context_width, direction_norm=float(self.placement[row].norm()),
                     bootstrap_learning=False)
                for row in self.placement_witnesses.nonzero().flatten().tolist()]

    @torch.no_grad()
    def collapse_audit(self, rows, batch=0):
        from IdentityRows import meet_ratios
        excess = self.owner.placement_context(batch)
        placed = self.placement_witnesses > 0
        records = []
        for row in map(int, rows):
            if not bool(placed[row]):
                continue
            selected = (excess > 0) & placed
            selected[row] = False
            positives = meet_ratios(self.placement[row], self.placement[selected])[0]
            weights = excess[selected]
            negatives = self._negatives(row, excess, placed, b'placement-collapse-audit-v1')
            random = meet_ratios(self.placement[row], self.placement[negatives])[0]
            records.append(dict(row=row, context=float((positives*weights).sum()/weights.sum().clamp_min(1e-12)),
                                random=float(random.mean()) if len(random) else 0., negatives=negatives.tolist()))
        return dict(context=sum(r['context'] for r in records)/max(1,len(records)),
                    random=sum(r['random'] for r in records)/max(1,len(records)), rows=records)

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
    def _subspace_bounds(self, edges):
        like = self.owner.similarity_codebook.W
        bounds = []
        for tower in (0, 1):
            start, stop = ((0, self.identity_width) if tower == 0 else
                           (self.identity_width, self.percept_width))
            selected = [(pid, d) for t, pid, d in edges if t == tower and d > 0]
            if not selected or start == stop:
                bounds.append(like.new_full((stop-start,), float(tower)))
                continue
            ids = torch.tensor([pid for pid, _ in selected], device=like.device)
            codes = self._native(tower).lookup_rows(ids).detach()
            if codes.shape[-1] < self.percept_width:
                raise RuntimeError('native towers must share the form coordinates')
            codes = codes[:, start:stop]
            # Grouped feature records can overlap: summed net evidence may
            # exceed one, but the soft-meet presence remains a cube weight.
            d = like.new_tensor([d for _, d in selected]).clamp(0, 1)
            values = codes if tower == 0 else 1-d[:, None]*(1-codes)
            bounds.append(values.amax(0) if tower == 0 else values.amin(0))
        return tuple(bounds)

    @torch.no_grad()
    def containment_audit(self, *, before=None, after=None):
        """Observe both subspaces and their native containment, without writes."""
        from FormContainment import audit
        return audit(self, before=before, after=after)

    @torch.no_grad()
    def derive(self, indices, b=0):
        owner = self.owner
        cached = owner.similarity_codebook.W
        indices = torch.as_tensor(indices, dtype=torch.long, device=cached.device)
        shape = indices.shape
        requested = indices.flatten().cpu().tolist()
        unique = list(dict.fromkeys(requested))
        row_index = {row: i for i, row in enumerate(unique)}
        self._invalidate_forms()
        out = cached.new_zeros(len(unique), self.code_width)
        from Spaces import _concept_alloc_of
        alloc = _concept_alloc_of(owner)
        retained = set()
        for row in unique:
            cid = owner.concept_id_at_row(row)
            if (cid is not None and cid not in alloc.retired and
                    ('concept' in alloc.symbol_types.get(cid, ()) or cid in alloc.individuals
                     or cid in alloc.form_regions.values())):
                retained.add(row)
        needed = [row for row in unique if row not in retained and not bool(self._form_valid[row])]
        if needed:
            for row, edges in self._definitions(needed).items():
                self._forms[row].copy_(torch.cat(self._subspace_bounds(edges)))
                self._form_valid[row] = True
            self.form_rows_derived += len(needed)
        for row in unique:
            out[row_index[row], :self.percept_width] = (
                cached[row, :self.percept_width] if row in retained else self._forms[row])
            self._form_valid[row] = True
        from MeaningCodes import defined_code
        if unique and self.meaning_pairs:
            start, k = self.percept_event_width, self.meaning_pairs
            out[:, start:start+k] = defined_code(self.placement[unique], self.placement_witnesses[unique])
        # All bands and the against lane remain zero; retained rows cache form
        # only. Every meaning, including a promoted identity's, reads placement.
        result = out[torch.tensor([row_index[r] for r in requested], device=out.device, dtype=torch.long)]
        return result.reshape(*shape, self.code_width)
