"""One index pairs a perceptual form with a conceptual meaning.

The cached representation is [form | meaning]. Perception reconstruction alone
writes native prototypes and evidence; the sentence read detaches them. Meaning at each order
is placed at kept closes from the observed field and its definitions;
the for lane reads its close-owned trust radius. Letters need no conceptual row.
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
    if not bool(gradient.any()):
        return q
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
        self.register_buffer('placement_version', torch.tensor(2, dtype=torch.long, device=initial.device))
        self.register_buffer('placement', initial.new_zeros(count, self.meaning_pairs))
        self.register_buffer('placement_witnesses', torch.zeros(count, dtype=torch.long, device=initial.device))
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
        from FeatureTable import FeatureTable
        self._feature_table = FeatureTable()
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
            rows = [row if row >= 0 else self.owner._concept_row_of(int(cid))
                    for row, cid in zip(rows, identities.reshape(-1).tolist())]
        return sentence_key([self.content_key(int(row)) for row in rows if row is not None and row >= 0])

    def placement_step(self, q, context, weights, negatives):
        """One subgradient step of the meet-ratio objective; negatives brake collapse."""
        return placement_step(q, context, weights, negatives,
                              rate=self.placement_rate, negative_weight=self.negative_weight)

    def same_order(self, row):
        """A physical block fixes the order of every constructed seat."""
        order = self.owner._row_order(int(row))
        caps = self.owner._order_caps()
        start = sum(caps[:order])
        rows = torch.arange(len(self.placement), device=self.placement.device)
        return (rows >= start) & (rows < start + caps[order])

    def _negatives(self, row, excess, placed, key):
        import hashlib
        eligible = ((excess == 0) & placed & self.same_order(row)).nonzero().flatten()
        eligible = eligible[eligible != row]
        generator = torch.Generator(device='cpu')
        generator.manual_seed(int.from_bytes(hashlib.sha256(key).digest()[:8], 'little'))
        order = torch.randperm(len(eligible), generator=generator)[:self.negative_count].to(eligible.device)
        return eligible[order]

    @torch.no_grad()
    def place_rows(self, rows, context, key):
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
            if self.owner.is_frozen(self.owner.concept_id_at_row(row)):
                continue
            q = codes[row]
            selected = (context > 0) & placed & self.same_order(row)
            selected[row] = False
            negatives = self._negatives(row, context, placed, key)
            self.placement[row].copy_(self.placement_step(q, codes[selected], context[selected], codes[negatives]))
            self.placement_witnesses[row] += 1
            self.placement_dirty[row] = True

    @torch.no_grad()
    def place_groups(self, groups, key):
        """Membership selects groups; every geometric operand is same-order."""
        if not self.meaning_pairs:
            return
        codes = self.placement.detach().clone()
        placed = self.placement_witnesses > 0
        targets, memberships = {}, {}
        for movers, members, group_key in groups:
            if not members:
                continue
            orders = {self.owner._row_order(int(row)) for row in (*movers, *members)}
            if len(orders) != 1:
                raise ValueError('placement groups cannot cross orders')
            if bool(placed[list(members)].all()):
                from ConceptHierarchy import group_centroid
                centroid = group_centroid(self.owner,
                    [self.owner.concept_id_at_row(row) for row in members], key=group_key)
            else:
                centroid = codes[list(members)].mean(0)
            for row in movers:
                targets.setdefault(row, []).append(centroid)
                memberships.setdefault(row, set()).update(members)
        for row, contexts in targets.items():
            if self.owner.is_frozen(self.owner.concept_id_at_row(row)):
                continue
            selected = codes.new_zeros(len(codes))
            selected[list(memberships[row])] = 1.
            negatives = self._negatives(row, selected, placed, key)
            self.placement[row] = self.placement_step(codes[row], torch.stack(contexts),
                codes.new_ones(len(contexts)) / len(contexts), codes[negatives])
            self.placement_witnesses[row] += 1
            self.placement_dirty[row] = True

    @torch.no_grad()
    def observe_close(self, reading, batch=0):
        """Place the registered sentence once, on its kept close only."""
        if reading is None:
            return None
        context = self.owner.placement_context(batch)
        sentence = context.clone()
        language = getattr(getattr(self.owner, '_model', None), 'languageSpace', None)
        if language is not None:
            for cid in language.parts_of_speech.concepts.values():
                row = self.owner._concept_row_of(cid)
                if row is not None:
                    sentence[row] = 0
        for cid, kinds in self.owner._concept_allocator.symbol_types.items():
            if 'object' in kinds:
                row = self.owner._concept_row_of(cid)
                if row is not None:
                    sentence[row] = 0
        rows = (sentence > 0).nonzero().flatten()
        key = self.reading_key(reading)
        self.place_rows(rows.tolist(), sentence, key)
        from ConceptPlacement import meta_groups
        self.place_groups(meta_groups(self.owner, context), key)
        return context

    def placement_extras(self):
        return dict(version=2, placement=self.placement.detach().cpu().clone(),
                    witnesses=self.placement_witnesses.detach().cpu().clone(),
                    keys=self.placement_keys.detach().cpu().clone(),
                    keyed=self.placement_keyed.detach().cpu().clone())

    @torch.no_grad()
    def load_placement_extras(self, saved):
        if not isinstance(saved, dict) or saved.get('version') not in (1, 2):
            raise ValueError('placement checkpoint version 1 or 2 is required')
        pairs = (('placement', self.placement), ('witnesses', self.placement_witnesses),
                 ('keys', self.placement_keys), ('keyed', self.placement_keyed))
        self._retired_flags(saved, ('derived', 'valid'))
        for key, target in pairs:
            value = saved.get(key)
            if not torch.is_tensor(value) or value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError('placement checkpoint differs from constructed column: '+key)
        self._validate_placement(saved['placement'], saved['witnesses'])
        for key, target in pairs:
            target.copy_(saved[key])
        self.placement_dirty.fill_(True)
        self._form_valid.zero_()

    @staticmethod
    def _validate_placement(codes, witnesses):
        norms = codes.norm(dim=-1)
        tolerance = max(1e-5, 8 * torch.finfo(codes.dtype).eps)
        if (not bool(torch.isfinite(codes).all())
                or bool((witnesses < 0).any())
                or bool(((norms > 1e-6) & ((norms - 1).abs() > tolerance)).any())):
            raise ValueError('invalid placement directions or witness counts')

    def _load_from_state_dict(self, state, prefix, *args, **kwargs):
        if self.property_width:
            layout = state.get(prefix+'_form_layout')
            if layout is None or not torch.equal(layout.to(self._form_layout), self._form_layout):
                saved = 'none' if layout is None else 'x'.join(map(str, layout.tolist()))
                raise ValueError('checkpoint form subspaces differ from the constructed layout: '
                                 f'the checkpoint has parts, wholes and event {saved}, this model '
                                 f"{'x'.join(map(str, self._form_layout.tolist()))}; a code is "
                                 'two halves of 64 on two lanes and is not converted')
        if int(state.get(prefix+'placement_version', -1)) not in (1, 2):
            raise ValueError('placement checkpoint version 1 or 2 is required')
        for name in ('placement', 'placement_witnesses', 'placement_keys', 'placement_keyed'):
            value = state.get(prefix+name)
            expected = getattr(self, name)
            if value is None or value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError('placement checkpoint differs from constructed column: '+name)
        self._validate_placement(state[prefix+'placement'], state[prefix+'placement_witnesses'])
        names = tuple(prefix+name for name in ('placement_derived', 'placement_valid'))
        self._retired_flags(state, names)
        for name in names:
            state.pop(name, None)
        state[prefix+'placement_version'] = self.placement_version.clone().fill_(2)
        super()._load_from_state_dict(state, prefix, *args, **kwargs)
        self.placement_dirty.fill_(True)
        self._form_valid.zero_()
        self._form_revision = None

    def _retired_flags(self, saved, names):
        """Old class flags have no reader; retain their learned directions."""
        import warnings
        present = [name for name in names if name in saved]
        for name in present:
            value = saved[name]
            if not torch.is_tensor(value) or value.shape != self.placement_witnesses.shape or value.dtype != torch.bool:
                raise ValueError('placement checkpoint differs from constructed column: '+name)
        if present:
            warnings.warn('discarding retired class placement flags; all orders now use field context', UserWarning)

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
        members = _concept_alloc_of(self.owner).order_zero_members
        return (id(members), members.revision,
                tuple(native), self.owner.basic_percepts.revision, None if identity is None else identity.revision)

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
            selected = (excess > 0) & placed & self.same_order(row)
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
        """Net evidence over basic percepts, the only order-zero literal kind."""
        from Spaces import _concept_alloc_of
        members = _concept_alloc_of(self.owner).order_zero_members
        definitions = {int(row): {} for row in rows}
        for (row, percept, negated), weight in members.items():
            if row in definitions:
                poles = definitions[row].setdefault(percept, [0., 0.])
                poles[int(negated)] += weight
        return {row: [(percept, max(0., poles[0]-poles[1])) for percept, poles in sorted(edges.items())]
                for row, edges in definitions.items()}

    @torch.no_grad()
    def containment_audit(self, *, before=None, after=None):
        """Observe both subspaces and their native containment, without writes."""
        from FormContainment import audit
        return audit(self.owner.basic_percepts, before=before, after=after)

    @torch.no_grad()
    def positive_directions(self, rows):
        """Render the native positive cone; the signed place itself stays intact."""
        return F.normalize(self.placement[rows].clamp_min(0), dim=-1)

    def positive_codes(self, rows):
        """The for-lane cache is derived from placement and its trusted radius."""
        return self.positive_directions(rows) * self.owner.similarity_codebook.trust.lengths[rows, 0, None]

    @torch.no_grad()
    def derive(self, indices, b=0):
        owner = self.owner
        cached = owner.similarity_codebook.W
        indices = torch.as_tensor(indices, dtype=torch.long, device=cached.device)
        shape = indices.shape
        unique, inverse = torch.unique(indices.flatten(), return_inverse=True)
        if bool(((unique < 0) | (unique >= len(cached))).any()):
            raise IndexError('concept row is outside the constructed inventory')
        self._invalidate_forms()
        from Spaces import _concept_alloc_of
        alloc = _concept_alloc_of(owner)
        basic = unique < owner._order_caps()[0]
        retained, fallback = self._feature_table.inventory(alloc, owner.basic_percepts, self._form_valid)
        kept = retained[unique]
        needed = unique[basic & ~kept & ~self._form_valid[unique]]
        if len(needed):
            self._forms[needed] = self._feature_table.forms(owner, alloc.order_zero_members,
                                                           needed, fallback[needed])
            self.form_rows_derived += len(needed)
        self._forms[unique[~basic]] = 0
        self._form_valid[unique] = True
        out = cached.new_zeros(len(unique), self.code_width)
        out[basic, :self.percept_width] = torch.where(kept[basic, None],
            cached[unique[basic], :self.percept_width], self._forms[unique[basic]])
        from PerceptCollections import definitions, code
        collections = definitions(owner)
        if collections:
            for row in collections:
                where = (unique == row) & basic
                if bool(where.any()):
                    out[where, :self.percept_width] = code(owner, row)[:self.percept_width]
        if len(unique) and self.meaning_pairs:
            start, k = self.percept_event_width, self.meaning_pairs
            out[:, start:start+k] = self.positive_codes(unique)
            out[:, start+k:start+2*k] = cached[unique][:, start+k:start+2*k]
        # Bands remain zero. Placement supplies only for evidence; written
        # against evidence keeps its own magnitude and direction.
        result = out[inverse]
        return result.reshape(*shape, self.code_width)
