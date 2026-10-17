"""Individual signatures and legacy sparse coding of change coordinates.

The dictionaries contain order-one concepts. Their addresses name columns;
addresses and word spellings never enter a numerical score. Only B retains
a sparse-coding objective; A stores the signatures and
witnesses allocated and revised by the reference closing.
A individuals are admitted at a kept close; B retains recurring innovations.
"""
from dataclasses import dataclass
import io
import math

import torch
from torch import nn
from torch.nn import functional as F


def _pack_state(state):
    """Keep the model's tensor-only weight contract for variable metadata."""
    stream = io.BytesIO()
    torch.save(state, stream)
    return torch.frombuffer(bytearray(stream.getvalue()), dtype=torch.uint8).clone()


def _unpack_state(state):
    if torch.is_tensor(state):
        if state.dtype != torch.uint8 or state.ndim != 1:
            raise ValueError('invalid independent-component metadata tensor')
        return torch.load(io.BytesIO(bytes(state.detach().cpu().tolist())),
                          map_location='cpu', weights_only=True)
    return state


@dataclass(frozen=True)
class SparseEncoding:
    codes: torch.Tensor
    reconstruction: torch.Tensor
    residual: torch.Tensor


class ColumnDictionary(nn.Module):
    """Native signatures, relevance and witnesses; no sparse inference or admission."""
    def __init__(self, dimension):
        super().__init__()
        if dimension < 1:
            raise ValueError('column dimension must be positive')
        self.dimension = int(dimension)
        self.directions = nn.ParameterDict()
        self.relevance = nn.ParameterDict()
        self.register_buffer('_anchor', torch.empty(0), persistent=False)
        self._witnesses = {}
        self._next_id = 1
        self.admissions = []

    @property
    def ids(self):
        return tuple(int(key) for key in self.directions)

    def matrix(self, features=None):
        if not self.directions:
            width = self.dimension if features is None else len(features)
            return self._anchor.new_zeros(0, width)
        values = F.normalize(torch.stack(tuple(self.directions.values())), dim=-1)
        return values if features is None else values.index_select(-1, features)

    def pruning_candidates(self):
        """Item 5 consumes these values; this owner never deletes a concept."""
        return tuple(int(key) for key, scale in self.relevance.items()
                     if float(scale.detach()) <= 0)

    def witness_count(self, identity):
        return len(self._witnesses.get(int(identity), ()))

    def get_extra_state(self):
        return _pack_state(dict(version=1, dimension=self.dimension, next_id=self._next_id,
                    witnesses={key: sorted(value, key=repr) for key, value in self._witnesses.items()}))

    def set_extra_state(self, state):
        state = _unpack_state(state)
        if state.get('version') != 1 or state.get('dimension') != self.dimension:
            raise ValueError('incompatible independent-component dictionary checkpoint')
        self._next_id = int(state['next_id'])
        self._witnesses = {int(key): set(value) for key, value in state.get('witnesses', {}).items()}

    def prepare_checkpoint(self, state_dict, prefix):
        for name, target in (('directions.', self.directions), ('relevance.', self.relevance)):
            for key, value in state_dict.items():
                if key.startswith(prefix + name):
                    local = key[len(prefix + name):]
                    shape = (self.dimension,) if name == 'directions.' else ()
                    if (not local.isdigit() or int(local) <= 0 or str(int(local)) != local
                            or not torch.is_tensor(value) or tuple(value.shape) != shape
                            or not value.is_floating_point() or not bool(torch.isfinite(value).all())):
                        raise ValueError('invalid independent-component column checkpoint')
                    if local not in target:
                        target[local] = nn.Parameter(torch.zeros_like(value, device=self._anchor.device))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        self.prepare_checkpoint(state_dict, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

class SparseDictionary(ColumnDictionary):
    """Legacy B innovation coding, pending its temporal-difference teacher.

    Greedy support is added only while it lowers the remainder. Sparse energy
    and recurrence apply to changes, never to A's individual signatures.
    """
    def __init__(self, dimension, *, max_sources, prior_scale, mint_threshold,
                 recurrence=4):
        super().__init__(dimension)
        if dimension < 1 or max_sources < 1 or recurrence < 1:
            raise ValueError('component dimensions, source count and recurrence must be positive')
        if not math.isfinite(prior_scale) or prior_scale <= 0:
            raise ValueError('the sparsity prior scale must be finite and positive')
        if not math.isfinite(mint_threshold) or mint_threshold <= 0:
            raise ValueError('the component mint threshold must be finite and positive')
        self.dimension, self.max_sources = int(dimension), int(max_sources)
        self.prior_scale, self.mint_threshold = float(prior_scale), float(mint_threshold)
        self.recurrence = int(recurrence)
        self._pending = []

    def encode(self, values, *, features=None, max_sources=None):
        width = self.dimension if features is None else len(features)
        if values.shape[-1] != width:
            raise ValueError('component observations must use the declared event coordinates')
        matrix = self.matrix(features).to(values)
        if not len(matrix):
            return SparseEncoding(values.new_zeros(*values.shape[:-1], 0),
                                  torch.zeros_like(values), values)
        scales = torch.stack(tuple(self.relevance.values())).clamp_min(0).to(values)
        projected = values @ matrix.T
        count = min(self.max_sources if max_sources is None else int(max_sources), len(matrix))
        if count < 1:
            raise ValueError('a component observation needs a positive source ceiling')
        selected = torch.empty(*values.shape[:-1], 0, dtype=torch.long, device=values.device)
        used = torch.zeros_like(projected, dtype=torch.bool)
        coefficients = torch.zeros_like(projected)
        reconstruction, residual = torch.zeros_like(values), values
        continuing = torch.ones(values.shape[:-1], dtype=torch.bool, device=values.device)
        for step in range(count):
            score = ((residual @ matrix.T) * scales).abs().masked_fill(used, -torch.inf)
            next_column = score.argmax(-1, keepdim=True)
            selected = torch.cat((selected, next_column), -1)
            active = matrix[selected]
            gram = active @ active.transpose(-1, -2)
            ridge = torch.eye(step + 1, device=values.device, dtype=values.dtype) * 1e-6
            coordinates = torch.linalg.solve(gram + ridge,
                projected.gather(-1, selected).unsqueeze(-1)).squeeze(-1)
            sparse = F.softshrink(coordinates, self.prior_scale) * scales[selected]
            proposed = torch.zeros_like(projected).scatter(-1, selected, sparse)
            recovered = proposed @ matrix
            remainder = values - recovered
            energy = residual.square().sum(-1)
            tolerance = torch.finfo(values.dtype).eps * values.square().sum(-1).clamp_min(1.)
            continuing = continuing & (remainder.square().sum(-1) < energy - tolerance)
            coefficients = torch.where(continuing[..., None], proposed, coefficients)
            reconstruction = torch.where(continuing[..., None], recovered, reconstruction)
            residual = values - reconstruction
            used = used.scatter(-1, next_column, True)
        return SparseEncoding(coefficients, reconstruction, residual)

    def loss(self, population, *, features=None, source_limits=None):
        """Mean sparse-coding energy over the population, independent of batching."""
        if not population.numel():
            return population.sum() * 0
        # Sum over coordinates, mean over observations: adding unused event
        # capacity cannot make a residual or its admission threshold smaller.
        limits = ([self.max_sources] * len(population) if source_limits is None else list(source_limits))
        if len(limits) != len(population) or any(count < 1 for count in limits):
            raise ValueError('one positive grammar source ceiling is required per observation')
        reconstruction, sparse_prior = population.new_zeros(()), population.new_zeros(())
        for count in sorted(set(limits)):
            rows = torch.tensor([i for i, limit in enumerate(limits) if limit == count],
                                device=population.device, dtype=torch.long)
            encoded = self.encode(population.index_select(0, rows), features=features, max_sources=count)
            reconstruction = reconstruction + .5 * encoded.residual.square().sum() / len(population)
            sparse_prior = sparse_prior + self.prior_scale * encoded.codes.abs().sum() / len(population)
        if not self.directions:
            return reconstruction
        matrix = self.matrix()
        gram = matrix @ matrix.T
        off_diagonal = gram - torch.diag_embed(gram.diagonal())
        coherence = off_diagonal.square().sum() / max(1, len(matrix) * (len(matrix) - 1))
        scales = torch.stack(tuple(self.relevance.values())).clamp_min(0)
        relevance_prior = self.prior_scale * scales.mean()
        return reconstruction + sparse_prior + self.prior_scale * coherence + relevance_prior

    @torch.no_grad()
    def observe(self, value, *, witness, prediction=None, allocate=None, max_sources=None, allow_mint=True):
        """Admit unexplained recurring content after a kept closing, as an effect.

        ``allocate`` reserves an order-one native concept or returns None when
        its physical inventory is full. No trial calls this method. Pending
        prototypes retain at most the recurrence count's distinct witnesses.
        """
        if value.shape != (self.dimension,) or not bool(torch.isfinite(value).all()):
            raise ValueError('component admission needs one finite event-coordinate vector')
        encoded = self.encode(value, max_sources=max_sources)
        for identity, used in zip(self.ids, encoded.codes.ne(0).tolist()):
            if used:
                self._witnesses.setdefault(identity, set()).add(witness)
        if not allow_mint:
            return
        # Witness the whole observed row, but admit only the part of its
        # surprise that existing columns cannot explain.
        surprise = value if prediction is None else value - prediction
        residual = self.encode(surprise, max_sources=max_sources).residual
        if float(residual.norm()) <= self.mint_threshold:
            return
        direction = F.normalize(residual.detach().to(self._anchor), dim=-1)
        pending = next((entry for entry in self._pending
                        if float((entry['direction'] @ direction).abs()) >= 1 - self.mint_threshold), None)
        if pending is None:
            pending = dict(direction=direction.clone(), witnesses=[])
            self._pending.append(pending)
        if witness in pending['witnesses']:
            return
        if len(pending['witnesses']) < self.recurrence:
            pending['witnesses'].append(witness)
        if len(pending['witnesses']) < self.recurrence:
            return
        identity = self._next_id if allocate is None else allocate(pending['direction'])
        if identity is None:
            return
        key = str(int(identity))
        if int(identity) <= 0 or key in self.directions:
            raise ValueError('component admission requires a fresh positive native concept identity')
        self.directions[key] = nn.Parameter(pending['direction'].clone())
        self.relevance[key] = nn.Parameter(self._anchor.new_tensor(1.))
        self._witnesses[int(identity)] = set(pending['witnesses'])
        self._next_id = max(self._next_id, int(identity) + 1)
        self._pending = [entry for entry in self._pending if entry is not pending]
        self.admissions.append(int(identity))

    def get_extra_state(self):
        state = _unpack_state(super().get_extra_state())
        state['pending'] = [dict(direction=entry['direction'].detach().cpu(),
                                 witnesses=list(entry['witnesses'])) for entry in self._pending]
        return _pack_state(state)

    def set_extra_state(self, state):
        state = _unpack_state(state)
        super().set_extra_state(state)
        self._pending = [dict(direction=entry['direction'].to(self._anchor),
                              witnesses=list(entry['witnesses'])) for entry in state.get('pending', ())]

class EventChart:
    """Order-zero event coordinates, with independent evidence lanes.

    Only the conceptual content enters this chart. Native and outer address
    bands never become independent components. The chart's code snapshot is
    detached; the current observation may carry a graph.
    """
    def __init__(self, rows, codes, *, capacity, content_width, meaning_start=None,
                 meaning_pairs=None, pole_stride=None):
        self.rows = rows.to(dtype=torch.long, device=codes.device)
        self.capacity, self.content_width = int(capacity), int(content_width)
        self.meaning_start, self.meaning_pairs = meaning_start, meaning_pairs
        self.pole_stride = meaning_pairs if pole_stride is None else pole_stride
        self.width = codes.shape[-1]
        if meaning_start is None:
            self.basis = F.normalize(codes[..., :content_width].detach(), dim=-1)
        else:
            self.basis = F.normalize(codes[..., meaning_start:meaning_start + meaning_pairs].detach(), dim=-1)
        self.features = torch.cat((self.rows, self.rows + self.capacity))

    def project(self, values):
        if self.meaning_start is None:
            positive = values[..., :self.content_width] @ self.basis.T.to(values)
            negative = torch.zeros_like(positive)
        else:
            start, count = self.meaning_start, self.meaning_pairs
            positive = values[..., start:start + count] @ self.basis.T.to(values)
            offset = start + self.pole_stride
            negative = values[..., offset:offset + count] @ self.basis.T.to(values)
        return torch.cat((positive, negative), -1)

    def expand(self, values):
        shape = (*values.shape[:-1], 2 * self.capacity)
        indices = self.features.to(values.device).expand(*values.shape[:-1], -1)
        return values.new_zeros(shape).scatter(-1, indices, values)

    def render(self, values, *, like=None):
        count = len(self.rows)
        positive, negative = values[..., :count], values[..., count:]
        shape = (*values.shape[:-1], self.width)
        result = values.new_zeros(shape) if like is None else like.detach().clone()
        if self.meaning_start is None:
            result[..., :self.content_width] = positive @ self.basis.to(values)
        else:
            start, pairs = self.meaning_start, self.meaning_pairs
            result[..., start:start + pairs] = positive @ self.basis.to(values)
            offset = start + self.pole_stride
            result[..., offset:offset + pairs] = negative @ self.basis.to(values)
        return result


class IndependentComponents(nn.Module):
    """ConceptualSpace owns A, B and their occurrence-addressed statistics.

    LTM remains the observation store. The only added durable population is
    the surprise in identified column coordinates, not sentence input traces.
    The bounded situation supplies candidates; this owner never scans LTM to
    choose a referent. Admission occurs solely on committed observations.
    """
    def __init__(self, owner, *, weight, prior_scale, mint_threshold, recurrence=4):
        super().__init__()
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('independence weight must be finite and nonnegative')
        object.__setattr__(self, 'owner', owner)
        self.weight = float(weight)
        self.capacity = int(owner.nVectors)
        options = dict(prior_scale=prior_scale, mint_threshold=mint_threshold, recurrence=recurrence)
        self.nouns = ColumnDictionary(2 * self.capacity)
        self.verbs = SparseDictionary(3 * self.capacity, max_sources=1, **options)
        self._innovations = {}
        self._chart = None
        self.admission_drops = 0
        from ReferenceClose import ReferenceClose
        self.reference_close = ReferenceClose(self)

    @property
    def source_capacity(self):
        return int(getattr(self.owner, 'stm_capacity', getattr(self.owner.stm, 'capacity', 8)))

    def getParameters(self):
        return list(self.parameters())

    def begin_forward(self):
        """One pre-reading dictionary and LTM population snapshot for both trials."""
        from Spaces import _concept_alloc_of
        owner = self.owner
        basis = owner.similarity_codebook
        rows = sorted(row for row in _concept_alloc_of(owner).layer()._tensor_row_keys
                      if owner._order0_inventory_row(row))
        indices = torch.tensor(rows, dtype=torch.long, device=basis.W.device)
        codes = basis.lookup_rows(indices).detach() if rows else basis.W[:0].detach()
        derived = getattr(basis, 'mereology', None)
        self._chart = EventChart(indices, codes, capacity=self.capacity,
            content_width=int(owner.nWhat),
            meaning_start=None if derived is None else derived.percept_event_width,
            meaning_pairs=None if derived is None else derived.meaning_pairs,
            pole_stride=None if derived is None else derived.reserved_pairs)
        store = owner._closed_clause_store()
        retained = set()
        if store is not None:
            retained = {int(store.row_ids[row]) for row in range(len(store))
                        if int(store.rel_type[row]) != store.REL_DEF
                        and store.KINDS[int(store.record_kind[row])] == 'observation'}
        # Innovations are observations of the same retained chain. Discard
        # an orphan when its LTM occurrence leaves; this neither removes nor
        # prunes an identity/change column (item 5 owns that decision).
        self._innovations = {identity: value for identity, value in self._innovations.items()
                             if identity in retained}

    def _ready(self):
        if self._chart is None:
            self.begin_forward()
        return self._chart

    def column_point(self, identity):
        from MeaningCodes import defined_code
        chart = self._ready()
        key = str(int(identity))
        if key in self.nouns.directions:
            direction = F.normalize(self.nouns.directions[key], dim=-1)
            point = chart.render(direction.index_select(-1, chart.features))
            return defined_code(point, self.nouns.witness_count(identity))
        if key in self.verbs.directions and self.nouns.ids:
            rows = torch.tensor([self.owner._csw_row_of(cid) for cid in self.nouns.ids],
                                device=chart.features.device, dtype=torch.long)
            changes = self.verbs.directions[key].reshape(3, self.capacity).index_select(-1, rows)
            point = chart.render(changes.sum(0) @ self.nouns.matrix(chart.features))
            return defined_code(point, self.verbs.witness_count(identity))
        return None

    def innovation(self, observation, prediction):
        """Per-role innovation in A's coordinates; no difference between poles."""
        # Legacy B coordinates are projections onto a detached signature
        # snapshot, not sparse identification or an independence loss on A.
        # The per-individual temporal-difference teacher remains deferred.
        chart = self._ready()
        axes = self.nouns.matrix(chart.features).detach()
        observed = chart.project(observation) @ axes.T.to(observation)
        expected = chart.project(prediction) @ axes.T.to(prediction)
        values = observed - expected
        rows = [self.owner._csw_row_of(cid) for cid in self.nouns.ids]
        if any(row is None for row in rows):
            raise RuntimeError('an identity column has lost its native conceptual row')
        indices = torch.tensor(rows, dtype=torch.long, device=values.device).expand(3, -1)
        return values.new_zeros(3, self.capacity).scatter(-1, indices, values).flatten()

    def cost(self, meanings, predictions, active, *, clauses=None):
        """Legacy B energy only; A receives no sparse-coding pressure."""
        chart = self._ready()
        like = chart.basis
        past = list(self._innovations.values())
        changes = (torch.stack([value.to(like) for value in past]) if past else
                   like.new_zeros(0, 3 * self.capacity))
        losses = []
        for b, meaning in enumerate(meanings):
            loss = like.new_zeros(())
            if meaning is not None and bool(active[b]):
                prediction = predictions[b] if predictions is not None else None
                if prediction is not None and self.nouns.ids:
                    residual = self.innovation(meaning.roles, prediction.roles.detach())
                    loss = self.verbs.loss(torch.cat((changes, residual[None]), 0))
                elif len(changes):
                    loss = self.verbs.loss(changes)
            losses.append(loss)
        return torch.stack(losses)

    @torch.no_grad()
    def _allocate(self, direction, *, verb=False):
        from Spaces import _concept_alloc_of
        owner = self.owner
        try:
            owner._preflight_concept_row(1)
            owner._preflight_concept_allocation(1, context='independent component')
        except RuntimeError:
            self.admission_drops += 1
            return None
        identity = owner.new_concept()
        alloc = _concept_alloc_of(owner)
        alloc.reference_orders[identity] = 1
        # Event-coordinate constituents declare the native concept's order;
        # learned coefficients are exclusively the dictionary's parameters.
        support = direction.reshape(3 if verb else 2, self.capacity).abs().sum(0).nonzero().flatten()
        if not verb:
            for row in support.tolist():
                event = owner.concept_id_at_row(row)
                if event is not None:
                    alloc.add(identity, 'part', ('sym', event))
        alloc.settle(identity)
        if owner._csw_concept_row(1, identity) is None:
            raise RuntimeError('component inventory changed after its allocation preflight')
        return identity

    @torch.no_grad()
    def commit(self, row, meaning, prediction=None):
        if getattr(self.owner, '_online_learning_frozen', False):
            return
        chart = self._ready()
        if not len(chart.rows):
            return
        store = self.owner._closed_clause_store()
        identity = int(store.row_ids[row])
        witness = (identity, int(store.witness_count[row]))
        before = {id(p) for p in self.parameters()}
        # A's individuals were admitted from the kept NP occurrences at this
        # close. Whole-row residual recurrence is not individual identity.
        # The deferred order-two mechanism owns cross-document recurrence.
        if prediction is not None and self.nouns.ids:
            residual = self.innovation(meaning.roles.detach(), prediction.roles.detach())
            self._innovations[identity] = residual.detach().cpu()
            self.verbs.observe(residual, witness=witness,
                allocate=lambda direction: self._allocate(direction, verb=True))
        model = getattr(self.owner, '_model', None)
        if model is not None:
            model.__dict__.setdefault('_fresh_synthesis_params', []).extend(
                p for p in self.parameters() if id(p) not in before)

    def candidate_frames(self, frames):
        """Columns and their latest situated witness, without reading more LTM."""
        if not self.nouns.ids:
            return ()
        result = {}
        store = self.owner._closed_clause_store()
        for frame in frames:
            row = store.index_of_row(frame.row_id) if store is not None else None
            if row is None:
                continue
            for reference in self.reference_close.occurrences(row):
                address = self.reference_close.address(reference)
                if address is None:
                    continue
                index = store.index_of_row(address)
                if index is not None and int(store.order[index]) == 1:
                    identity = int(store.refs[index, 0])
                    if identity in self.nouns.ids:
                        result.pop(identity, None)
                        result[identity] = frame
        return tuple(result.items())[-self.source_capacity:]

    def candidates(self, frames):
        return tuple(identity for identity, _ in self.candidate_frames(frames))

    def situated_point(self, identity, frame):
        """A column's content beside its witness's continuity evidence.

        The reserved reference band is available to the same operation
        scorer. It never enters A, B or the independence chart.
        """
        point = self.column_point(identity)
        if point is None:
            return None
        derived = getattr(getattr(self.owner, 'similarity_codebook', None), 'mereology', None)
        if derived is not None:
            start, end = derived.percept_width, derived.percept_event_width
            bands = [value.detach().to(point).flatten() for value in (frame.where, frame.when)
                     if value is not None]
            if bands:
                evidence = torch.cat(bands)
                if len(evidence) > end - start:
                    raise ValueError('situated identity evidence exceeds its reserved reference band')
                point = point.clone()
                point[start:start + len(evidence)] = evidence
        return point

    def pruning_candidates(self):
        return dict(nouns=self.nouns.pruning_candidates(), verbs=self.verbs.pruning_candidates())

    def get_extra_state(self):
        return _pack_state(dict(version=1, innovations=self._innovations, admission_drops=self.admission_drops))

    def set_extra_state(self, state):
        state = _unpack_state(state)
        if state.get('version') != 1:
            raise ValueError('incompatible independent-component population checkpoint')
        self._innovations = state['innovations']
        self.admission_drops = int(state['admission_drops'])
        self._chart = None

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # A pre-6.5 checkpoint has no component parameters or recurrence yet.
        if not any(key.startswith(prefix) for key in state_dict):
            for name, value in self.state_dict().items():
                state_dict[prefix + name] = value
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
