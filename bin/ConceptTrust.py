"""Observed concept radii and relation admission (item 2.3)."""
import math
from contextlib import contextmanager
import torch
from torch import nn


def definition_threshold():
    from util import TheXMLConfig
    value = float(TheXMLConfig.space('ConceptualSpace', 'definitionTrust', .5))
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('definitionTrust must lie in [0, 1]')
    return value


class ConceptTrust(nn.Module):
    def __init__(self, owner, bank):
        super().__init__()
        object.__setattr__(self, 'owner', owner)
        from util import TheXMLConfig
        self.trust_views = float(TheXMLConfig.space('ConceptualSpace', 'trustViews', 3.))
        if not math.isfinite(self.trust_views) or self.trust_views <= 0:
            raise ValueError('trustViews must be finite and positive')
        definition_threshold()
        derived = getattr(bank, 'mereology', None)
        self.tracked = derived is not None and derived.meaning_pairs > 0
        # Untracked charts retain their existing whole-value norm.
        self.start = 0 if derived is None else derived.percept_event_width
        self.width = bank.W.shape[-1] if derived is None else derived.meaning_pairs
        self.lanes = 1 if derived is None else 2
        self.register_buffer('lengths', bank.W.new_zeros(len(bank.W), self.lanes))
        self.register_buffer('views', torch.zeros(len(bank.W), dtype=torch.long, device=bank.W.device))
        self.register_buffer('agreement', bank.W.new_zeros(len(bank.W)))
        self.register_buffer('content', torch.zeros_like(bank.W))
        self.register_buffer('known', torch.zeros(len(bank.W), dtype=torch.bool, device=bank.W.device))
        self.register_buffer('allocated', torch.zeros(len(bank.W), dtype=torch.bool, device=bank.W.device))
        self._deferred_word_views = None
        if isinstance(bank.W, nn.Parameter):
            import weakref
            bank.W._concept_trust_projection = weakref.ref(self)

    def tracks(self, identity):
        if self.tracked:
            return True
        row = self.owner._csw_row_of(int(identity))
        return row is not None and bool(self.allocated[row])

    @property
    def revision(self):
        bank = self.owner.similarity_codebook
        return (id(self), self.lengths._version, self.allocated._version,
                None if self.tracked else bank.W._version)

    @torch.no_grad()
    def capture(self, row):
        if not self.tracked:
            return
        value = self.owner.similarity_codebook.W[row]
        self.allocated[row] = True
        for lane in range(self.lanes):
            start = self.start + lane*self.width
            self.lengths[row, lane] = value[start:start+self.width].norm()

    @torch.no_grad()
    def remember(self, identity, content):
        if not self.tracks(identity):
            return
        from IdentityRows import carrier
        row = self.owner._csw_row_of(identity)
        if row is None:
            raise ValueError('trust requires an allocated concept row')
        if not bool(self.known[row]):
            self.content[row].copy_(carrier(self.owner, content, region=self.tracked))
            self.known[row] = True
        self.refresh(identity)

    @torch.no_grad()
    def refresh(self, identity):
        from Memory import NULL_CONCEPT
        if not self.tracks(identity) or identity == NULL_CONCEPT or self.owner.is_frozen(identity):
            return
        row = self.owner._csw_row_of(identity)
        if row is None or not bool(self.known[row]):
            return
        views = int(self.views[row])
        length = (float(self.agreement[row]) / views * -math.expm1(-views / self.trust_views)
                  if views else 0.)
        self.lengths[row, 0] = length
        self.project([row])

    @torch.no_grad()
    def project(self, rows=None):
        bank = self.owner.similarity_codebook
        update_content = rows is None
        rows = self.allocated.nonzero().flatten() if rows is None else torch.as_tensor(rows, device=bank.W.device, dtype=torch.long)
        if not len(rows):
            return
        values = bank.W[rows].clone()
        derived = getattr(bank, 'mereology', None)
        start, width = (self.start, self.width) if self.tracked else (0, bank.W.shape[-1])
        direction = values[:, start:start+width]
        if self.tracked:
            direction = torch.where(self.known[rows, None], direction.clamp_min(0), direction)
        fallback = self.content[rows, start:start+width] if not self.tracked else derived.placement[rows]
        direction = torch.where(direction.norm(dim=-1, keepdim=True) > 0, direction, fallback)
        direction = torch.nn.functional.normalize(direction, dim=-1)
        values[:, start:start+width] = direction * self.lengths[rows, 0, None]
        if update_content:
            raw = self.content[rows, start:start+width]
            self.content[rows, start:start+width] = torch.where(
                direction.norm(dim=-1, keepdim=True) > 0, direction * raw.norm(dim=-1, keepdim=True), raw)
        bank.W[rows] = values

    def _load_from_state_dict(self, state, prefix, *args, **kwargs):
        for name in ('lengths', 'views', 'agreement', 'content', 'known', 'allocated'):
            value = state.get(prefix+name)
            target = getattr(self, name)
            if value is None or value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError('concept trust checkpoint differs from constructed column: '+name)
        lengths = state[prefix+'lengths']
        if not bool(torch.isfinite(lengths).all()) or bool((lengths < 0).any()):
            raise ValueError('invalid concept lane lengths')
        if bool((state[prefix+'views'] < 0).any()) or not bool(torch.isfinite(state[prefix+'agreement']).all()) or bool((state[prefix+'agreement'] < 0).any()):
            raise ValueError('invalid concept view counters')
        super()._load_from_state_dict(state, prefix, *args, **kwargs)


def positive_length(owner, identity=None, *, value=None):
    """Only an allocated concept on a tracked chart contributes a radius."""
    if identity is None and value is None:
        raise ValueError('operand trust requires its concept or written value')
    radii = owner.similarity_codebook.trust
    if identity is None or not radii.tracks(identity):
        return 1.
    row = owner._csw_row_of(identity)
    if row is None or not bool(radii.allocated[row]):
        return 1.
    return float(radii.lengths[row, 0])


@contextmanager
def defer_word_views(owner):
    """Finish a kept close's scored links before learning its word radii."""
    trust = owner.similarity_codebook.trust
    if trust._deferred_word_views is not None:
        yield
        return
    pending = trust._deferred_word_views = []
    completed = False
    try:
        yield
        completed = True
    finally:
        trust._deferred_word_views = None
        if completed:
            for identity, content in pending:
                observe(owner, identity, content)


@torch.no_grad()
def observe(owner, identity, content):
    """Add one kept view without allocating a memory row."""
    from Memory import NULL_CONCEPT
    from IdentityRows import carrier, meet_ratios
    trust = owner.similarity_codebook.trust
    if (trust._deferred_word_views is not None
            and 'word' in owner._concept_allocator.symbol_types.get(identity, ())):
        trust._deferred_word_views.append((identity, content.detach().clone()))
        return
    if not trust.tracks(identity) or identity == NULL_CONCEPT or owner.is_frozen(identity):
        return
    trust.remember(identity, content)
    row = owner._csw_row_of(identity)
    value = carrier(owner, content, region=trust.tracked)
    if not bool(trust.content[row].any()):
        coverage = 1.
        if bool(value.any()):
            trust.content[row].copy_(value)
    else:
        candidate = trust.content[row]
        if not trust.tracked:
            # Signed whole-value charts have distinct positive and negative
            # coordinates for coverage; ordinary rows on them remain untracked.
            value = torch.cat((value.clamp_min(0), (-value).clamp_min(0)))
            candidate = torch.cat((candidate.clamp_min(0), (-candidate).clamp_min(0)))
        coverage = meet_ratios(value, candidate)[1]
    trust.views[row] += 1
    trust.agreement[row] += coverage
    trust.refresh(identity)


def measurements(owner):
    """Read curriculum readiness; this function never opens a gate or advances a lesson."""
    words = tuple(sorted(owner._word_definition_ids()))
    named = tuple(owner.definitions.word_ids)
    threshold = definition_threshold()
    store = owner._closed_clause_store()
    reference = None if store is None else getattr(store, '_occupation_owner', None)
    occupations = None if reference is None else reference()
    radii = owner.similarity_codebook.trust
    # Read the measured radius, not the neutral operand factor of D4/D8.
    rows = (owner._csw_row_of(word) for word in words)
    stable = sum(row is not None and bool(radii.allocated[row])
                 and float(radii.lengths[row, 0]) >= threshold for row in rows) if radii.tracked else 0
    denoted = sum(bool(owner.definitions.objects(word)) for word in named)
    return dict(admitted_words=len(words), stable_words=stable,
        stable_word_fraction=stable/len(words) if words else 0.,
        named_words=len(named), denoting_words=denoted,
        denoting_word_fraction=denoted/len(named) if named else 0.,
        operations={} if occupations is None else
            {name: occupations.operation(name) for name in occupations.operations})
