"""Loss-side targets and adaptive omission of visible sentence content."""
from dataclasses import dataclass
import torch
from torch import nn


@dataclass(frozen=True)
class Draw:
    words: torch.Tensor
    forms: torch.Tensor
    protected: torch.Tensor


class Omission(nn.Module):
    """Two rates, each estimated from visible and omitted reconstruction accuracy."""
    def __init__(self, target, limit, horizon):
        super().__init__()
        if not 0 <= target <= 1 or not 0 <= limit <= 1 or horizon <= 0:
            raise ValueError('omissionTarget and omissionLimit require [0,1]; omissionHorizon must be positive')
        self.target, self.limit, self.horizon = target, limit, horizon
        self.register_buffer('rates', torch.zeros(2))
        self.register_buffer('accuracies', torch.zeros(2, 2))
        self.register_buffer('observations', torch.zeros(2, 2, dtype=torch.long))

    @staticmethod
    def rate(visible, omitted, target, limit):
        denominator = visible - omitted
        ordinary = (visible - target) / torch.where(denominator != 0, denominator, 1.)
        return torch.where(visible <= target, 0.,
            torch.where(omitted >= target, limit, ordinary.clamp(0., limit)))

    def draw(self, valid, *, training):
        """Independent word and kept-word slot draws; protect one complete word."""
        if valid.ndim != 3:
            raise ValueError('omission requires [stream, word, percept] validity')
        words = valid.any(-1)
        empty = torch.zeros_like(valid)
        protected = torch.full((len(valid),), -1, device=valid.device, dtype=torch.long)
        # This branch consumes no RNG, including when explicitly disabled.
        if not training or self.target == 1 or not bool(self.rates.any()):
            return Draw(torch.zeros_like(words), empty, protected)
        from WalkTrials import departure_at
        protected = departure_at(words)
        kept = torch.arange(words.shape[1], device=words.device)[None] == protected[:, None]
        dropped = (torch.rand(words.shape, device=valid.device) < self.rates[0]) & words & ~kept
        forms = ((torch.rand(valid.shape, device=valid.device) < self.rates[1])
                 & valid & ~dropped[..., None] & ~kept[..., None])
        return Draw(dropped, forms | (dropped[..., None] & valid), protected)

    @torch.no_grad()
    def observe(self, correct, valid, draw):
        """Update once per present sentence from the kept reconstruction's bytes."""
        if correct.shape != valid.shape or draw.forms.shape != valid.shape:
            raise ValueError('omission accuracy must align with the original percept slots')
        word_valid = valid.any(-1)
        word_correct = (correct | ~valid).all(-1)
        # Form accuracy excludes words omitted as a whole.
        form_valid = valid & ~draw.words[..., None]
        hits, counts = [], []
        for right, present, hidden in ((word_correct, word_valid, draw.words), (correct, form_valid, draw.forms)):
            present, hidden = present.flatten(1), hidden.flatten(1)
            masks = torch.stack((present & ~hidden, present & hidden), -1)
            counts.append(masks.sum(1))
            hits.append((masks & right.flatten(1)[..., None]).sum(1))
        counts = torch.stack(counts, 1)
        fractions = torch.stack(hits, 1).to(self.accuracies) / counts.clamp_min(1)
        # Rows teach in presentation order. Both scales and both sides
        # share each update. An absent category leaves its history intact.
        for n in range(len(valid)):
            seen = counts[n] > 0
            filtered = torch.lerp(self.accuracies, fractions[n], 1. / self.horizon)
            updated = torch.where(self.observations == 0, fractions[n], filtered)
            self.accuracies.copy_(torch.where(seen, updated, self.accuracies))
            self.observations.add_(seen.long())
        self.rates.copy_(self.rate(self.accuracies[:, 0], self.accuracies[:, 1],
                                   self.target, self.limit))
        if self.target == 1:
            self.rates.zero_()

    def report(self):
        return dict(word_rate=float(self.rates[0]), form_rate=float(self.rates[1]),
            word_visible=float(self.accuracies[0, 0]), word_omitted=float(self.accuracies[0, 1]),
            form_visible=float(self.accuracies[1, 0]), form_omitted=float(self.accuracies[1, 1]))


def covered_bytes(spans, forms, length):
    """Union of the drawn byte intervals, including overlapping nested forms."""
    starts, stops = spans.clamp(0, length).flatten(1, 2).unbind(-1)
    counts = (forms.flatten(1) & (starts < stops)).long()
    changes = counts.new_zeros(len(forms), length + 1)
    changes.scatter_add_(1, starts.contiguous(), counts)
    changes.scatter_add_(1, stops.contiguous(), -counts)
    return changes[:, :-1].cumsum(1) > 0


def prepare(model, surface, spans, cover):
    """Draw from the lexical layout before any unit is inserted.

    ``surface`` [B, L] bytes, ``spans`` [B, W, 2] unit extents, ``cover``
    the parts of each unit: rows [B, W, P], extents [B, W, P, 2] and the
    number of tiles [B, W].
    """
    isp = model.inputSpace
    policy = model.omission
    units = (surface, spans)
    if ((not model.training or policy.target == 1)
            and getattr(policy.draw, '__func__', None) is Omission.draw):
        # The draw is identically empty and consumes no random numbers.
        # Use the finalized part layout for the loss and observation counts.
        model._omission_prepared = dict(raw=model._staged_concepts_in.detach().clone(), units=units)
        return None
    _, extents, counts = cover
    valid = torch.arange(extents.shape[2], device=extents.device) < counts[..., None]
    active = valid.any(-1)
    isp._finalize_sentence_word_layout(active)
    sentences = isp._packed_sentence_ids
    forms, words, protected = torch.zeros_like(valid), torch.zeros_like(active), {}
    for sid in range(isp._packed_sentence_slot_mask.shape[1]):
        draw = model.omission.draw(valid & (sentences == sid)[..., None], training=model.training)
        forms |= draw.forms
        words |= draw.words
        protected[sid] = draw.protected
    raw_target = model._staged_concepts_in.detach().clone()
    model._omission_prepared = dict(raw=raw_target, units=units,
        draw=Draw(words, forms, protected), sentences=sentences.detach().clone())
    if bool(forms.any()):
        model._omission_prepared['bytes'] = covered_bytes(extents, forms, raw_target.shape[-1])
    return model._omission_prepared


def host_units(units):
    """Each row's bytes and its units' extents, read once at the host boundary."""
    surface, spans = units
    width = surface.shape[1]
    rows = []
    for row in torch.cat((surface, spans.flatten(1)), 1).tolist():
        ends = row[width:]
        rows.append((bytes(row[:width]).rstrip(b'\0'),
                     [(s, e) for s, e in zip(ends[0::2], ends[1::2]) if e > s]))
    return rows


def native_values(ps, units):
    """Loss-only byte codes, without admitting an unseen target to the store.

    Known bytes use their actual code. A new target byte gets the store's
    normalized Gaussian initializer from a private copy of the random stream;
    neither the inventory nor the forward random stream observes that target.
    """
    store = ps.percept_store
    like = store._basis.W
    alphabet = dict.fromkeys(byte for raw, _ in units for byte in raw)
    width = max((len(raw) for raw, _ in units), default=0)
    if not width:
        return like.new_zeros(len(units), 0, store.dim), like.new_zeros(len(units), 0, dtype=torch.bool)
    # Use the first encounter order for the private initializer. The final
    # zero row supplies padding even when NUL itself is a real target.
    codes = like.new_zeros(len(alphabet) + 1, store.dim)
    generator = torch.Generator(device='cpu')
    generator.set_state(torch.random.get_rng_state())
    known, rows, missing, initial = [], [], [], []
    for seat, byte in enumerate(alphabet):
        alphabet[byte] = seat
        pid = store.get_id(bytes([byte]))
        if pid is None:
            value = torch.randn(store.dim, generator=generator, dtype=like.dtype, device='cpu')
            initial.append((value / value.norm().clamp_min(1e-8)).clamp(0., 1.))
            missing.append(seat)
        else:
            known.append(seat)
            rows.append(pid)
    if rows:
        codes[known] = store._basis.lookup_rows(torch.tensor(rows, device=like.device)).detach()
    if missing:
        codes[missing] = torch.stack(initial).to(like)
    seats = torch.tensor([[alphabet[byte] for byte in raw] + [len(alphabet)] * (width - len(raw))
                          for raw, _ in units], device=like.device, dtype=torch.long).reshape(len(units), width)
    return codes[seats], seats < len(alphabet)


def stage(model):
    """Retain detached targets and publish only the already omitted stem."""
    isp, ps = model.inputSpace, model.perceptualSpace
    model._omission_targets = None
    if not model.word_brackets:
        return
    prepared = model.__dict__.pop('_omission_prepared')
    valid = isp._ar_word_part_mask
    draw = prepared.get('draw')
    can_hide = draw is not None
    if can_hide:
        width = valid.shape[-1]
        forms = torch.nn.functional.pad(draw.forms, (0, max(0, width-draw.forms.shape[-1])))
        if forms.shape[-1] > width and bool(forms[..., width:].any()):
            raise ValueError('an omitted percept disappeared from its staged layout')
        forms = forms[..., :width]
        prepared['draw'] = Draw(draw.words, forms, draw.protected)
    else:
        active = isp._word_active_mask
        forms = torch.zeros_like(valid)
        protected = active.new_full((len(active),), -1, dtype=torch.long)
        slots = isp._packed_sentence_slot_mask.shape[1]
        prepared['draw'] = Draw(torch.zeros_like(active), forms, dict.fromkeys(range(slots), protected))
        prepared['sentences'] = isp._packed_sentence_ids.detach().clone()
    source = ps._forward_input
    # The element stem emits fresh byte, address and extent tensors for each
    # presentation. Later attention may replace entries in the opening dict.
    # Hold the original tensor handles for the two target readers.
    elements = source.get('element_input', False)
    native = {}
    for key in ('sentence_bytes', 'unit_spans', 'native_indices', 'native_part_spans'):
        if key in source:
            value = source[key].detach()
            native[key] = value if elements else value.clone()
    spans = source['part_spans'].reshape(*valid.shape, 2).detach()
    if not elements:
        spans = spans.clone()
    hiding = 'bytes' in prepared
    word_values = isp._ar_embedded_N.detach()
    if hiding:
        word_values = word_values.clone()  # the target is about to restore hidden content
    prepared.update(native=native, word_values=word_values, valid=valid.detach().clone(),
                    ids=isp._ar_word_part_ids.detach().clone(), spans=spans)
    if hiding or 'unit_spans' not in native:
        prepared['units'] = host_units(prepared['units'])
        prepared['word_bytes'] = tuple(tuple(raw[lo:hi] for lo, hi in units)
                                       for raw, units in prepared['units'])
    model._omission_targets = prepared
    if not hiding:
        prepared.pop('units')
        return
    values, native_valid = native_values(ps, prepared['units'])
    prepared.update(native_values=values, native_valid=native_valid)
    identity = getattr(ps.percept_store, 'identity', None)
    for b, w in forms.any(-1).nonzero().tolist():
        raw, units = prepared['units'][b]
        lo, hi = units[w]
        if identity is not None:
            form = identity.form(raw[lo:hi]).to(word_values)
            word_values[b, w, :ps.nWhat] = 0
            word_values[b, w, :len(form)] = form
        else:
            word_values[b, w, :ps.nWhat] = values[b, lo:hi].amax(0)
    original = prepared['raw']
    model._staged_concepts_in = original.masked_fill(prepared['bytes'][:, None], 0)
    object.__setattr__(model, '_ws_universe', model._staged_concepts_in)
    # A direct input names only the document actually presented. Explicit
    # dataset/conversation provenance remains authoritative.
    from Occurrence import restage_presented
    restage_presented(model, model._staged_concepts_in)
    prepared.pop('units')


def target(model):
    return getattr(model, '_omission_targets', None)


def correct_spans(raw, decoded, spans):
    """Exact equality of each observed byte interval using one prefix of mismatches."""
    common = min(raw.shape[1], decoded.shape[1])
    prefix = torch.nn.functional.pad((raw[:, :common] != decoded[:, :common]).long().cumsum(1), (1, 0))
    lo, hi = spans.flatten(1, 2).unbind(-1)
    mismatch = prefix.gather(1, hi.clamp(0, common)) != prefix.gather(1, lo.clamp(0, common))
    # A decoded interval past the supplied surface cannot equal a shorter
    # original slice. Empty brackets compare equal within the decoded extent.
    raw_length = (hi.clamp(0, raw.shape[1]) - lo.clamp(0, raw.shape[1])).clamp_min(0)
    decoded_length = (hi.clamp(0, decoded.shape[1]) - lo.clamp(0, decoded.shape[1])).clamp_min(0)
    equal = (hi <= decoded.shape[1]) & (raw_length == decoded_length) & (~mismatch | (hi <= lo))
    return equal.reshape(spans.shape[:-1])


def observe(model, sid, reverse, *, active=None):
    targets = target(model)
    if targets is None or not model.training:
        return
    raw = targets['raw'][:, 0]
    decoded = reverse.materialize()[:, 0]
    valid = targets['valid'] & (targets['sentences'] == sid)[..., None]
    if active is not None:
        valid = valid & active[:, None, None]
    correct = valid & correct_spans(raw, decoded, targets['spans'])
    draw = targets['draw']
    model.omission.observe(correct, valid, Draw(draw.words, draw.forms, draw.protected[sid]))
