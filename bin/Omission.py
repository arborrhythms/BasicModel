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
        samples = ((word_correct, word_valid, draw.words), (correct, form_valid, draw.forms))
        for row in range(len(valid)):
            if not bool(word_valid[row].any()):
                continue
            for scale, (right, present, hidden) in enumerate(samples):
                for side in range(2):
                    mask = present[row] & (hidden[row] if side else ~hidden[row])
                    if bool(mask.any()):
                        value = right[row][mask].to(self.accuracies).mean()
                        count = self.observations[scale, side]
                        alpha = 1. if not int(count) else 1. / self.horizon
                        self.accuracies[scale, side].lerp_(value, alpha)
                        count.add_(1)
        self.rates.copy_(self.rate(self.accuracies[:, 0], self.accuracies[:, 1],
                                   self.target, self.limit))
        if self.target == 1:
            self.rates.zero_()

    def report(self):
        return dict(word_rate=float(self.rates[0]), form_rate=float(self.rates[1]),
            word_visible=float(self.accuracies[0, 0]), word_omitted=float(self.accuracies[0, 1]),
            form_visible=float(self.accuracies[1, 0]), form_omitted=float(self.accuracies[1, 1]))


def prepare(model, units):
    """Draw from the lexical layout before any unit or byte is admitted."""
    from WordIdentity import base_atoms
    ps, isp = model.perceptualSpace, model.inputSpace
    identity = getattr(ps.percept_store, 'identity', None)
    width = int(ps._serial_word_capacity)
    plans, capacity = [], 3
    for raw, spans in units:
        row = []
        for lo, hi in spans[:width]:
            word = raw[lo:hi]
            atoms = (tuple(identity.words.get(word, base_atoms(word))) if identity is not None else ())
            nested = () if identity is None else tuple(identity.nested.get(word, ()))
            parts = ([(lo, hi)] * len(atoms) + [(lo+a, lo+z) for _, a, z in nested]
                     if identity is not None else [(i, i+1) for i in range(lo, hi)])
            row.append(dict(atoms=atoms, nested=nested, spans=parts))
            capacity = max(capacity, len(parts))
        plans.append(row)
    device = model._staged_concepts_in.device
    valid = torch.zeros(len(units), width, capacity, dtype=torch.bool, device=device)
    for b, row in enumerate(plans):
        for w, plan in enumerate(row):
            valid[b, w, :len(plan['spans'])] = True
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
    hidden = torch.zeros_like(raw_target[:, 0], dtype=torch.bool)
    for b, w, part in forms.nonzero().tolist():
        lo, hi = plans[b][w]['spans'][part]
        hidden[b, lo:hi] = True
    model._omission_prepared = dict(raw=raw_target, units=units, plans=plans,
        draw=Draw(words, forms, protected), bytes=hidden,
        sentences=sentences.detach().clone())
    return model._omission_prepared


def native_values(ps, units):
    """Loss-only byte codes, without admitting an unseen target to the store.

    Known bytes use their actual code. A new target byte gets the store's
    normalized Gaussian initializer from a private copy of the random stream;
    neither the inventory nor the forward random stream observes that target.
    """
    store = ps.percept_store
    generator = torch.Generator(device='cpu')
    generator.set_state(torch.random.get_rng_state())
    like = store._basis.W
    values = like.new_zeros(len(units), max((len(raw) for raw, _ in units), default=0), store.dim)
    valid = torch.zeros(values.shape[:2], dtype=torch.bool, device=values.device)
    unknown = {}
    for b, (raw, _) in enumerate(units):
        for position, byte in enumerate(raw):
            pid = store.get_id(bytes([byte]))
            if pid is not None:
                value = store._basis.lookup_rows(torch.tensor(pid, device=like.device)).detach()
            else:
                if byte not in unknown:
                    value = torch.randn(store.dim, generator=generator, dtype=like.dtype)
                    unknown[byte] = (value / value.norm().clamp_min(1e-8)).clamp(0., 1.).to(like)
                value = unknown[byte]
            values[b, position] = value
            valid[b, position] = True
    return values, valid


def stage(model):
    """Retain detached targets and publish only the already omitted stem."""
    isp, ps = model.inputSpace, model.perceptualSpace
    model._omission_targets = None
    if not model.word_brackets:
        return
    prepared = model.__dict__.pop('_omission_prepared')
    draw = prepared['draw']
    valid = isp._ar_word_part_mask
    width = valid.shape[-1]
    forms = torch.nn.functional.pad(draw.forms, (0, max(0, width-draw.forms.shape[-1])))
    if forms.shape[-1] > width and bool(forms[..., width:].any()):
        raise ValueError('an omitted percept disappeared from its staged layout')
    forms = forms[..., :width]
    prepared['draw'] = Draw(draw.words, forms, draw.protected)
    native = {key: value.detach().clone() if torch.is_tensor(value) else value
              for key, value in ps._forward_input.items()}
    word_values = isp._ar_embedded_N.detach().clone()
    spans = torch.zeros(*valid.shape, 2, dtype=torch.long, device=valid.device)
    fi_spans = ps._forward_input['part_spans'].reshape(*valid.shape, 2)
    spans.copy_(fi_spans)
    prepared.update(native=native, word_values=word_values, valid=valid.detach().clone(),
                    ids=isp._ar_word_part_ids.detach().clone(), spans=spans,
                    word_bytes=tuple(tuple(raw[lo:hi] for lo, hi in units)
                                     for raw, units in prepared['units']))
    model._omission_targets = prepared
    if not bool(forms.any()):
        prepared.pop('bytes')
        prepared.pop('units'); prepared.pop('plans')
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
    prepared.pop('units'); prepared.pop('plans')


def target(model):
    return getattr(model, '_omission_targets', None)


def visible_events(model, predicted, original):
    """Exclude hidden bytes from the grammar-free reconstruction on both sides."""
    targets = target(model)
    hidden = None if targets is None else targets.get('bytes')
    if hidden is None:
        return predicted, original
    def mask(value):
        width = value.shape[1]
        gate = torch.nn.functional.pad(hidden[:, :width], (0, max(0, width-hidden.shape[1])))
        return value.masked_fill(gate[..., None], 0)
    return mask(predicted), mask(original)


def observe(model, sid, reverse, *, active=None):
    targets = target(model)
    if targets is None or not model.training:
        return
    raw = targets['raw'][:, 0]
    decoded = reverse.materialize()[:, 0]
    correct = torch.zeros_like(targets['valid'])
    valid = targets['valid'] & (targets['sentences'] == sid)[..., None]
    if active is not None:
        valid = valid & active[:, None, None]
    for row, word, part in valid.nonzero().tolist():
        lo, hi = targets['spans'][row, word, part].tolist()
        correct[row, word, part] = hi <= decoded.shape[1] and torch.equal(decoded[row, lo:hi], raw[row, lo:hi])
    draw = targets['draw']
    model.omission.observe(correct, valid, Draw(draw.words, draw.forms, draw.protected[sid]))
