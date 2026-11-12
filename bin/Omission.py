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


def stage(model):
    """Hide native content before identity admission or any attention read."""
    from SentenceBoundary import detached
    isp, ps = model.inputSpace, model.perceptualSpace
    model._omission_targets = None
    if not model.word_brackets:
        return
    valid = isp._ar_word_part_mask
    if valid is None:
        raise RuntimeError('sentence omission requires the staged native percept slots')
    active = isp._word_active_mask
    isp._finalize_sentence_word_layout(active)
    sentences = isp._packed_sentence_ids
    forms = torch.zeros_like(valid)
    words = torch.zeros_like(active)
    protected = {}
    for sid in range(isp._packed_sentence_slot_mask.shape[1]):
        here = active & (sentences == sid)
        draw = model.omission.draw(valid & here[..., None], training=model.training)
        forms |= draw.forms
        words |= draw.words
        protected[sid] = draw.protected
    # Keep the original targets even before the first omission, so accuracy
    # can raise the rate. No consumer on the forward path receives this dict.
    original = model._staged_concepts_in.detach().clone()
    native = {key: value.detach().clone() if torch.is_tensor(value) else value
              for key, value in ps._forward_input.items()}
    model._omission_targets = dict(raw=original, native=native,
        word_values=isp._ar_embedded_N.detach().clone(), valid=valid.detach().clone(),
        ids=isp._ar_word_part_ids.detach().clone(), offsets=isp._ar_word_part_offsets.detach().clone(),
        draw=Draw(words, forms, protected), sentences=sentences.detach().clone())
    if not bool(forms.any()):
        return
    changed = forms.any(-1)
    # Keep the original sub-brackets; a zero is neither padding nor a token.
    byte_mask = torch.zeros_like(original[:, 0], dtype=torch.bool)
    for row, word, part in forms.nonzero().tolist():
        part_id = int(isp._ar_word_part_ids[row, word, part])
        lo = int(isp._ar_word_part_offsets[row, word, part])
        hi = lo + len(ps.percept_store.bytes_for(part_id))
        byte_mask[row, lo:hi] = True
    model._omission_targets['bytes'] = byte_mask
    model._staged_concepts_in = original.masked_fill(byte_mask[:, None], 0)
    object.__setattr__(model, '_ws_universe', model._staged_concepts_in)
    isp._ar_word_part_ids = isp._ar_word_part_ids.masked_fill(forms, -1)
    ps._serial_word_part_ids = isp._ar_word_part_ids
    slab = isp._ar_embedded_N
    width = int(ps.nWhat)
    # Reconstruct the visible form, preserving the preallocated word layout.
    visible = ps.synthesize_word_parts(isp._ar_word_part_ids, valid, isp._ar_word_part_offsets)
    isp._ar_embedded_N = torch.where(changed[..., None], visible.to(slab), slab)
    ps._embedded_input = isp._ar_embedded_N
    ps.subspace.set_event(isp._ar_embedded_N)
    fi = dict(ps._forward_input)
    for key in ('indices', 'native_indices'):
        if key not in fi:
            continue
        ids = fi[key].clone()
        spans = fi['native_part_spans' if key == 'native_indices' else 'part_spans']
        for row in range(len(ids)):
            for index, (lo, hi) in enumerate(spans[row].tolist()):
                if hi > lo and bool(byte_mask[row, lo:hi].any()):
                    ids[row, index] = -1
        fi[key] = ids
    if 'seed_event' in fi:
        seed = fi['seed_event']
        hidden = fi['indices'] < 0
        fi['seed_event'] = torch.cat((seed[..., :width].masked_fill(hidden[..., None], 0),
                                      seed[..., width:]), -1)
    fi['word_part_indices'] = isp._ar_word_part_ids
    for key in ('word_texts', 'tokens'):
        if key in fi:
            fi[key] = [[('\0' * len(value) if isinstance(value, str) else b'\0' * len(value))
                        if word < changed.shape[1] and bool(changed[row, word]) else value
                        for word, value in enumerate(values)] for row, values in enumerate(fi[key])]
    ps._forward_input = fi
    # These are alternate references to the same lexical stem, not targets.
    ps._last_tokens = fi.get('tokens', ())


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
        pid = int(targets['ids'][row, word, part])
        lo = int(targets['offsets'][row, word, part])
        hi = lo + len(model.perceptualSpace.percept_store.bytes_for(pid))
        correct[row, word, part] = hi <= decoded.shape[1] and torch.equal(decoded[row, lo:hi], raw[row, lo:hi])
    draw = targets['draw']
    model.omission.observe(correct, valid, Draw(draw.words, draw.forms, draw.protected[sid]))
