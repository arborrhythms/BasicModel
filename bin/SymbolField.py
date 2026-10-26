"""The shared predictor read when a composed symbol enters working memory."""
import torch
from Expectation import ActivationPredictor, gain_field, symbol_activations, SymbolRead


class SymbolField:
    """A trial-local activation view; the native identity buffer is immutable.

    Predictions precede a word's composition. The resulting symbols, including
    higher orders, are scored on the same bank and receive scalar activation
    gains. Trial losses are consumed at the sentence boundary exactly once.
    """
    def __init__(self, owner, *, strength, epsilon, saturation):
        self.owner = owner
        self.strength, self.epsilon, self.saturation = strength, epsilon, saturation
        self.losses, self.reports = [], []
        self.pending = None
        self.source_lanes = None
        self.before = None

    def anticipate(self, state):
        if self.owner is None or not self.owner.expectation_enabled or not len(self.owner._bank_keys):
            self.pending = None
            return
        values, depth = state[:2]
        keys = self.owner._bank_keys.to(values)[None].expand(len(values), -1, -1)
        occupied = values.ne(0).any(-1)
        context = ActivationPredictor.context_from(values, occupied)
        source = symbol_activations(keys, values, occupied, self.source_lanes,
            identities=state[4], bank_ids=self.owner._bank_addresses)
        logits = self.owner._inter_predictor(keys, context, source)
        self.pending = keys, logits, context, source
        self.before = values.detach().clone(), occupied.detach().clone(), state[4].detach().clone()

    def arrive(self, state, rows, *, lanes=None):
        if self.pending is None:
            return state
        keys, logits, _, _ = self.pending
        values, depth, orders, grammar, identities, activation = state[:6]
        present = values.ne(0).any(-1)
        occupied = present & rows[:, None]
        observed = symbol_activations(keys, values, occupied, lanes,
            identities=identities, bank_ids=self.owner._bank_addresses)
        self.source_lanes = (torch.stack((present.to(values), torch.zeros_like(present).to(values)), -1)
                             if lanes is None else lanes.detach().clone())
        expected = logits.sigmoid()
        field = gain_field(observed, expected, strength=self.strength,
            attention=rows[:, None], epsilon=self.epsilon, saturation=self.saturation)
        self.losses.append(torch.where(rows, (observed-expected).square().mean((1,2)), 0.))
        self.reports.append(dict(observed=observed.detach(), expected=expected.detach(),
                                excess=field.excess.detach(), omission=field.omission.detach(),
                                orders=self.owner._bank_orders.detach().clone()))
        # Content is read through SymbolRead at the consumer. The STM holds
        # raw identities and original projection coefficients, so repeated
        # reads cannot compound a gain or re-identify dimmed coordinates.
        self.reports[-1].update(content=field.content.detach(), gain=field.gain.detach())
        return state

    def read(self, *, incoming_ids=None, incoming_lanes=None, incoming_rows=None):
        if self.pending is None:
            return None
        keys, logits, _, source = self.pending
        values, occupied, ids = self.before
        addresses = self.owner._bank_addresses
        known = symbol_activations(keys, values, occupied,
            identities=ids, bank_ids=addresses)[..., 0] > 0
        actual = source.detach().clone()
        if incoming_ids is not None:
            matches = (incoming_ids[:, None] == addresses[None]) & (addresses[None] >= 0)
            matches &= incoming_rows[:, None]
            actual = torch.maximum(actual, matches[..., None].to(actual)*incoming_lanes[:, None].detach())
            known = known | matches
        return SymbolRead(keys, logits.detach().sigmoid(), self.strength,
                          self.epsilon, self.saturation, addresses, actual, known)

    def closing_predictions(self):
        """The shared head's last pre-arrival read, separate from the D prior."""
        if self.pending is None:
            return ()
        keys, _, context, source = self.pending
        return tuple(self.owner._prediction((keys[b:b+1], context[b:b+1], source[b:b+1]),
                     self.owner._bank_ids, self.owner._bank_orders).detached()
                     for b in range(len(keys)))

    def loss(self):
        return None if not self.losses else torch.stack(self.losses).mean(0)
