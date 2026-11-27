"""The shared predictor read when a composed symbol enters working memory."""
import torch
from Expectation import ActivationPredictor, gain_field, symbol_activations, SymbolRead


def _anticipate_core(owner, source_lanes, values, orders, identities):
    """Predict next-step bank activations without mutating host trial state."""
    keys = owner._bank_keys.to(values)[None].expand(len(values), -1, -1)
    occupied = values.ne(0).any(-1)
    context = ActivationPredictor.context_by_order(values, occupied, orders, owner._bank_order_count)
    source = symbol_activations(keys, values, occupied, source_lanes,
        identities=identities, bank_ids=owner._bank_addresses,
        key_orders=owner._bank_orders, value_orders=orders)
    logits = owner._inter_predictor(*owner._prediction_inputs((keys, context, source)))
    return keys, logits, context, source, occupied


def _arrive_core(keys, logits, values, orders, identities, rows, lanes,
                 bank_addresses, bank_orders, strength, epsilon, saturation):
    """Score observed arrivals against anticipated logits as pure tensors."""
    present = values.ne(0).any(-1)
    occupied = present & rows[:, None]
    observed = symbol_activations(keys, values, occupied, lanes,
        identities=identities, bank_ids=bank_addresses,
        key_orders=bank_orders, value_orders=orders)
    source_lanes = (torch.stack((present.to(values), torch.zeros_like(present).to(values)), -1)
                    if lanes is None else lanes.detach().clone())
    expected = logits.sigmoid()
    field = gain_field(observed, expected, strength=strength,
        attention=rows[:, None], epsilon=epsilon, saturation=saturation)
    loss = torch.where(rows, (observed - expected).square().mean((1, 2)), 0.)
    return source_lanes, loss, observed, expected, field.excess, field.omission, field.content, field.gain


def _read_core(keys, logits, source, values, occupied, ids, orders,
               addresses, bank_orders, incoming_ids, incoming_lanes, incoming_rows):
    """Compute factual activation masks and gain inputs as pure tensors."""
    known = symbol_activations(keys, values, occupied,
        identities=ids, bank_ids=addresses,
        key_orders=bank_orders, value_orders=orders)[..., 0] > 0
    actual = source.detach().clone()
    if incoming_ids is not None:
        matches = (incoming_ids[:, None] == addresses[None]) & (addresses[None] >= 0)
        matches &= incoming_rows[:, None]
        actual = torch.maximum(actual, matches[..., None].to(actual) * incoming_lanes[:, None].detach())
        known = known | matches
    return logits.detach().sigmoid(), actual, known


class SymbolField:
    """A trial-local activation view; the native identity buffer is immutable.

    Predictions precede a word's composition. The resulting symbols, including
    higher orders, are scored on the same bank and receive scalar activation
    gains. Trial losses are consumed at the sentence boundary exactly once.
    """
    def __init__(self, owner, *, strength, epsilon, saturation, _regions=None):
        self.owner = owner
        self.strength, self.epsilon, self.saturation = strength, epsilon, saturation
        self._regions = _regions
        self.losses, self.reports = [], []
        self.pending = None
        self.source_lanes = None
        self.before = None

    def anticipate(self, state):
        if self.owner is None or not self.owner.expectation_enabled or not len(self.owner._bank_keys):
            self.pending = None
            return
        values = state[0]
        lanes = self.source_lanes
        if lanes is None:
            occupied = values.ne(0).any(-1)
            lanes = torch.stack((occupied.to(values), torch.zeros_like(occupied).to(values)), -1)
        regions = None if torch.compiler.is_compiling() else self._regions
        fn = regions['symbol_field_anticipate'] if regions else _anticipate_core
        keys, logits, context, source, occupied = fn(
            self.owner, lanes, values, state[2], state[4])
        self.pending = keys, logits, context, source
        self.before = (values.detach().clone(), occupied.detach().clone(),
                       state[4].detach().clone(), state[2].detach().clone())

    def arrive(self, state, rows, *, lanes=None):
        if self.pending is None:
            return state
        keys, logits, _, _ = self.pending
        values, _, orders, _, identities = state[:5]
        regions = None if torch.compiler.is_compiling() else self._regions
        fn = regions['symbol_field_arrive'] if regions else _arrive_core
        source_lanes, loss, observed, expected, excess, omission, content, gain = fn(
            keys, logits, values, orders, identities, rows, lanes,
            self.owner._bank_addresses, self.owner._bank_orders,
            self.strength, self.epsilon, self.saturation)
        self.source_lanes = source_lanes
        self.losses.append(loss)
        # Content is read through SymbolRead at the consumer. The STM holds
        # raw identities and original projection coefficients, so repeated
        # reads cannot compound a gain or re-identify dimmed coordinates.
        self.reports.append(dict(observed=observed.detach(), expected=expected.detach(),
                                excess=excess.detach(), omission=omission.detach(),
                                orders=self.owner._bank_orders.detach().clone(),
                                content=content.detach(), gain=gain.detach()))
        return state

    def read(self, *, incoming_ids=None, incoming_lanes=None, incoming_rows=None):
        if self.pending is None:
            return None
        keys, logits, _, source = self.pending
        values, occupied, ids, orders = self.before
        addresses = self.owner._bank_addresses
        regions = None if torch.compiler.is_compiling() else self._regions
        fn = regions['symbol_field_read'] if regions else _read_core
        expected, actual, known = fn(
            keys, logits, source, values, occupied, ids, orders,
            addresses, self.owner._bank_orders, incoming_ids, incoming_lanes, incoming_rows)
        return SymbolRead(keys, expected, self.strength,
                          self.epsilon, self.saturation, addresses, actual, known, self.owner._bank_orders)


    def loss(self):
        return None if not self.losses else torch.stack(self.losses).mean(0)
