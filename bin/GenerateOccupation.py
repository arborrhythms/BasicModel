"""Formal argument occupation scores for parent and proposed children."""
import torch
from PartsOfSpeech import operation_name


def features(language, parent, left, right, *, expected=None):
    result = parent.new_zeros(*left.shape[:-1], 3)
    speech = language.parts_of_speech
    rules = (*language._generate_binary_rules, *language._generate_unary_rules)
    parent_scores = speech.value_scores(parent)
    left_scores, right_scores = speech.value_scores(left), speech.value_scores(right)
    for index, rule in enumerate(rules):
        name = operation_name(rule)
        for lane, face, scores in ((0, 'argout1', parent_scores),
                                  (1, 'argin1', left_scores[:, index]),
                                  (2, 'argin2', right_scores[:, index])):
            argument = speech.index.get(name + '_' + face) if name else None
            if argument is not None:
                result[:, index, lane] = scores[..., argument]
                if expected is not None:
                    result[:, index, lane] += expected[:, argument]
    # No rule is attached to STOP. Undeclared fixed faces also remain zero.
    return result.detach()


def bank(language):
    """Snapshot the current native bank at the eager walk boundary."""
    space = getattr(language.parts_of_speech, '_space', None)
    if space is None or space() is None:
        return ()
    model = getattr(space(), '_model', None)
    owner = getattr(getattr(model, 'symbolSpace', None), 'expectation', None)
    if owner is None or not owner.expectation_enabled or not len(owner._bank_keys):
        return ()
    # The active bank can grow after every kept sentence. The compiled walk
    # receives the owner's construction-time inventory extent, with unused
    # rows explicitly untyped. Padding contributes neither geometry nor an
    # argument identity; it does not enlarge the concept store.
    capacity = len(space().similarity_codebook.W)
    missing = capacity - len(owner._bank_keys)
    if missing < 0:
        raise ValueError('forecast bank exceeds its constructed inventory')
    from torch.nn.functional import pad
    return (pad(owner._bank_keys, (0,0,0,missing)),
            pad(owner._bank_orders, (0,missing), value=-1),
            pad(owner._bank_addresses, (0,missing), value=-1))


def clock(language, like, bank):
    """A tensor-only time read, invariant during this generate walk."""
    if not bank:
        return None
    model = language.parts_of_speech.owner._model
    return model.present_code().to(like).expand(len(like), -1)


def forecast(language, emitted, count, *, orders=None, now=None, bank):
    """The shared head predicts the next place from typed emitted terminals."""
    from Expectation import ActivationPredictor, SymbolRead, symbol_activations
    if not bank:
        return None
    owner = language.parts_of_speech.owner._model.symbolSpace.expectation
    native, key_orders, addresses = bank
    keys = native.to(emitted)[None].expand(len(emitted), -1, -1)
    occupied = torch.arange(emitted.shape[1], device=emitted.device)[None] < count[:, None]
    orders = torch.zeros_like(occupied, dtype=torch.long) if orders is None else orders
    context = ActivationPredictor.context_by_order(emitted, occupied, orders, owner._bank_order_count)
    source = symbol_activations(keys, emitted, occupied, key_orders=key_orders,
                                value_orders=orders)
    inputs = (keys, context, source) if now is None else (keys, context, source, now)
    logits = owner._inter_predictor(*owner._prediction_inputs(inputs, orders=key_orders))
    read = SymbolRead(keys, logits.detach().sigmoid(), 0., 1e-6, 'clip', addresses)
    return language.parts_of_speech.expected_arguments(read, emitted)
