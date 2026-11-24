"""Cut completed sentence writes while preserving the batch perception graph."""
from dataclasses import fields, is_dataclass, replace
from types import MappingProxyType
import torch
from Meaning import frozen_metadata


def detached(value):
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {key: detached(item) for key, item in value.items()}
    if isinstance(value, MappingProxyType):
        return MappingProxyType({key: detached(item) for key, item in value.items()})
    if isinstance(value, tuple) and hasattr(value, '_fields'):
        return type(value)(*(detached(item) for item in value))
    if type(value) is tuple and frozen_metadata(value):
        return value
    if isinstance(value, (tuple, list)):
        return type(value)(detached(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{f.name: detached(getattr(value, f.name)) for f in fields(value) if f.init})
    return value


def finish(model, sub, sym, perception_sub, perception_sym, written, perception_presences):
    """Only sentence-produced state is severed; pre-loop slabs remain live."""
    trace = model._reconstruction_stack()
    for owner, names in ((trace, ('_choice_values', '_choice_action_records',
            '_choice_log_probabilities', '_forward_losses', '_chunk_state', '_case_state')),
        (model, ('_tensor_pushed_ideas', '_tensor_sentence_roots_live', '_packed_sentence_roots',
            '_attention_words', '_attention_greedy', '_word_expectation',
            '_last_decoder_trace', '_last_output_walk_trace',
            '_last_decoder_comparison', '_compose_sampling_scale',
            '_sentence_expectation_comparison', '_sentence_gradient_objectives',
            '_sentence_open_fields', '_sentence_look_fields', '_sentence_fields',
            '_input_reconstructions',
            '_sentence_order_zero_reverse', '_sentence_target', '_last_sentence_credit',
            '_sentence_credit_audits', '_sentence_walk_residuals',
            '_sentence_output_walk_credits', '_sentence_order_zero_field', '_last_answer_construction',
            '_last_closing_thoughts', '_intuition_bracket',
            '_concept_prediction'))):
        for name in names:
            if hasattr(owner, name):
                setattr(owner, name, detached(getattr(owner, name)))

    # Reverse carriers are modules, so recursive value detachment does not
    # traverse their transient native activity. These are sentence writes.
    for carrier in (getattr(model, '_sentence_order_zero_reverse', None),):
        if carrier is not None:
            for name in ('_radix_activity', '_reconstruction_percepts'):
                if hasattr(carrier, name):
                    setattr(carrier, name, detached(getattr(carrier, name)))

    for space in model.conceptualSpaces:
        stm = getattr(space, 'stm', None)
        if stm is not None:
            from Memory import STMChunks
            # Include inactive capacity rows: set_state would concatenate
            # their old graph back onto the detached active prefix.
            if hasattr(stm, 'strata'):
                for store in stm.strata.recent_stores:
                    store.slots = store.slots.detach()
            for field in STMChunks.FIELDS:
                name = '_live_' + field
                stm._assign_live(stm, name, getattr(stm, name).detach())
        carrier = space.subspace
        for name in ('_concept_activations', '_concept_codes'):
            if hasattr(carrier, name):
                setattr(carrier, name, detached(getattr(carrier, name)))
        for name in ('_percept_field', '_cs_open_field', '_cs_field_wires'):
            if hasattr(space, name):
                setattr(space, name, detached(getattr(space, name)))
    if getattr(model.percept_expectation, 'last_arrival', None) is not None:
        model.percept_expectation.last_arrival = detached(model.percept_expectation.last_arrival)
    # Publishing the chosen syntax replaces this slab with sentence writes.
    # Preserve any pre-loop perception values at positions not yet written.
    presences = model.symbolSpace._word_reference_presences
    if torch.is_tensor(presences):
        model.symbolSpace._word_reference_presences = (presences.detach() if perception_presences is None else
            torch.where(written[..., None], presences.detach(), perception_presences))

    def cut(state, perception, *, word_slab=False):
        result = []
        for index, (value, original) in enumerate(zip(state, perception, strict=True)):
            mask = written if word_slab and index == 1 else written.any(-1)
            while mask.ndim < value.ndim:
                mask = mask.unsqueeze(-1)
            result.append(torch.where(mask, value.detach(), original))
        return tuple(result)
    return cut(sub, perception_sub, word_slab=True), cut(sym, perception_sym)
