"""Owned grammatical meaning values and explicit physical-layout adapters.

This module owns no memory, executor, or learned parameters. Integer references
are addresses beside semantic payloads, never numeric features of those payloads.
"""
from dataclasses import dataclass, field
from collections.abc import Mapping
import math
from typing import Any

import torch
from torch.nn import functional as F






@dataclass(frozen=True)
class ClosingImage:
    """An attenuated view beside raw identity, prior and two residual signs."""
    observed: torch.Tensor
    conceived: torch.Tensor
    image: torch.Tensor
    concept_width: int
    gain: object = None
    expected: object = None
    excess: object = None
    omission: object = None
    identities: object = None
    row_expected: object = None

    @classmethod
    def form(cls, observed, estimate, confidence, *, prediction=None, occupied=None, identities=None, withheld=None, lanes=None, activation=None, epsilon=1e-6,
             saturation='clip', gain=0., object_mask=None):
        from Expectation import gain_field, symbol_activations, match_symbols
        raw = observed.detach().clone()
        if prediction is None:
            if estimate is not None:
                raise ValueError('closing estimates require an explicit symbol-bank prediction')
            return cls(raw, raw.clone(), torch.zeros_like(raw), raw.shape[-1],
                       raw.new_ones(raw.shape[0]))
        keys = prediction.symbol_keys.to(raw)
        occupied = raw.abs().sum(-1)>0 if occupied is None else occupied
        activation = (symbol_activations(keys[None], raw[None], occupied[None],
                                       None if lanes is None else lanes[None], identities=None if identities is None else identities[None],
                                       bank_ids=prediction.symbol_ids, key_valid=None if getattr(prediction, 'key_valid', None) is None else prediction.key_valid[None])[0]
                      if activation is None else activation.detach().to(raw))
        if activation.shape != (len(keys), 2):
            raise ValueError('retained activation target must align with its prior bank')
        expected = prediction.symbol_logits.detach().to(raw).sigmoid()
        prior_withheld = withheld_symbols(prediction, prediction)
        if prior_withheld is not None:
            expected = expected * (1.-prior_withheld[:, None])
        indices, matched = match_symbols(keys[None], raw[None],
            identities=None if identities is None else identities[None], bank_ids=prediction.symbol_ids,
            key_valid=None if getattr(prediction, 'key_valid', None) is None else prediction.key_valid[None])
        indices, matched = indices[0], matched[0]
        exempt = (torch.zeros(len(keys), device=raw.device, dtype=raw.dtype)
                  if prior_withheld is None else prior_withheld.to(raw).clone())
        role_exempt = object_mask
        if role_exempt is not None:
            exempt.scatter_reduce_(0, indices, role_exempt.to(raw)*matched, reduce='amax')
        if withheld is not None:
            exempt = torch.maximum(exempt, withheld.to(exempt))
        suppressed = exempt.new_zeros(exempt.shape) if prior_withheld is None else prior_withheld.to(exempt)
        if withheld is not None:
            suppressed = torch.maximum(suppressed, withheld.to(suppressed))
        confidence = torch.as_tensor(confidence, device=raw.device, dtype=raw.dtype)
        if confidence.shape != (len(keys),):
            raise ValueError('closing confidence must align with native symbol identities')
        field = gain_field(activation, expected, strength=gain, confidence=confidence,
            exempt=exempt, withheld=suppressed, epsilon=epsilon, saturation=saturation)
        applied = torch.where(matched & occupied, field.gain[indices], 1.)
        content = raw*applied[:, None]
        return cls(raw, content, content-raw, raw.shape[-1], applied,
                   field.expected, field.excess, field.omission, prediction.symbol_ids)

    def restore(self):
        # At gain 1 a confirmed identity may have zero content. Its retained
        # observation, never division by gain, makes restoration lossless.
        return self.observed.clone()

    def awareness(self):
        return dict(observation=self.observed, content=self.conceived,
                    expectation=self.expected, row_expectation=self.row_expected,
                    identities=self.identities)




def canonical_role_payload(payload, depth, layout, role_mask=None, *, concept_dim=None):
    """Return infix NP1/VP/NP2 roles and their explicit occupancy mask.

    STM is newest first: three roles map by [1, 2, 0], two by [1, 0].
    ``role_mask`` is already in canonical order, not physical STM order.
    Only unoccupied padding may be non-finite; it has no semantic value.
    """
    if layout not in ("stm", "infix"):
        raise ValueError("meaning layout must be stm or infix")
    if not torch.is_tensor(payload) or payload.ndim != 2 or payload.shape[-1] < 1:
        raise ValueError("meaning payload must have shape [slots, concept_dim]")
    if concept_dim is not None and payload.shape[-1] != int(concept_dim):
        raise ValueError("meaning payload must have shape [slots, concept_dim]")
    depth = int(depth)
    if not 1 <= depth <= 3 or depth > payload.shape[0]:
        raise ValueError("local meaning depth must be 1, 2 or 3")
    value = payload[:depth]
    if not value.is_floating_point():
        value = value.float()
    if layout == "stm":
        value = value[[1, 2, 0]] if depth == 3 else value.flip(0)
    value = F.pad(value, (0, 0, 0, 3 - depth))
    if role_mask is None:
        occupied = torch.arange(3, device=value.device) < depth
    else:
        occupied = torch.as_tensor(role_mask, device=value.device, dtype=torch.bool)
        if occupied.shape != (3,):
            raise ValueError("canonical role mask must have shape [3]")
        if bool(occupied[depth:].any()):
            raise ValueError("occupied role has no supplied payload")
    value = torch.where(occupied[:, None], value, torch.zeros_like(value))
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("non-finite occupied meaning role")
    return value, occupied


def _freeze_metadata(value):
    """Own semantic addresses/scope without tensors or mutable containers."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("semantic metadata mapping keys must be strings")
        return tuple((key, _freeze_metadata(item)) for key, item in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError("semantic metadata must contain finite scalar values or typed references")


@dataclass(frozen=True, eq=False)
class ConceptualMeaning:
    """One full-width grammatical description; occurrence ownership is external.

    Cloning owns tensors without detaching the current computation. Durable
    stores detach on write. Mode does not establish truth; a fact's evidential
    degree and source belong to its stored occurrence, not the concept vector.
    """
    roles: torch.Tensor = field(repr=False)
    role_mask: torch.Tensor = field(repr=False)
    mode: str = "assertive"
    polarity: bool = True
    role_refs: tuple = (None, None, None)
    bindings: Any = ()
    scope: Any = ()
    constituents: tuple = field(default=(), repr=False)
    sentence_kind: str | None = None

    def __post_init__(self):
        if not torch.is_tensor(self.roles) or self.roles.ndim != 2 or self.roles.shape[0] != 3:
            raise ValueError("canonical meaning roles must have shape [3, concept_dim]")
        roles, mask = canonical_role_payload(
            self.roles, 3, "infix", self.role_mask)
        if not bool(mask.any()):
            raise ValueError("a conceptual meaning requires an occupied role")
        if self.mode not in ("assertive", "interrogative", "unspecified"):
            raise ValueError(f"unknown grammatical mode {self.mode!r}")
        if self.sentence_kind not in (None, 'idea', 'relation'):
            raise ValueError('sentence kind must be idea, relation or unspecified')
        refs = _freeze_metadata(self.role_refs)
        if not isinstance(refs, tuple) or len(refs) != 3:
            raise ValueError("role references must contain three canonical entries")
        for present, ref in zip(mask.tolist(), refs):
            if ref is not None and not present:
                raise ValueError("an absent role cannot carry a semantic reference")
        object.__setattr__(self, "roles", roles.clone())
        object.__setattr__(self, "role_mask", mask.clone())
        object.__setattr__(self, "polarity", bool(self.polarity))
        object.__setattr__(self, "role_refs", refs)
        object.__setattr__(self, "bindings", _freeze_metadata(() if self.bindings is None else self.bindings))
        object.__setattr__(self, "scope", _freeze_metadata(() if self.scope is None else self.scope))
        children = tuple(self.constituents)
        if any(not isinstance(child, ConceptualMeaning)
               or child.roles.shape[-1] != roles.shape[-1] for child in children):
            raise ValueError("constituents must be complete meanings of the same width")
        object.__setattr__(self, "constituents", children)

    @classmethod
    def from_payload(cls, payload, *, depth, layout, role_mask=None, **metadata):
        roles, occupied = canonical_role_payload(payload, depth, layout, role_mask)
        return cls(roles, occupied, **metadata)

    @classmethod
    def from_description(cls, value):
        """Explicit legacy interface: a vector is NP1; a matrix is infix.

        Never flatten multiple roles or batches into an apparent unary concept.
        Callers holding physical STM must use ``from_payload(layout='stm')``.
        """
        if isinstance(value, cls):
            return value
        payload = value if torch.is_tensor(value) else torch.as_tensor(value, dtype=torch.float32)
        if payload.ndim == 1:
            payload = payload[None]
        if payload.ndim != 2 or not 1 <= payload.shape[0] <= 3:
            raise ValueError("description must be a concept vector or 1–3 infix roles")
        return cls.from_payload(payload, depth=payload.shape[0], layout="infix")

    @property
    def has_context(self):
        return bool(self.bindings or self.scope or any(ref is not None for ref in self.role_refs))

    def metadata(self):
        """Tensor-free checkpoint metadata; roles remain in the existing store."""
        return {"mode": self.mode, "polarity": self.polarity,
                "role_refs": self.role_refs, "bindings": self.bindings, "scope": self.scope,
                "sentence_kind": self.sentence_kind}

    def detached(self):
        return type(self)(self.roles.detach(), self.role_mask, **self.metadata(),
                          constituents=tuple(child.detached() for child in self.constituents))


def symbol_identities(meaning):
    """Typed native handles beside an observation; absent handles stay unknown."""
    named = dict(meaning.bindings).get('_symbol_handles')
    if named is None:
        named = [ref[1] if isinstance(ref, tuple) and len(ref)==2 and ref[0]=='sym' else -1
                 for ref in meaning.role_refs]
    return torch.tensor(named, device=meaning.roles.device, dtype=torch.long)


def retained_symbol_prediction(meaning):
    """Rehydrate a complete activation prior from an estimate occurrence."""
    from types import SimpleNamespace
    entries = dict(meaning.bindings).get('_expected_symbols')
    if not entries:
        return None
    ids, orders, keys, logits = zip(*entries)
    like = meaning.roles
    valid = torch.tensor([key is not None for key in keys], device=like.device, dtype=torch.bool)
    values = like.new_zeros(len(keys), like.shape[-1])
    if bool(valid.any()):
        values[valid] = like.new_tensor([key for key in keys if key is not None])
    return SimpleNamespace(symbol_keys=values, key_valid=valid, symbol_logits=like.new_tensor(logits),
        symbol_ids=torch.tensor(ids, device=like.device, dtype=torch.long),
        symbol_orders=torch.tensor(orders, device=like.device, dtype=torch.long), bindings=meaning.bindings,
        confidence=like.new_tensor(logits).sigmoid().amax(-1))


def retained_symbol_activation(meaning, prediction):
    """Read the original local target after the stored row has been fused."""
    entries = dict(meaning.bindings).get('_observed_symbol_activations')
    if entries is None or prediction is None:
        return None
    values = {int(identity): pair for identity, pair in entries}
    return prediction.symbol_keys.new_tensor([
        values.get(int(identity), (0., 0.)) for identity in prediction.symbol_ids.tolist()])


def withheld_symbols(meaning, prediction):
    """A non modifier's explicit retained identity has no prior at this level."""
    marks = dict(meaning.bindings).get('_withheld_symbols', ())
    keys = prediction.symbol_keys
    if keys is None:
        return None
    mask = torch.zeros(len(keys), device=keys.device, dtype=keys.dtype)
    if marks:
        values = keys.new_tensor([point for _identity, point in marks])
        ids = prediction.symbol_ids
        marked_ids = ids.new_tensor([identity for identity, _point in marks])
        handles = (ids[:,None] == marked_ids[None]) & (ids[:,None] != -1) & (marked_ids[None] != -1)
        distances = torch.cdist(keys, values, compute_mode='donot_use_mm_for_euclid_dist')
        exact = distances <= 1e-5
        valid = getattr(prediction, 'key_valid', None)
        if valid is not None:
            exact &= valid[:,None]
        mask = (handles | exact).any(-1).to(keys)
    return mask
