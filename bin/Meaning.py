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


@torch.no_grad()
def negative_image(observed, estimate, presence, *, gain=1., object_mask=None):
    """Seal-time subtraction, detached from all reading and policy choices.

    These are derived serial values, never writes to a presence field or a
    third memory record. The predictor is trained separately against the
    complete observation, including empty roles, irrespective of this gain.
    """
    if observed.ndim != 2 or observed.shape[0] != 3:
        raise ValueError("negative image requires three canonical roles")
    gain = torch.as_tensor(gain, device=observed.device, dtype=observed.dtype).detach()
    mask = (torch.zeros(3, device=observed.device, dtype=observed.dtype)
            if object_mask is None else torch.as_tensor(
                object_mask, device=observed.device, dtype=observed.dtype).detach())
    if gain.numel() != 1 or mask.shape != (3,):
        raise ValueError("expectation gain is scalar and object mask has three roles")
    if not bool(torch.isfinite(gain).all() and ((gain >= 0) & (gain <= 1)).all()
                and torch.isfinite(mask).all() and ((mask >= 0) & (mask <= 1)).all()):
        raise ValueError("expectation gain and object mask must be in [0, 1]")
    if estimate is None:
        return observed.detach().clone(), torch.zeros_like(observed)
    presence = torch.as_tensor(presence, device=observed.device, dtype=observed.dtype).detach()
    if estimate.shape != observed.shape or presence.shape != (3,):
        raise ValueError("estimate and presence must align with the three roles")
    if not bool(torch.isfinite(estimate).all() and torch.isfinite(presence).all()
                and ((presence >= 0) & (presence <= 1)).all()):
        raise ValueError("estimate must be finite and presence in [0, 1]")
    image = -gain * ((1 - mask) * presence)[:, None] * estimate.detach().to(observed)
    return observed.detach() + image, image


@torch.no_grad()
def expectation_surprise(observed, estimate):
    """Bound the honest all-role mean squared residual by s/(1+s).

    This monotone normalization leaves an exact prediction at zero and makes
    surprise comparable to other retention terms in [0, 1]. Neither gain nor
    the active question enters it. Unknown surprise is stored separately as -1.
    """
    error = (observed.to(estimate) - estimate).square().mean()
    return error / (1 + error)


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

    def __post_init__(self):
        if not torch.is_tensor(self.roles) or self.roles.ndim != 2 or self.roles.shape[0] != 3:
            raise ValueError("canonical meaning roles must have shape [3, concept_dim]")
        roles, mask = canonical_role_payload(
            self.roles, 3, "infix", self.role_mask)
        if not bool(mask.any()):
            raise ValueError("a conceptual meaning requires an occupied role")
        if self.mode not in ("assertive", "interrogative", "unspecified"):
            raise ValueError(f"unknown grammatical mode {self.mode!r}")
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
                "role_refs": self.role_refs, "bindings": self.bindings, "scope": self.scope}

    def detached(self):
        return type(self)(self.roles.detach(), self.role_mask, **self.metadata(),
                          constituents=tuple(child.detached() for child in self.constituents))
