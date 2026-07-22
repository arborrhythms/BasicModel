"""Legacy Space adapters for immutable sparse pipeline carriers.

The value and executor layer lives in pipeline.py and deliberately does not
depend on the legacy model. This mixin is the narrow migration seam in the
other direction: it lets a durable Space issue read-only codebook
capabilities, snapshot its current legacy result into a fresh carrier, and
adapt a carrier for a not-yet-migrated Space method.

New stage implementations should consume and return pipeline.SubSpace
directly. The conversion methods here are compatibility code scheduled for
removal in Phase 6 of the sparse-carrier design.
"""

from __future__ import annotations

import torch

from util import TheDevice


class SpaceCarrierMixin:
    # -- sparse carrier compatibility -----------------------------------
    # Contract: doc/plans/2026-07-16-sparse-subspace-carrier-design.md
    #
    # The legacy ``self.subspace`` remains the internal adapter during the
    # migration.  New callers receive only immutable pipeline values and
    # capability readers; no Basis/Parameter is attached to those values.

    @property
    def carrier_schema(self):
        """Immutable schema shared by carriers emitted from this Space."""
        from pipeline import SubSpaceSchema

        schema = getattr(self, '_carrier_schema', None)
        if schema is None:
            schema = SubSpaceSchema(
                role=str(self.config_section or self.__class__.__name__),
                n_what=int(self.nWhat),
                n_where=int(self.nWhere),
                n_when=int(self.nWhen),
                geometry=(
                    "sphere"
                    if bool(getattr(self, "use_dot_product", False))
                    else "euclidean"
                ),
            )
            object.__setattr__(self, '_carrier_schema', schema)
        return schema

    @property
    def codebook_parameter_version(self):
        """Current content epoch advertised by this Space's readers."""
        return int(self._codebook_parameter_version)

    def bind_codebook_owner_path(self, owner_path):
        """Bind the stable model path used in future reader identities.

        Model assembly may call this once it knows the final ``named_modules``
        path (important for repeated conceptual/whole towers).  Rebinding is
        rejected after a different path has been published.
        """
        owner_path = str(owner_path).strip()
        if not owner_path:
            raise ValueError("codebook owner path must not be empty")
        current = self._codebook_owner_path
        if current is not None and current != owner_path:
            raise RuntimeError(
                f"codebook owner path already bound to {current!r}; "
                f"cannot rebind to {owner_path!r}"
            )
        self._codebook_owner_path = owner_path

    def set_codebook_parameter_version(self, version):
        """Set the executor-owned parameter epoch monotonically."""
        version = int(version)
        if version < self._codebook_parameter_version:
            raise ValueError(
                f"codebook parameter version cannot regress from "
                f"{self._codebook_parameter_version} to {version}"
            )
        self._codebook_parameter_version = version
        return version

    def mark_codebook_parameters_changed(self):
        """Advance the content epoch after an optimizer/EMA update barrier."""
        return self.set_codebook_parameter_version(
            self._codebook_parameter_version + 1
        )

    def mark_codebook_structure_changed(self, role=None):
        """Advance structural and content epochs for an owner-side mutation."""
        role, _ = self._carrier_basis(role)
        versions = self._codebook_structure_versions
        versions[role] = int(versions.get(role, 0)) + 1
        self.mark_codebook_parameters_changed()
        return self.codebook_identity(role)

    def _carrier_basis(self, role=None):
        sub = self.subspace
        role = str(role or getattr(sub, 'codebook_slot', None) or '')
        if role not in ('event', 'what', 'where', 'when', 'activation'):
            raise LookupError(
                f"{self.__class__.__name__} has no codebook role {role!r}"
            )
        basis = getattr(sub, role, None)
        prototype = basis.prototype() if hasattr(basis, 'prototype') else None
        if not torch.is_tensor(prototype) or prototype.ndim != 2:
            raise LookupError(
                f"{self.__class__.__name__}.{role} is not a live codebook"
            )
        return role, basis

    def codebook_identity(self, role=None):
        """Return the non-storage identity for one owned codebook role."""
        from pipeline import CodebookIdentity

        role, _ = self._carrier_basis(role)
        base = (
            self._codebook_owner_path
            or self.config_section
            or self.__class__.__name__
        )
        return CodebookIdentity(
            owner_path=f"{base}.{role}",
            structure_version=int(
                self._codebook_structure_versions.get(role, 0)
            ),
            parameter_version=self.codebook_parameter_version,
        )

    def codebook_reader(self, role=None):
        """Issue a read-only, weak capability for one Space-owned basis."""
        import weakref
        from pipeline import CodebookIdentity, make_codebook_reader

        role, basis = self._carrier_basis(role)
        owner_ref = weakref.ref(self)
        initial = self.codebook_identity(role)

        def identity():
            owner = owner_ref()
            if owner is None:
                raise RuntimeError(
                    f"codebook owner {initial.owner_path!r} no longer exists"
                )
            return CodebookIdentity(
                owner_path=initial.owner_path,
                structure_version=int(
                    owner._codebook_structure_versions.get(role, 0)
                ),
                parameter_version=owner.codebook_parameter_version,
            )

        return make_codebook_reader(
            basis,
            owner_path=initial.owner_path,
            identity=identity,
            use_dot_product=bool(getattr(basis, 'use_dot_product', False)),
        )

    @staticmethod
    def _carrier_activation(legacy_subspace):
        for method_name in ('effective_activation', 'activation_presence'):
            method = getattr(legacy_subspace, method_name, None)
            if callable(method):
                value = method()
                if torch.is_tensor(value):
                    return value
        return None

    @staticmethod
    def _carrier_band_from_index(legacy_subspace, indices, role, band_index):
        width = int(getattr(legacy_subspace, f'n{role.title()}', 0))
        if width <= 0:
            return None
        if (
            torch.is_tensor(indices)
            and indices.ndim == 3
            and indices.shape[-1] > band_index
        ):
            encoding = getattr(legacy_subspace, f'{role}Encoding', None)
            if encoding is not None and hasattr(encoding, 'encode'):
                return encoding.encode(indices[:, :, band_index])
        basis = getattr(legacy_subspace, role, None)
        value = basis.getW() if basis is not None and hasattr(basis, 'getW') else None
        return value if torch.is_tensor(value) and value.ndim >= 3 else None

    def to_pipeline_carrier(self, control, legacy_subspace=None, *, prior=None):
        """Snapshot a legacy result as an immutable sparse carrier.

        Codebook selections stay as indices whenever the legacy adapter has
        an integral ``_index``.  Dense fallback reads the ungated event once;
        it never installs that tensor as a cache on the new carrier.
        """
        from dataclasses import replace as dataclass_replace
        from pipeline import (
            DenseEvent,
            LossTerm,
            PipelineControl,
            PipelineEffects,
            ReverseTrace,
            SelectedEvent,
            SelectionSlot,
            SubSpace as PipelineSubSpace,
        )

        if not isinstance(control, PipelineControl):
            raise TypeError("control must be pipeline.PipelineControl")
        legacy = legacy_subspace or self.subspace
        indices = getattr(legacy, '_index', None)
        slot = getattr(legacy, 'codebook_slot', None)
        integral = (
            torch.is_tensor(indices)
            and not torch.is_floating_point(indices)
            and indices.dtype != torch.bool
        )
        activation = self._carrier_activation(legacy)
        if torch.is_tensor(activation):
            activation = activation.clone()

        if slot in ('event', 'what') and integral:
            selected = (
                indices[:, :, 0] if indices.ndim == 3 else indices
            ).long().clone()
            if slot == 'event':
                payload = SelectedEvent(
                    self.codebook_reader('event'),
                    selected,
                    slot=SelectionSlot.EVENT,
                    activation=activation,
                )
            else:
                where = self._carrier_band_from_index(
                    legacy, indices, 'where', 1
                )
                when_index = 1 + (1 if int(getattr(legacy, 'nWhere', 0)) > 0 else 0)
                when = self._carrier_band_from_index(
                    legacy, indices, 'when', when_index
                )
                payload = SelectedEvent(
                    self.codebook_reader('what'),
                    selected,
                    slot=SelectionSlot.WHAT,
                    activation=activation,
                    where=where.clone() if torch.is_tensor(where) else None,
                    when=when.clone() if torch.is_tensor(when) else None,
                )
        else:
            event = legacy.materialize(mode='event')
            if not torch.is_tensor(event):
                event = torch.empty(0, 0, self.carrier_schema.event_width)
            payload = DenseEvent(event=event.clone(), activation=activation)

        legacy_valid = getattr(legacy, 'valid_mask', None)
        if control.valid_mask is None and torch.is_tensor(legacy_valid):
            control = dataclass_replace(
                control, valid_mask=legacy_valid.bool().clone()
            )

        losses = []
        errors = getattr(legacy, 'errors', None)
        if errors is not None and hasattr(errors, 'terms'):
            for name, value, weight, _space, category in errors.terms():
                if torch.is_tensor(value):
                    losses.append(
                        LossTerm(
                            name=str(name),
                            value=value,
                            weight=float(weight),
                            category=str(category),
                        )
                    )

        if prior is not None and not isinstance(prior, PipelineSubSpace):
            raise TypeError("prior must be pipeline.SubSpace")
        prior_effects = prior.effects if prior is not None else PipelineEffects()
        # The legacy Error object is pipeline-wide and therefore already
        # contains upstream terms.  Treat its loss snapshot as canonical;
        # retain only typed diagnostics/mutations from the prior carrier.
        effects = PipelineEffects(
            losses=tuple(losses) if losses else prior_effects.losses,
            diagnostics=prior_effects.diagnostics,
            deferred_mutations=prior_effects.deferred_mutations,
        )
        trace = prior.trace if prior is not None else ReverseTrace()
        return PipelineSubSpace(
            schema=self.carrier_schema,
            payload=payload,
            control=control,
            effects=effects,
            trace=trace,
        )

    def from_pipeline_carrier(self, carrier):
        """Build an unowned legacy input adapter from an immutable carrier."""
        from pipeline import MaterializeMode, SubSpace as PipelineSubSpace

        if not isinstance(carrier, PipelineSubSpace):
            raise TypeError("carrier must be pipeline.SubSpace")
        event = carrier.materialize(MaterializeMode.EVENT)
        if event is None:
            event = torch.empty(0, 0, carrier.schema.event_width)
        if event.ndim < 2:
            raise ValueError("pipeline event must have at least [N, D] dimensions")
        n_active = int(event.shape[-2])
        width = int(event.shape[-1])
        from Spaces import SubSpace as LegacySubSpace

        legacy = LegacySubSpace(
            inputShape=[n_active, width],
            outputShape=[n_active, width],
        )
        legacy.set_event(event.clone())
        activation = carrier.materialize(MaterializeMode.ACTIVATION)
        if torch.is_tensor(activation):
            legacy.set_activation(activation.clone())
        legacy.valid_mask = (
            carrier.control.valid_mask.clone()
            if torch.is_tensor(carrier.control.valid_mask)
            else None
        )
        for term in carrier.effects.losses:
            legacy.errors.add(
                term.name,
                term.value,
                weight=term.weight,
                space=carrier.schema.role,
                category=term.category,
            )
        return legacy

    def as_pipeline_stage(
        self,
        *,
        forward_call=None,
        reverse_call=None,
        output_selector=None,
        name=None,
        device=None,
        globally_ordered=False,
        pipeline_safe=False,
        training_safe=False,
        serialization_reason=None,
    ):
        """Wrap this Space as a bounded-executor compatibility stage.

        Dense work stays local to the call. The legacy result is snapshotted
        immediately into a fresh immutable carrier before this stage accepts
        its next microbatch, so queued outputs never alias ``self.subspace``.

        Payload isolation alone cannot prove that a legacy call leaves its
        durable codebook/recurrent state unchanged. ``pipeline_safe`` is
        therefore opt-in after that stage has been audited. Training has the
        additional ``training_safe`` gate for optimizer/EMA semantics.

        Legacy reverse is deliberately opt-in: many old reverse methods still
        consult last-forward caches and are not safe for several in-flight
        microbatches. Pass a ``reverse_call`` only after that Space's reverse
        state has moved into typed ``ReverseTrace`` entries.
        """
        from pipeline import PipelineStage, SubSpace as PipelineSubSpace

        if forward_call is None:
            forward_call = self.forward
        if not callable(forward_call):
            raise TypeError("forward_call must be callable")
        if reverse_call is not None and not callable(reverse_call):
            raise TypeError("reverse_call must be callable")
        if output_selector is not None and not callable(output_selector):
            raise TypeError("output_selector must be callable")

        def invoke(call, carrier):
            legacy_in = self.from_pipeline_carrier(carrier)
            result = call(legacy_in)
            if output_selector is not None:
                result = output_selector(result)
            if isinstance(result, PipelineSubSpace):
                return result
            if isinstance(result, (tuple, list)):
                raise TypeError(
                    f"{self.__class__.__name__} returned multiple values; "
                    f"provide output_selector"
                )
            if result is None:
                raise TypeError(
                    f"{self.__class__.__name__} returned None, expected a SubSpace"
                )
            return self.to_pipeline_carrier(
                carrier.control,
                legacy_subspace=result,
                prior=carrier,
            )

        def stage_forward(carrier):
            return invoke(forward_call, carrier)

        stage_reverse = None
        if reverse_call is not None:
            def stage_reverse(carrier):
                return invoke(reverse_call, carrier)

        if device is None:
            tensor = next(self.parameters(), None)
            if tensor is None:
                tensor = next(self.buffers(), None)
            device = tensor.device if tensor is not None else TheDevice.get()
        return PipelineStage(
            name=str(name or self.config_section or self.__class__.__name__),
            forward=stage_forward,
            reverse=stage_reverse,
            device=device,
            globally_ordered=bool(globally_ordered),
            parameter_owner=self,
            pipeline_safe=bool(pipeline_safe),
            training_safe=bool(training_safe),
            serialization_reason=(
                serialization_reason
                or (
                    None
                    if training_safe
                    else "legacy Space may mutate EMA/recurrent state in forward"
                )
            ),
        )
