"""Read-only codebook capabilities and ownership versions for pipeline spaces.

Pipeline stages consume immutable SubSpace carriers directly. The completed
legacy snapshot conversion adapters were retired in the October 1 suite trim.
"""

from __future__ import annotations

import torch



class SpaceCarrierMixin:
    # -- sparse carrier compatibility -----------------------------------
    # Contract: doc/plans/2026-07-16-sparse-subspace-carrier-design.md
    #
    # The legacy ``self.subspace`` remains the internal adapter during the
    # migration.  New callers receive only immutable pipeline values and
    # capability readers; no Basis/Parameter is attached to those values.


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
