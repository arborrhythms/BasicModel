"""Versioned, side-effect-free checkpoint migrations.

The checkpoint loader deliberately keeps migration policy out of the model
graph.  Functions in this module operate on ordinary mappings and optimizer
objects; they never write the source artifact and never mutate tensors.

Schema 2 establishes the forward-path ownership rule that WholeSpace contains
only analytic whole-properties.  Pre-schema checkpoints used the terminal
WholeSpace ``what`` Codebook as a concept/symbol inventory and also carried a
second, concept-sized ``analysis_store``.  Those tensors cannot be prefix-
loaded into the property basis: even rows whose shapes happen to match have a
different meaning.  The migration therefore omits them so the model's
deterministic property initialization survives ``load_state_dict``.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping


CHECKPOINT_SCHEMA_KEY = "checkpoint_schema"
CHECKPOINT_SCHEMA_VERSION = 2
WHOLESPACE_ROLE = "property_basis"
LEGACY_WHOLE_STRUCTURE_KEY = "legacy_whole_structure"
OPTIMIZER_PARAM_NAMES_KEY = "optimizer_param_names"


# Exact module boundaries matter here.  In particular,
# ``wholeSpaces.N.conceptualSpace.*`` is a historical registered alias of a
# ConceptualSpace and must remain checkpoint material.
_WHOLE_DENSE_RE = re.compile(
    r"^(?:wholeSpaces\.\d+|wholeSpace|body_stages\.\d+\.ws)"
    r"(?:\._owned_bases\.what|\.subspace\.what|\.what|\.analysis_store)"
    r"(?:\.|$)"
)
_WHOLE_PARAMETER_RE = re.compile(
    r"^(?:wholeSpaces\.\d+|wholeSpace|body_stages\.\d+\.ws)"
    r"(?:\._owned_bases\.what|\.subspace\.what|\.what|\.analysis_store)"
    r"\.W$"
)
_CANONICAL_WHOLE_PARAMETER_RE = re.compile(
    r"^(wholeSpaces\.\d+)"
    r"(\._owned_bases\.what|\.subspace\.what|\.what|\.analysis_store)"
    r"\.W$"
)

# Before aligned ConceptualSpace folds shared one dictionary, every stage
# registered its own similarity Codebook twice: once below ``layers.K`` (the
# canonical ``named_parameters`` path) and once as ``similarity_codebook`` (a
# state-dict alias).  The optimizer saw only the canonical ``layers.K.W``
# Parameter.  Keep the layer index generic because optional parameter-free
# layers can move the Codebook within the ModuleList.
_CANONICAL_CONCEPT_PARAMETER_RE = re.compile(
    r"^(conceptualSpaces\.(\d+))\.layers\.(\d+)\.W$"
)

# The boundary-growth implementation made the radix PartSpace table an
# explicit optimizer Parameter.  Manifest-less checkpoints predate that
# enlistment: they contain the registered tensor in ``state_dict`` but it was
# absent from the optimizer's parameter group.  A checkpoint produced after
# enlistment always carries a name manifest, so excluding it is specific to
# this legacy-inference path.
_PRE_MANIFEST_UNOPTIMIZED_PARAMETERS = frozenset({
    "perceptualSpace._owned_bases.what.W",
})


def is_legacy_wholespace_dense_key(name: str) -> bool:
    """Return whether *name* belongs to the retired WS dense inventories."""

    return bool(_WHOLE_DENSE_RE.match(str(name)))


def is_reset_wholespace_parameter(name: str) -> bool:
    """Return whether optimizer moments for *name* must start fresh."""

    return bool(_WHOLE_PARAMETER_RE.match(str(name)))


def _tensor_nbytes(value: Any) -> int:
    numel = getattr(value, "numel", None)
    element_size = getattr(value, "element_size", None)
    if not callable(numel) or not callable(element_size):
        return 0
    try:
        return int(numel()) * int(element_size())
    except (TypeError, ValueError, RuntimeError):
        return 0


def _shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(v) for v in shape)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class CheckpointMigrationDiagnostics:
    """Auditable summary of a schema migration."""

    source_version: int
    target_version: int
    dropped_state_keys: tuple[str, ...] = ()
    dropped_state_bytes: int = 0
    quarantined_fields: tuple[str, ...] = ()

    @property
    def migrated(self) -> bool:
        return self.source_version != self.target_version or bool(
            self.dropped_state_keys or self.quarantined_fields)

    def messages(self) -> tuple[str, ...]:
        lines = []
        if self.dropped_state_keys:
            mib = self.dropped_state_bytes / (1024.0 * 1024.0)
            lines.append(
                "checkpoint schema migration reset "
                f"{len(self.dropped_state_keys)} obsolete WholeSpace dense "
                f"tensor(s) ({mib:.1f} MiB); live property rows retain "
                "deterministic initialization"
            )
        if self.quarantined_fields:
            lines.append(
                "checkpoint schema migration quarantined legacy WholeSpace "
                "structure for ConceptualSpace import: "
                + ", ".join(self.quarantined_fields)
            )
        return tuple(lines)


@dataclass(frozen=True)
class CheckpointMigrationResult:
    """Migrated mapping plus metadata needed for optimizer restoration."""

    checkpoint: Mapping[str, Any]
    diagnostics: CheckpointMigrationDiagnostics
    legacy_state_shapes: Mapping[str, tuple[int, ...]]

    @property
    def migrated(self) -> bool:
        return self.diagnostics.migrated


def checkpoint_schema() -> dict[str, Any]:
    """Return a fresh schema-2 marker suitable for a saved bundle."""

    return {
        "version": CHECKPOINT_SCHEMA_VERSION,
        "whole_space_role": WHOLESPACE_ROLE,
    }


def stamp_checkpoint_schema(
    checkpoint: Mapping[str, Any],
    *,
    optimizer_param_names: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a shallow-stamped bundle without modifying *checkpoint*."""

    out = dict(checkpoint)
    out[CHECKPOINT_SCHEMA_KEY] = checkpoint_schema()
    if optimizer_param_names is not None:
        out[OPTIMIZER_PARAM_NAMES_KEY] = optimizer_param_names
    return out


def _source_schema_version(checkpoint: Mapping[str, Any]) -> int:
    marker = checkpoint.get(CHECKPOINT_SCHEMA_KEY)
    if not isinstance(marker, Mapping):
        return 1
    try:
        return int(marker.get("version", 1) or 1)
    except (TypeError, ValueError):
        raise ValueError(
            f"invalid {CHECKPOINT_SCHEMA_KEY}.version: "
            f"{marker.get('version')!r}"
        )


def _quarantine_legacy_whole_structure(
    out: dict[str, Any],
) -> tuple[str, ...]:
    """Move active legacy WS sidecars into a non-restored quarantine."""

    quarantine: dict[str, Any] = {"version": 1}
    fields: list[str] = []

    vocab = out.get("vocab_extras")
    if isinstance(vocab, Mapping):
        new_vocab = dict(vocab)
        old_vocab: dict[str, Any] = {}
        for key in ("well_known_atoms", "ws_taxonomy_extras"):
            if key in new_vocab:
                old_vocab[key] = new_vocab.pop(key)
                fields.append(f"vocab_extras.{key}")
        if old_vocab:
            quarantine["vocab_extras"] = old_vocab
            out["vocab_extras"] = new_vocab

    structural = out.get("structural_extras")
    if isinstance(structural, Mapping) and "whole_spaces" in structural:
        new_structural = dict(structural)
        quarantine["structural_whole_spaces"] = new_structural.pop(
            "whole_spaces"
        )
        fields.append("structural_extras.whole_spaces")
        out["structural_extras"] = new_structural

    if len(quarantine) > 1:
        previous = out.get(LEGACY_WHOLE_STRUCTURE_KEY)
        if isinstance(previous, Mapping):
            merged = dict(previous)
            merged.update(quarantine)
            quarantine = merged
        out[LEGACY_WHOLE_STRUCTURE_KEY] = quarantine
    return tuple(fields)


def migrate_wholespace_checkpoint(
    checkpoint: Mapping[str, Any],
    live_state_dict: Mapping[str, Any] | None = None,
) -> CheckpointMigrationResult:
    """Migrate one checkpoint to the properties-only WholeSpace schema.

    The function is intentionally non-destructive: container mappings are
    shallow-copied, retained tensor objects are shared, and no tensor is ever
    modified.  ``live_state_dict`` is accepted as part of the stable loader
    API and lets diagnostics distinguish keys whose destination disappeared;
    semantic reset does not depend on shape because equal-shaped legacy rows
    are still concept/symbol rows rather than properties.

    Raw, pre-bundle state dicts are supported.  They cannot carry a schema
    marker, but their retired keys are still omitted.
    """

    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    source_version = _source_schema_version(checkpoint)
    if source_version > CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"checkpoint schema {source_version} is newer than supported "
            f"schema {CHECKPOINT_SCHEMA_VERSION}"
        )

    is_bundle = "state_dict" in checkpoint
    state_obj = checkpoint.get("state_dict") if is_bundle else checkpoint
    if not isinstance(state_obj, Mapping):
        raise TypeError("checkpoint state_dict must be a mapping")
    legacy_shapes = {
        str(key): shape
        for key, value in state_obj.items()
        if (shape := _shape(value)) is not None
    }

    marker = checkpoint.get(CHECKPOINT_SCHEMA_KEY)
    if (source_version == CHECKPOINT_SCHEMA_VERSION
            and (not isinstance(marker, Mapping)
                 or marker.get("whole_space_role") != WHOLESPACE_ROLE)):
        raise ValueError(
            f"checkpoint schema {source_version} has unsupported "
            f"whole_space_role "
            f"{None if not isinstance(marker, Mapping) else marker.get('whole_space_role')!r}"
        )
    already_properties = (
        source_version == CHECKPOINT_SCHEMA_VERSION
        and isinstance(marker, Mapping)
        and marker.get("whole_space_role") == WHOLESPACE_ROLE
    )
    if already_properties:
        diagnostics = CheckpointMigrationDiagnostics(
            source_version=source_version,
            target_version=CHECKPOINT_SCHEMA_VERSION,
        )
        unchanged = dict(checkpoint)
        if is_bundle:
            # The loader performs further key canonicalization in-place.  Give
            # it an independent container even on an idempotent schema-2 load
            # so retrying/inspecting the caller's mapping stays safe.
            unchanged["state_dict"] = dict(state_obj)
        return CheckpointMigrationResult(
            checkpoint=unchanged,
            diagnostics=diagnostics,
            legacy_state_shapes=legacy_shapes,
        )

    dropped = tuple(
        str(key) for key in state_obj if is_legacy_wholespace_dense_key(key)
    )
    dropped_bytes = sum(_tensor_nbytes(state_obj[key]) for key in dropped)
    migrated_state = {
        key: value for key, value in state_obj.items()
        if not is_legacy_wholespace_dense_key(key)
    }

    if is_bundle:
        out: dict[str, Any] = dict(checkpoint)
        out["state_dict"] = migrated_state
        quarantined = _quarantine_legacy_whole_structure(out)
        out[CHECKPOINT_SCHEMA_KEY] = checkpoint_schema()
    else:
        # A raw state_dict must remain a raw state_dict; inserting metadata
        # would turn it into an unexpected model key.
        out = migrated_state
        quarantined = ()

    diagnostics = CheckpointMigrationDiagnostics(
        source_version=source_version,
        target_version=CHECKPOINT_SCHEMA_VERSION,
        dropped_state_keys=dropped,
        dropped_state_bytes=dropped_bytes,
        quarantined_fields=quarantined,
    )
    return CheckpointMigrationResult(
        checkpoint=out,
        diagnostics=diagnostics,
        legacy_state_shapes=legacy_shapes,
    )


# ---------------------------------------------------------------------------
# Optimizer name manifests and safe remapping


def _optimizer_objects(optimizer: Any) -> list[Any]:
    children = getattr(optimizer, "optimizers", None)
    if children is None:
        return [optimizer]
    leaves: list[Any] = []
    for child in children:
        leaves.extend(_optimizer_objects(child))
    return leaves


def _optimizer_state_leaves(state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    children = state.get("optimizers")
    if children is None:
        if "state" not in state or "param_groups" not in state:
            raise ValueError("malformed optimizer state")
        return [state]
    leaves: list[Mapping[str, Any]] = []
    for child in children:
        if not isinstance(child, Mapping):
            raise ValueError("malformed child optimizer state")
        leaves.extend(_optimizer_state_leaves(child))
    return leaves


def _manifest_leaves(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if int(manifest.get("version", 0) or 0) != 1:
        raise ValueError("unsupported optimizer parameter manifest version")
    leaves = manifest.get("leaves")
    if not isinstance(leaves, list):
        raise ValueError("optimizer parameter manifest has no leaves list")
    return leaves


def build_optimizer_param_manifest(
    optimizer: Any,
    named_parameters: Iterable[tuple[str, Any]],
) -> dict[str, Any]:
    """Describe optimizer parameter order by stable model names and shapes."""

    name_by_identity: dict[int, str] = {}
    for name, parameter in named_parameters:
        name_by_identity.setdefault(id(parameter), str(name))

    leaves = []
    for leaf in _optimizer_objects(optimizer):
        groups = []
        for group in getattr(leaf, "param_groups", ()):
            entries = []
            for parameter in group.get("params", ()):
                name = name_by_identity.get(id(parameter))
                if name is None:
                    raise ValueError(
                        "optimizer contains a parameter absent from "
                        "model.named_parameters()"
                    )
                entries.append({
                    "name": name,
                    "shape": list(_shape(parameter) or ()),
                })
            groups.append(entries)
        leaves.append({"param_groups": groups})
    return {"version": 1, "leaves": leaves}


def _entry_name(entry: Any) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, Mapping) and isinstance(entry.get("name"), str):
        return entry["name"]
    raise ValueError(f"invalid optimizer manifest entry: {entry!r}")


def _entry_shape(entry: Any) -> tuple[int, ...] | None:
    if not isinstance(entry, Mapping) or "shape" not in entry:
        return None
    try:
        return tuple(int(v) for v in entry["shape"])
    except (TypeError, ValueError):
        raise ValueError(f"invalid optimizer manifest shape: {entry!r}")


def _moment_shape(param_state: Mapping[str, Any]) -> tuple[int, ...] | None:
    shapes = set()
    for key, value in param_state.items():
        shape = _shape(value)
        if shape is None or len(shape) == 0:
            continue
        # Adam variants occasionally store a one-element tensor step.
        if str(key) == "step" and shape in ((), (1,)):
            continue
        shapes.add(shape)
    if not shapes:
        return None
    if len(shapes) != 1:
        raise ValueError(f"optimizer parameter has inconsistent moments: {shapes}")
    return shapes.pop()


def _insert_legacy_ws_parameter(
    names: list[str],
    parameter_name: str,
) -> None:
    """Insert one retired WS parameter at its version-1 optimizer anchor."""

    match = _CANONICAL_WHOLE_PARAMETER_RE.match(parameter_name)
    if match is None:
        raise ValueError(
            "cannot infer optimizer position for non-canonical legacy "
            f"WholeSpace parameter {parameter_name!r}"
        )
    prefix, role = match.groups()
    if "analysis_store" in role:
        anchors = (
            f"{prefix}._pi_stack_modules.",
            f"{prefix}.pi_stack.",
        )
        positions = [
            i for i, name in enumerate(names)
            if any(name.startswith(anchor) for anchor in anchors)
        ]
        if not positions:
            raise ValueError(
                f"cannot anchor retired {parameter_name!r} before WS pi stack"
            )
        names.insert(min(positions), parameter_name)
        return

    # The main codebook preceded every learned WholeSpace analysis layer.
    anchors = (
        f"{prefix}.layers.",
        f"{prefix}.pi.",
        f"{prefix}._pi_stack_modules.",
    )
    positions = [
        i for i, name in enumerate(names)
        if any(name.startswith(anchor) for anchor in anchors)
    ]
    if not positions:
        raise ValueError(
            f"cannot anchor retired {parameter_name!r} before WS layers"
        )
    names.insert(min(positions), parameter_name)


def _legacy_concept_dictionary_parameters(
    legacy_state_shapes: Mapping[str, tuple[int, ...]],
) -> list[str]:
    """Return old per-stage CS dictionary Parameters in stage order.

    A direct ``conceptualSpaces.N.layers.K.W`` is considered the historical
    similarity dictionary only when its registered compatibility alias exists
    with the same shape.  This avoids mistaking an unrelated direct ``W`` layer
    for a retired stage dictionary.
    """

    found: list[tuple[int, int, str]] = []
    for name, shape in legacy_state_shapes.items():
        match = _CANONICAL_CONCEPT_PARAMETER_RE.match(name)
        if match is None:
            continue
        stage = int(match.group(2))
        layer = int(match.group(3))
        alias = f"conceptualSpaces.{stage}.similarity_codebook.W"
        if legacy_state_shapes.get(alias) != shape:
            continue
        found.append((stage, layer, name))
    return [name for _stage, _layer, name in sorted(found)]


def _insert_legacy_concept_parameter(
    names: list[str],
    parameter_name: str,
) -> None:
    """Insert one retired stage dictionary at that CS stage's old tail."""

    match = _CANONICAL_CONCEPT_PARAMETER_RE.match(parameter_name)
    if match is None:
        raise ValueError(
            f"cannot infer optimizer position for non-canonical legacy "
            f"ConceptualSpace parameter {parameter_name!r}"
        )
    prefix = match.group(1) + "."

    # ``self.spaces`` historically visited all ConceptualSpaces before the
    # WholeSpaces.  Some terminal-CS modules are encountered again later via
    # SymbolSpace aliases, so restrict the anchor search to the initial CS
    # block (everything before the first WholeSpace parameter).
    whole_boundary = next(
        (i for i, name in enumerate(names) if name.startswith("wholeSpaces.")),
        len(names),
    )
    positions = [
        i for i, name in enumerate(names[:whole_boundary])
        if name.startswith(prefix)
    ]
    if not positions:
        raise ValueError(
            f"cannot anchor retired {parameter_name!r} at its "
            "ConceptualSpace stage"
        )
    names.insert(max(positions) + 1, parameter_name)


def infer_legacy_optimizer_param_manifest(
    live_manifest: Mapping[str, Any],
    legacy_state_shapes: Mapping[str, tuple[int, ...]],
    saved_optimizer_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove and reconstruct a missing schema-1 optimizer name manifest.

    Version-1 checkpoints did not save parameter names.  Inference is limited
    to one optimizer leaf with one parameter group—the layout used by the
    affected training run.  Unchanged live parameters are retained only when
    an exact legacy state key exists, except for parameters explicitly enlisted
    after manifest-bearing saves were introduced.  Retired per-stage CS and WS
    ``W`` parameters are then inserted at their documented module anchors.
    Finally every materialized Adam moment must agree with the inferred legacy
    parameter shape; a count or shape discrepancy raises rather than risking
    positional misassignment.
    """

    live_leaves = _manifest_leaves(live_manifest)
    saved_leaves = _optimizer_state_leaves(saved_optimizer_state)
    if len(live_leaves) != 1 or len(saved_leaves) != 1:
        raise ValueError(
            "legacy optimizer inference requires exactly one optimizer leaf"
        )
    live_groups = live_leaves[0].get("param_groups")
    saved_groups = saved_leaves[0].get("param_groups")
    if (not isinstance(live_groups, list) or len(live_groups) != 1
            or not isinstance(saved_groups, list) or len(saved_groups) != 1):
        raise ValueError(
            "legacy optimizer inference requires exactly one parameter group"
        )

    names = [
        _entry_name(entry) for entry in live_groups[0]
        if _entry_name(entry) in legacy_state_shapes
        and _entry_name(entry) not in _PRE_MANIFEST_UNOPTIMIZED_PARAMETERS
    ]

    # Aligned/property-basis models now share stage 0's CS dictionary across
    # every fold.  Restore the old independent-stage optimizer layout before
    # remapping by name: stage 0 remains live/canonical, while later stages are
    # present only in this inferred saved manifest and are therefore dropped
    # safely by ``remap_optimizer_state_by_name``.
    retired_concepts = [
        name for name in _legacy_concept_dictionary_parameters(
            legacy_state_shapes)
        if name not in names
    ]
    for name in retired_concepts:
        _insert_legacy_concept_parameter(names, name)

    retired = sorted(
        name for name in legacy_state_shapes
        if is_reset_wholespace_parameter(name) and name not in names
    )
    # Primary W must precede analysis_store W within each stage.  Sorting by
    # stage and role explicitly avoids depending on lexical underscore order.
    def _retired_order(name: str) -> tuple[int, int]:
        match = _CANONICAL_WHOLE_PARAMETER_RE.match(name)
        if match is None:
            return (1 << 30, 1 << 30)
        stage = int(match.group(1).split(".")[-1])
        role = 1 if "analysis_store" in match.group(2) else 0
        return stage, role

    for name in sorted(retired, key=_retired_order):
        _insert_legacy_ws_parameter(names, name)

    saved_ids = list(saved_groups[0].get("params", ()))
    if len(names) != len(saved_ids):
        raise ValueError(
            "cannot prove legacy optimizer layout: inferred "
            f"{len(names)} names for {len(saved_ids)} saved parameters"
        )

    saved_state = saved_leaves[0].get("state") or {}
    for parameter_id, name in zip(saved_ids, names):
        param_state = saved_state.get(parameter_id)
        if not isinstance(param_state, Mapping):
            continue
        moment_shape = _moment_shape(param_state)
        expected = legacy_state_shapes.get(name)
        if moment_shape is not None and moment_shape != expected:
            raise ValueError(
                "cannot prove legacy optimizer layout: saved parameter "
                f"{parameter_id} has moment shape {moment_shape}, but "
                f"inferred {name!r} has checkpoint shape {expected}"
            )

    entries = [
        {"name": name, "shape": list(legacy_state_shapes[name])}
        for name in names
    ]
    return {"version": 1, "leaves": [{"param_groups": [entries]}]}


@dataclass(frozen=True)
class OptimizerRemapDiagnostics:
    restored_parameter_states: int
    reset_wholespace_states: tuple[str, ...]
    dropped_saved_states: tuple[str, ...]

    def message(self) -> str:
        return (
            "optimizer checkpoint remapped by parameter name: restored "
            f"{self.restored_parameter_states} state(s), reset "
            f"{len(self.reset_wholespace_states)} WholeSpace property "
            f"state(s), ignored {len(self.dropped_saved_states)} removed "
            "parameter state(s)"
        )


@dataclass(frozen=True)
class OptimizerRemapResult:
    state: Mapping[str, Any]
    diagnostics: OptimizerRemapDiagnostics


def remap_optimizer_state_by_name(
    saved_optimizer_state: Mapping[str, Any],
    saved_manifest: Mapping[str, Any],
    live_optimizer_state: Mapping[str, Any],
    live_manifest: Mapping[str, Any],
    *,
    reset_wholespace: bool = False,
) -> OptimizerRemapResult:
    """Return optimizer state aligned to the live named parameter layout.

    ``reset_wholespace`` is true only at the schema-1 -> schema-2 semantic
    boundary.  Future schema-2 resumes must leave it false: the small dynamic
    property basis is a legitimate trainable inventory and its named moments
    remain valid checkpoint state.
    """

    saved_leaves = _optimizer_state_leaves(saved_optimizer_state)
    live_leaves = _optimizer_state_leaves(live_optimizer_state)
    saved_mleaves = _manifest_leaves(saved_manifest)
    live_mleaves = _manifest_leaves(live_manifest)
    counts = {
        len(saved_leaves), len(live_leaves),
        len(saved_mleaves), len(live_mleaves),
    }
    if len(counts) != 1:
        raise ValueError("optimizer state/manifest leaf counts differ")

    remapped_leaves: list[dict[str, Any]] = []
    restored = 0
    reset: list[str] = []
    dropped: list[str] = []

    for saved_leaf, saved_ml, live_leaf, live_ml in zip(
        saved_leaves, saved_mleaves, live_leaves, live_mleaves
    ):
        saved_groups = saved_leaf.get("param_groups")
        live_groups = live_leaf.get("param_groups")
        saved_mgroups = saved_ml.get("param_groups")
        live_mgroups = live_ml.get("param_groups")
        if not (
            isinstance(saved_groups, list)
            and isinstance(live_groups, list)
            and isinstance(saved_mgroups, list)
            and isinstance(live_mgroups, list)
            and len(saved_groups) == len(saved_mgroups)
            and len(live_groups) == len(live_mgroups)
            and len(saved_groups) == len(live_groups)
        ):
            raise ValueError("optimizer parameter-group layout differs")

        saved_by_name: dict[str, tuple[Any, Any, tuple[int, ...] | None]] = {}
        for group, entries in zip(saved_groups, saved_mgroups):
            ids = list(group.get("params", ()))
            if len(ids) != len(entries):
                raise ValueError("saved optimizer manifest group length differs")
            for parameter_id, entry in zip(ids, entries):
                name = _entry_name(entry)
                if name in saved_by_name:
                    raise ValueError(f"duplicate saved optimizer name {name!r}")
                saved_by_name[name] = (
                    parameter_id,
                    (saved_leaf.get("state") or {}).get(parameter_id),
                    _entry_shape(entry),
                )

        new_state: dict[Any, Any] = {}
        new_groups: list[dict[str, Any]] = []
        live_names: set[str] = set()
        for saved_group, live_group, entries in zip(
            saved_groups, live_groups, live_mgroups
        ):
            live_ids = list(live_group.get("params", ()))
            if len(live_ids) != len(entries):
                raise ValueError("live optimizer manifest group length differs")
            # Retain saved hyperparameters (LR, betas, etc.) but replace the
            # positional ids with the current optimizer's ids.
            new_group = {
                key: value for key, value in saved_group.items()
                if key != "params"
            }
            new_group["params"] = live_ids
            new_groups.append(new_group)
            for live_id, entry in zip(live_ids, entries):
                name = _entry_name(entry)
                live_names.add(name)
                if (reset_wholespace
                        and is_reset_wholespace_parameter(name)):
                    saved_reset = saved_by_name.get(name)
                    if (saved_reset is not None
                            and isinstance(saved_reset[1], Mapping)):
                        reset.append(name)
                    continue
                saved = saved_by_name.get(name)
                if saved is None or not isinstance(saved[1], Mapping):
                    continue
                saved_shape = saved[2]
                live_shape = _entry_shape(entry)
                if saved_shape is not None and live_shape is not None:
                    same = saved_shape == live_shape
                    first_axis_growth = (
                        len(saved_shape) > 0
                        and len(saved_shape) == len(live_shape)
                        and saved_shape[1:] == live_shape[1:]
                        and saved_shape[0] <= live_shape[0]
                    )
                    if not (same or first_axis_growth):
                        raise ValueError(
                            f"optimizer parameter {name!r} cannot migrate "
                            f"from shape {saved_shape} to {live_shape}"
                        )
                new_state[live_id] = saved[1]
                restored += 1

        for name, (_pid, param_state, _shape_hint) in saved_by_name.items():
            if not isinstance(param_state, Mapping) or name in live_names:
                continue
            if (reset_wholespace
                    and is_reset_wholespace_parameter(name)):
                reset.append(name)
            else:
                dropped.append(name)
        remapped_leaves.append({"state": new_state, "param_groups": new_groups})

    if "optimizers" in live_optimizer_state:
        # MultiOptimizer is a flat list in this project.  Reject a nested
        # shape here rather than silently rebuilding the wrong tree.
        if len(remapped_leaves) != len(live_optimizer_state["optimizers"]):
            raise ValueError("nested optimizer topology is unsupported")
        remapped: Mapping[str, Any] = {"optimizers": remapped_leaves}
    else:
        if len(remapped_leaves) != 1:
            raise ValueError("single optimizer expected one remapped leaf")
        remapped = remapped_leaves[0]

    return OptimizerRemapResult(
        state=remapped,
        diagnostics=OptimizerRemapDiagnostics(
            restored_parameter_states=restored,
            reset_wholespace_states=tuple(sorted(set(reset))),
            dropped_saved_states=tuple(sorted(set(dropped))),
        ),
    )
