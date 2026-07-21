"""Regression contract for the WholeSpace property-inventory migration.

WholeSpace is perceptual and upstream.  Its capacity is therefore independent
of the ConceptualSpace dictionary, and legacy WS concept/META tensors must not
be restored into the new property basis.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
import sys
import warnings
import xml.etree.ElementTree as ET

import torch

os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "bin"
if str(BIN) not in sys.path:
    sys.path.insert(0, str(BIN))

import Language  # noqa: E402
import Models  # noqa: E402
from Spaces import SubSpace  # noqa: E402
from util import init_config  # noqa: E402


def _set_text(root: ET.Element, path: str, value: object) -> None:
    parent_path, _, tag = path.rpartition("/")
    parent = root.find(parent_path) if parent_path else root
    assert parent is not None, path
    node = parent.find(tag)
    if node is None:
        node = ET.SubElement(parent, tag)
    node.text = str(value).lower() if isinstance(value, bool) else str(value)


def _small_property_model(
        tmp_path: Path, *, native_event: int | None = None,
        concept_event: int | None = None, symbol_tower: bool = False,
        symbolic_lift: bool = False):
    """Build aligned live-width-8 towers with CS=16 and WS=8 rows."""
    tree = ET.parse(ROOT / "data" / "MM_xor_fixture.xml")
    root = tree.getroot()
    _set_text(root, "architecture/serial", True)
    _set_text(root, "architecture/conceptBinding", "aligned")
    _set_text(root, "architecture/training/autoload", False)
    _set_text(root, "ConceptualSpace/nVectors", 16)
    _set_text(root, "WholeSpace/nVectors", 8)
    _set_text(root, "WholeSpace/propertyBasis", True)
    if native_event is not None or concept_event is not None:
        assert native_event is not None and concept_event is not None
        _set_text(root, "architecture/serialObjectMeta", True)
        _set_text(root, "PartSpace/nDim", native_event)
        _set_text(root, "PartSpace/nOutputDim", native_event)
        _set_text(root, "ConceptualSpace/nInputDim", concept_event)
        _set_text(root, "ConceptualSpace/nDim", concept_event)
        _set_text(root, "ConceptualSpace/nOutputDim", concept_event)
        _set_text(root, "WholeSpace/nInputDim", concept_event)
        _set_text(root, "WholeSpace/nDim", native_event)
        _set_text(root, "WholeSpace/nOutputDim", native_event)
        _set_text(root, "OutputSpace/nInputDim", concept_event)
    if symbol_tower:
        _set_text(root, "architecture/symbolTower", True)
    if symbolic_lift:
        grammar = root.find("SymbolSpace/language/grammar")
        assert grammar is not None
        ET.SubElement(grammar, "S").text = "S = lift(S, S)"
    for section in ("ConceptualSpace", "WholeSpace"):
        active = root.find(f"{section}/activeVectors")
        if active is not None:
            root.find(section).remove(active)

    path = tmp_path / "unequal_property_inventory.xml"
    tree.write(path, encoding="utf-8", xml_declaration=True)
    init_config(path=str(path), defaults_path=str(ROOT / "data" / "model.xml"))
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, _ = Models.BasicModel.from_config(str(path))
    return model


def test_basicmodel_config_separates_concepts_properties_and_live_width():
    root = ET.parse(ROOT / "data" / "BasicModel.xml").getroot()
    cs_rows = int(root.findtext("ConceptualSpace/nVectors"))
    ws_rows = int(root.findtext("WholeSpace/nVectors"))

    assert root.findtext("WholeSpace/propertyBasis") == "true"
    assert cs_rows == 1048576
    assert int(root.findtext("ConceptualSpace/activeVectors")) == 32768
    assert int(root.findtext("PartSpace/nVectors")) == 32768
    assert int(root.findtext("PartSpace/maxVectors")) == 1048576
    assert ws_rows == 8
    assert root.find("WholeSpace/activeVectors") is None
    assert int(root.findtext("ConceptualSpace/nOutput")) == 8
    assert int(root.findtext("WholeSpace/nOutput")) == 8
    assert cs_rows != ws_rows  # alignment is over nOutput locations, not rows


def test_aligned_model_accepts_unequal_inventory_capacities(tmp_path):
    model = _small_property_model(tmp_path)

    assert model.concept_binding == "aligned"
    assert model.nConceptCodes == 16
    assert model.nSymbols == model.nConceptCodes  # downstream concept refs
    assert model.nWholeProperties == 8
    assert model.nSymbolSlots == 8
    assert model.wholeSpace.subspace.what.nVectors == 8


def test_aligned_property_folds_share_one_physical_concept_dictionary(tmp_path):
    model = _small_property_model(tmp_path)
    codebooks = [cs.similarity_codebook for cs in model.conceptualSpaces]

    assert len(codebooks) == 3
    assert all(cb is codebooks[0] for cb in codebooks)
    assert all(cs.subspace._index_basis is codebooks[0]
               for cs in model.conceptualSpaces)
    assert model._aligned_capacity_codebooks() == [codebooks[0]]

    # Logical growth mutates the one shared mask without reallocating W.
    cb = codebooks[0]
    cb.vq.set_active_rows(4)
    model._active_inventory_rows = 4
    identity = (id(cb.W), cb.W.data_ptr(), cb.vq.active_mask.data_ptr())
    assert model._ensure_aligned_active_rows(5) == 8
    assert model._active_prefix_rows(cb.vq) == 8
    assert identity == (id(cb.W), cb.W.data_ptr(),
                        cb.vq.active_mask.data_ptr())

    optimizer = model.getOptimizer(lr=1e-3)
    assert sum(
        parameter is cb.W
        for group in optimizer.param_groups
        for parameter in group["params"]
    ) == 1


def test_wholespace_owns_one_property_codebook_and_no_concept_store(tmp_path):
    model = _small_property_model(tmp_path)
    ws = model.wholeSpace
    state = ws.state_dict()

    assert ws.subspace.what.nVectors == model.nWholeProperties
    assert getattr(ws, "analysis_store", None) is None
    assert getattr(ws, "type_subspace", None) is None
    for legacy_structure in (
        "taxonomy", "taxonomy_parent_map", "meta_pair_to_idx",
        "meta_trust", "meta_fold_support", "part_chain",
        "_next_position", "_pos_kind", "_ps_pos_to_row",
        "_ps_row_to_pos", "_ws_pos_to_row", "_ws_row_to_pos",
        "_word_whole_ss", "_property_class_whole",
        "_lbg_disp_sum", "_lbg_disp_sum_sq", "_lbg_count",
        "_lbg_threshold", "_lbg_min_count", "_lbg_epsilon",
    ):
        assert not hasattr(ws, legacy_structure), legacy_structure
    for downstream_module in (
        "rule_codebook", "languageLayer", "syntacticLayer",
        "propositional_negation", "_sparsity", "_smoothing",
        "_impenetrable"):
        assert not hasattr(ws, downstream_module), downstream_module
    for downstream_state in (
        "conceptualSpace", "terminalConceptualSpace_ref", "symbolSpace",
        "semantic_arrangement_weight", "l1_lambda",
        "discontinuity_lambda", "symbol_residual_scale",
        "output_symbol_residual_scale", "commitment_beta", "use_vqvae",
        "gradient_mode", "decorrelation_weight",
        "spectral_flatness_weight", "impenetrable_overlap",
        "impenetrable_variance",
    ):
        assert not hasattr(ws, downstream_state), downstream_state
    assert model.conceptualSpace.terminalConceptualSpace_ref is (
        model.conceptualSpace)
    assert all("analysis_store" not in key for key in state)
    assert all("type_subspace" not in key for key in state)
    assert all(not key.startswith("conceptualSpace.") for key in state)

    # The sole WS inventory is the canonical property basis. In particular,
    # there must be no concept-sized WS W/EMA/mask tensor.
    property_rows = []
    for key, value in state.items():
        if "_owned_bases.what" in key and value.ndim > 0:
            property_rows.append((key, int(value.shape[0])))
    assert property_rows
    assert all(rows == 8 for _key, rows in property_rows)
    assert not any(rows == model.nConceptCodes for _key, rows in property_rows)


def test_property_model_stm_grammar_resolves_from_symbolspace(tmp_path):
    """Bounded STM operators stay downstream of the property analyzer."""
    model = _small_property_model(tmp_path)
    ws = model.wholeSpace
    symbols = model.symbolSpace

    assert not hasattr(ws, "syntacticLayer")
    assert not hasattr(ws, "languageLayer")
    assert hasattr(symbols, "syntacticLayer")
    assert hasattr(symbols, "languageLayer")

    reducer = model._stm_reducer()
    assert reducer is not None
    assert reducer is model._stm_reducer()  # cached, not re-minted

    expected_unary = symbols.languageLayer._unary_layers["CS"]
    assert expected_unary is not None
    assert model._stm_unary_rewriter() is expected_unary


def test_symbolspace_and_grammar_use_conceptual_not_property_width(tmp_path):
    """SS state is CS-width even when the upstream property basis is narrow."""
    model = _small_property_model(
        tmp_path, native_event=16, concept_event=24,
        symbol_tower=True, symbolic_lift=True)
    cs = model.conceptualSpace
    ws = model.wholeSpace
    ss = model.symbolSpace

    assert cs.subspace.nWhat == 16
    assert cs.subspace.muxedSize == 24
    assert ws.subspace.nWhat == 8
    assert ws.subspace.muxedSize == 16

    assert ss.nWhat == cs.subspace.nWhat
    assert ss.muxedSize == cs.subspace.muxedSize
    assert ss.languageLayer.feature_dim == cs.subspace.nWhat
    assert ss.truth_layer.nDim == cs.subspace.nWhat
    assert ss.relative_store.nDim == cs.subspace.nWhat
    assert ss._stm_payload_dim == cs.subspace.muxedSize
    assert ss.what.nDim == cs.subspace.nWhat

    lift = ss.syntacticLayer._by_name["lift"]
    assert lift.nInput == cs.subspace.nWhat
    assert lift.nOutput == cs.subspace.nWhat
    assert lift.nInput != ws.subspace.nWhat

    # Sparse SS activation scales only the conceptual WHAT row.  The event's
    # where/when band remains metadata and must survive the activation seam.
    event = torch.randn(1, 8, cs.subspace.muxedSize)
    concept_sub = SubSpace(
        inputShape=(8, cs.subspace.muxedSize),
        outputShape=(8, cs.subspace.muxedSize),
        nInputDim=cs.subspace.muxedSize,
        nOutputDim=cs.subspace.muxedSize)
    concept_sub.set_event(event)
    concept_sub._concept_activations = torch.ones(8, 1)
    symbol_leg = ss.forward_concept_to_symbol(concept_sub).materialize()
    assert tuple(symbol_leg.shape) == tuple(event.shape)
    assert torch.equal(symbol_leg[..., cs.subspace.nWhat:],
                       event[..., cs.subspace.nWhat:])


def test_property_model_category_vq_and_parser_context_are_cs_owned(tmp_path):
    """Fresh property models allocate/use categories downstream in CS.

    The first autobind opportunity must enable the requested role VQ before
    taking the property-basis early return.  Parser reads and round-0 role
    observations then stay on ConceptualSpace; WholeSpace retains no category
    module, category sidecar, taxonomy, or downstream reference.
    """
    model = _small_property_model(tmp_path)
    cs = model.conceptualSpace
    ws = model.wholeSpace
    router = model.symbolSpace.languageLayer

    assert getattr(cs, "_category_codebook_requested", False) is True
    assert not cs.category_codebook_enabled()
    assert not ws.category_codebook_enabled()

    # The grammar is configured by SymbolSpace construction.  Even an empty
    # first autobind call reaches lazy category allocation before validating
    # the (not-yet-present) percept slab.
    cs._maybe_autobind_meta(None, None)
    assert cs.category_codebook_enabled()
    assert not ws.category_codebook_enabled()
    assert router._category_owner(model.symbolSpace) is cs
    assert not any("category" in key for key in ws.state_dict())

    # A property-mode terminal resolves pid -> word-concept directly in CS,
    # rather than trying the retired WS pid -> taxonomy-META lookup.
    word_concept, _object, _meta = cs.create_word_object_meta(
        [7], [0], key="cat")
    cs._category_last_pid = [[7, -1]]
    cs._category_assign[word_concept] = 0
    cs._category_role[0].zero_()
    cs._category_role[0, 2] = 1.0
    x = torch.zeros(1, 2, int(router.feature_dim))
    ctx = router._build_category_context(x, cs)
    assert ctx is not None
    assert tuple(ctx.shape) == (1, 2, int(cs._category_n_roles))
    assert float(ctx[0, 0, 2]) == 1.0
    assert torch.count_nonzero(ctx[0, 1]) == 0

    # The parser's observation handoff is likewise stashed on CS, never WS.
    router.compose(x, model.symbolSpace)
    assert hasattr(cs, "_category_role_obs")
    assert not hasattr(ws, "_category_role_obs")


def test_property_sidecar_cannot_recreate_lbg_or_downstream_state(tmp_path):
    model = _small_property_model(tmp_path)
    ws = model.wholeSpace

    # Property-mode LBG compatibility methods are explicit no-ops and do not
    # lazily recreate the retired concept/META splitting inventory.
    sample = ws.subspace.what.getW()[0].detach().clone()
    assert ws.record_lbg_pull(1, sample) is None
    assert ws.maybe_split_lbg(1) is None
    assert not hasattr(ws, "_lbg_count")

    snapshot = model._collect_structural_extras()
    assert snapshot["version"] == 2
    for entry in snapshot.get("whole_properties", {}).values():
        attrs = entry.get("attributes", {})
        assert all(not str(name).startswith("_lbg_") for name in attrs)
        vocab = entry.get("vocab_extras", {})
        assert all("lbg" not in str(name).lower() for name in vocab)

    # Ignore an early schema-2 sidecar that accidentally carried the legacy
    # accumulators; loading it must not materialize those attributes on WS.
    model._restore_structural_extras({
        "version": 2,
        "conceptual_spaces": {},
        "whole_properties": {
            "0": {
                "attributes": {
                    "_lbg_disp_sum": {1: sample},
                    "_lbg_disp_sum_sq": {1: sample.square()},
                    "_lbg_count": {1: 9},
                    "semantic_arrangement_weight": 3.0,
                },
            },
        },
    })
    for name in (
        "_lbg_disp_sum", "_lbg_disp_sum_sq", "_lbg_count",
        "semantic_arrangement_weight",
    ):
        assert not hasattr(ws, name), name


def test_legacy_ws_concept_state_is_quarantined_not_loaded_as_properties():
    from checkpoint_migrations import migrate_wholespace_checkpoint

    live_state = {
        "conceptualSpaces.0.similarity_codebook.W": torch.zeros(16, 4),
        "wholeSpaces.0._owned_bases.what.W": torch.zeros(8, 4),
    }
    legacy_cs = torch.arange(64, dtype=torch.float32).reshape(16, 4)
    legacy_registered_cs_alias = legacy_cs + 500
    legacy_ws = legacy_cs + 1000
    legacy_analysis = legacy_cs + 2000
    checkpoint = {
        "state_dict": {
            "conceptualSpaces.0.similarity_codebook.W": legacy_cs,
            # Historical module registration placed a CS alias below WS.
            # It is concept state, not a WS dense inventory, so the migration
            # must preserve it for the loader's later alias canonicalization.
            "wholeSpaces.0.conceptualSpace.similarity_codebook.W": (
                legacy_registered_cs_alias),
            "wholeSpaces.0._owned_bases.what.W": legacy_ws,
            "wholeSpaces.0.analysis_store.W": legacy_analysis,
        },
        "vocab_extras": {
            "index_to_key": [],
            "well_known_atoms": {"everything": 7},
            "ws_taxonomy_extras": {"taxonomy": {9: [2, 7]}},
        },
        "structural_extras": {
            "version": 1,
            "conceptual_spaces": {"0": {"attributes": {"_words_concept_id": 3}}},
            "whole_spaces": {"0": {"attributes": {"_word_whole_ss": {"cat": 7}}}},
        },
    }
    before = copy.deepcopy(checkpoint)

    result = migrate_wholespace_checkpoint(checkpoint, live_state)

    assert result.migrated is True
    assert result.checkpoint["checkpoint_schema"] == {
        "version": 2,
        "whole_space_role": "property_basis",
    }
    migrated_state = result.checkpoint["state_dict"]
    assert "wholeSpaces.0._owned_bases.what.W" not in migrated_state
    assert "wholeSpaces.0.analysis_store.W" not in migrated_state
    torch.testing.assert_close(
        migrated_state["conceptualSpaces.0.similarity_codebook.W"], legacy_cs)
    torch.testing.assert_close(
        migrated_state[
            "wholeSpaces.0.conceptualSpace.similarity_codebook.W"],
        legacy_registered_cs_alias,
    )

    quarantine = result.checkpoint["legacy_whole_structure"]
    assert quarantine["version"] == 1
    assert quarantine["vocab_extras"]["well_known_atoms"] == {"everything": 7}
    assert quarantine["vocab_extras"]["ws_taxonomy_extras"] == {
        "taxonomy": {9: [2, 7]}}
    assert quarantine["structural_whole_spaces"] == {
        "0": {"attributes": {"_word_whole_ss": {"cat": 7}}}}

    # Migration is functional: callers may retain or retry the source bundle.
    assert checkpoint.keys() == before.keys()
    assert checkpoint["vocab_extras"] == before["vocab_extras"]
    torch.testing.assert_close(
        checkpoint["state_dict"]["wholeSpaces.0._owned_bases.what.W"],
        before["state_dict"]["wholeSpaces.0._owned_bases.what.W"],
    )

    again = migrate_wholespace_checkpoint(result.checkpoint, live_state)
    assert again.migrated is False
    assert again.checkpoint["checkpoint_schema"]["version"] == 2


def test_optimizer_remap_keeps_cs_moments_and_resets_both_old_ws_tables():
    from checkpoint_migrations import remap_optimizer_state_by_name

    cs_name = "conceptualSpaces.0.similarity_codebook.W"
    ws_name = "wholeSpaces.0._owned_bases.what.W"
    analysis_name = "wholeSpaces.0.analysis_store.W"

    def manifest(entries):
        return {
            "version": 1,
            "leaves": [{
                "param_groups": [[
                    {"name": name, "shape": list(shape)}
                    for name, shape in entries
                ]],
            }],
        }

    saved_manifest = manifest([
        (cs_name, (16, 4)),
        (ws_name, (16, 4)),
        (analysis_name, (16, 4)),
    ])
    live_manifest = manifest([
        (cs_name, (16, 4)),
        (ws_name, (8, 4)),
    ])
    cs_moments = {
        "step": torch.tensor(11.0),
        "exp_avg": torch.ones(16, 4),
        "exp_avg_sq": torch.full((16, 4), 2.0),
    }
    saved_optimizer = {
        "state": {
            0: cs_moments,
            1: {
                "step": torch.tensor(11.0),
                "exp_avg": torch.full((16, 4), 3.0),
                "exp_avg_sq": torch.full((16, 4), 4.0),
            },
            2: {
                "step": torch.tensor(11.0),
                "exp_avg": torch.full((16, 4), 5.0),
                "exp_avg_sq": torch.full((16, 4), 6.0),
            },
        },
        "param_groups": [{"lr": 1e-3, "params": [0, 1, 2]}],
    }
    live_optimizer = {
        "state": {},
        "param_groups": [{"lr": 1e-3, "params": [10, 11]}],
    }

    result = remap_optimizer_state_by_name(
        saved_optimizer,
        saved_manifest,
        live_optimizer,
        live_manifest,
        reset_wholespace=True,
    )

    assert set(result.state["state"]) == {10}
    assert result.state["state"][10] is cs_moments
    assert result.state["param_groups"][0]["params"] == [10, 11]
    assert result.diagnostics.restored_parameter_states == 1
    assert set(result.diagnostics.reset_wholespace_states) == {
        ws_name, analysis_name}
    assert result.diagnostics.dropped_saved_states == ()

    # Once a schema-2 property basis has been trained, its moments are valid
    # property state and ordinary resume must preserve them.
    property_saved = {
        "state": {
            0: cs_moments,
            1: {
                "step": torch.tensor(12.0),
                "exp_avg": torch.full((8, 4), 7.0),
                "exp_avg_sq": torch.full((8, 4), 8.0),
            },
        },
        "param_groups": [{"lr": 1e-3, "params": [0, 1]}],
    }
    property_manifest = manifest([
        (cs_name, (16, 4)),
        (ws_name, (8, 4)),
    ])
    resumed = remap_optimizer_state_by_name(
        property_saved,
        property_manifest,
        live_optimizer,
        live_manifest,
    )
    assert set(resumed.state["state"]) == {10, 11}
    assert resumed.diagnostics.restored_parameter_states == 2
    assert resumed.diagnostics.reset_wholespace_states == ()
