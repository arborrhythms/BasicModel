"""Reviewer probes for integrated spec §10.3 and the §11 expectation rollout."""
import copy
from pathlib import Path
import re
import pytest
import torch
from test_sentence_expectation import layer, admit










@pytest.mark.parametrize("setting,enabled", [(None, True), ("true", True), ("false", False)])
def test_omitted_enabled_and_disabled_model_settings(tmp_path, setting, enabled):
    from test_meronomy_ladder import _build_ladder_variant
    source = (Path(__file__).resolve().parents[1] / "data/MM_ladder.xml").read_text()
    replacement = "" if setting is None else f"<sentenceExpectation>{setting}</sentenceExpectation>"
    model = _build_ladder_variant(tmp_path, "expectation_default", [
        (re.search(r"<sentenceExpectation>.*?</sentenceExpectation>", source).group(), replacement),
        (re.search(r"<symbolFieldLossWeight>.*?</symbolFieldLossWeight>", source).group(), ""),
    ])
    try:
        assert model.symbolSpace.expectation is not None  # word level stays live
        assert model.symbolSpace.expectation.expectation_enabled == enabled
        assert ('field' in model.symbolSpace.expectation.enabled_levels) == enabled
        assert model.symbol_field_loss_weight == 0.1
        assert not hasattr(model, 'arma_scale')
        assert not hasattr(model, 'inter_contrastive_weight')
        if enabled:
            assert not hasattr(model.symbolSpace.expectation, "expectation_scope")
    finally:
        model.End()
        torch._dynamo.reset()


def test_canonical_configuration_enables_only_the_selected_prediction_objective():
    import xml.etree.ElementTree as ET
    config = ET.parse(Path(__file__).resolve().parents[1] / "data/BasicModel.xml")
    training = config.getroot().find(".//architecture/training")
    assert training.findtext("sentenceExpectation") == "true"
    assert float(training.findtext("symbolFieldLossWeight")) == .1
    assert training.find("sentenceExpectationArmaScale") is None
    assert training.find("sentenceExpectationContrastiveWeight") is None


def test_native_runtime_reports_pairs_without_accumulating_or_updating(tmp_path, monkeypatch, eager_reading):
    from test_meronomy_ladder import _build_ladder_variant
    monkeypatch.setenv("MODEL_COMPILE", "none")
    model = _build_ladder_variant(tmp_path, "runtime_expectation", [
        ("<serialWordCapacity>8</serialWordCapacity>", "<serialWordCapacity>16</serialWordCapacity>"),
        ("<serialWordBuckets>8</serialWordBuckets>", "<serialWordBuckets>16</serialWordBuckets>"),
        ("<sentenceExpectation>false</sentenceExpectation>", "<sentenceExpectation>true</sentenceExpectation>"),
        ("<symbolFieldLossWeight>0.0</symbolFieldLossWeight>", "<symbolFieldLossWeight>0.1</symbolFieldLossWeight>"),
    ])
    model._tensor_peer_while_eager = True
    model.inputSpace.data.has_supervised_outputs = False  # Input/storage probe has no answer target.
    model._chart_compose_per_word = lambda: None
    model._install_unit_span_fn()
    from reading_fixtures import read_in_source_order
    read_in_source_order(model, monkeypatch)
    model.train()  # the runtime call must apply its own declared training gate
    discourse = model.symbolSpace.expectation
    before = [p.detach().clone() for p in discourse.parameters()]
    try:
        inputs = model.inputSpace.prepPackedInput([["1 plus 2", "3 plus 4"]])
        with torch.enable_grad():
            result, _ = model.runBatch(
                train=False, batchSize=1, split="runtime",
                batch_override=(inputs, torch.zeros(1, 1, 0)))
        assert result is not None
        assert len(discourse.situation_references()) == 2
        assert not hasattr(discourse, '_inter_last_meaning')
        from ReferenceContext import SituationAddress
        assert all(isinstance(value, SituationAddress) for chain in discourse._inter_context for value in chain)
        assert all(not frame.roles.requires_grad for b in range(discourse._batch)
                   for frame in discourse.situation_references(b))
        for parameter, value in zip(discourse.parameters(), before):
            torch.testing.assert_close(parameter, value)
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()




@pytest.mark.usefixtures('eager_reading')
@pytest.mark.slow
def test_enable_after_disabled_construction_joins_optimizer_and_off_keeps_input_learning(tmp_path, monkeypatch):
    from test_meronomy_ladder import _build_ladder_variant
    monkeypatch.setenv("MODEL_COMPILE", "none")
    model = _build_ladder_variant(tmp_path, "toggle_expectation", [
        ("<serialWordCapacity>8</serialWordCapacity>", "<serialWordCapacity>16</serialWordCapacity>"),
        ("<serialWordBuckets>8</serialWordBuckets>", "<serialWordBuckets>16</serialWordBuckets>"),
        ("<symbolFieldLossWeight>0.0</symbolFieldLossWeight>", "<symbolFieldLossWeight>0.1</symbolFieldLossWeight>"),
        ("<training>", "<training><wordExpectation>false</wordExpectation><wordExpectationLossWeight>0</wordExpectationLossWeight>"),
    ])
    model._tensor_peer_while_eager = True
    model._chart_compose_per_word = lambda: None
    model._install_unit_span_fn()
    monkeypatch.setattr(model.inputSpace.data, "has_supervised_outputs", False)
    model.eval()
    optimizer = model.getOptimizer(lr=1e-5)
    owner = model._what_memory()
    assert model.symbolSpace.expectation is None
    try:
        for enabled in (True, False, True):
            model.begin_presentation()
            model.set_sentence_expectation(enabled)
            discourse = model.symbolSpace.expectation
            assert 'word' not in discourse.enabled_levels
            parameters = list(discourse._inter_predictor.parameters())
            before = [p.detach().clone() for p in parameters]
            count = len(discourse.situation_references())
            inputs = model.inputSpace.prepPackedInput([["1 plus 2", "3 plus 4"]])
            result, _ = model.runBatch(train=True, batchSize=1, optimizer=optimizer,
                                      batch_override=(inputs, torch.zeros(1, 1, 0)))
            assert model._what_memory() is owner
            assert result.lossIn is not None
            changes = [not torch.equal(p, old) for p, old in zip(parameters, before)]
            assert any(changes) if enabled else not any(changes)
            # Each presentation starts with a cover, then both sentence targets.
            assert len(discourse.situation_references()) - count == 2
            owned = {id(p) for group in optimizer.param_groups for p in group["params"]}
            assert all(id(p) in owned for p in parameters)
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()


@pytest.mark.usefixtures('eager_reading')
def test_expectation_off_keeps_every_packed_closing_in_ltm(tmp_path, monkeypatch):
    import Models
    from test_meronomy_ladder import _build_ladder_variant
    monkeypatch.setenv("MODEL_COMPILE", "none")
    model = _build_ladder_variant(tmp_path, "off_packed_memory", [
        ("<architecture>", "<architecture><ltmConsolidation>true</ltmConsolidation>"),
        ("<serialWordCapacity>8</serialWordCapacity>", "<serialWordCapacity>16</serialWordCapacity>"),
        ("<serialWordBuckets>8</serialWordBuckets>", "<serialWordBuckets>16</serialWordBuckets>"),
    ])
    model._tensor_peer_while_eager = True
    model.inputSpace.data.has_supervised_outputs = False  # Input/storage probe has no answer target.
    model._chart_compose_per_word = lambda: None
    model._install_unit_span_fn()
    # Since the loop of looks which words each packed sentence reads is the
    # chooser's draw; both sentences' words are read, in source order.
    from reading_fixtures import read_in_source_order
    read_in_source_order(model, monkeypatch)
    assert not model.symbolSpace.expectation.expectation_enabled
    store = model.symbolSpace.ltm_store
    written, sources = [], []
    append = Models._append_observed_meaning

    def capture(target, clause, **kwargs):
        written_rows = kwargs.setdefault('written_rows', {})
        row = append(target, clause, **kwargs)
        if target is store:
            written.append(written_rows['root_occurrence'])
            from Occurrence import source_at
            sources.append(source_at(model,kwargs['stream'],len(written)-1))
        return row

    monkeypatch.setattr(Models, '_append_observed_meaning', capture)
    try:
        before = len(store)
        inputs = model.inputSpace.prepPackedInput([["1 plus 2", "3 plus 4"]])
        model.runBatch(train=False, batchSize=1, split="runtime",
                       batch_override=(inputs, torch.zeros(1, 1, 0)))
        rows = [store.row(i) for i in range(before, len(store))]
        # An untrained reading can leave a bind unresolved and close as a
        # question. Both external closings must survive, independently of
        # their evidence and of the expectation switch. Tagged chain roots
        # and occurrence roots share this reader; sentence time is perceptual.
        assert len(written) == len(set(written)) == 2
        roots = [store.location(address) for address in written]
        assert all(row is not None for row in roots)
        closings = [store.row(row) for row in roots]
        assert len(set(sources)) == 2  # Both packed source positions reached the writer.
        rows.extend(row for address, row in zip(roots, closings) if isinstance(address, tuple))
        assert all(int(store.meaning_of(i).role_mask.sum()) == 1 for i in range(before, len(store)))
        assert not hasattr(store, 'clause_derivation')
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()


def test_packed_ltm_ignores_masked_slots_even_with_retained_storage(monkeypatch):
    from types import SimpleNamespace
    from Layers import SymbolExpectation
    from Models import BasicModel
    from test_clause_acceptance import SentenceFixture
    f = SentenceFixture(monkeypatch)
    entry = f.program('cat')
    discourse = SymbolExpectation(n_symbols=8, max_depth=8, n_dim=8,
        concept_dim=8)
    discourse.set_symbol_bank(torch.eye(discourse.concept_dim))
    from reading_fixtures import commit_reading
    store = f.store
    discourse._ltm_store = store
    for sid, active in enumerate((True, False)):
        commit_reading(f.language, f.registry, entry, store, discourse=discourse,
                       sid=sid, active=active, document='a')
    assert len(store) == 1
    assert len(discourse.situation_references()) == 1


def test_enabling_expectation_preserves_the_callers_random_stream(tmp_path):
    from test_meronomy_ladder import _build_ladder_variant
    model = _build_ladder_variant(tmp_path, "expectation_rng", [])
    assert not model.symbolSpace.expectation.expectation_enabled
    before = torch.get_rng_state().clone()
    try:
        model.set_sentence_expectation(True)
        assert torch.equal(torch.get_rng_state(), before)
    finally:
        model.End()
        torch._dynamo.reset()
