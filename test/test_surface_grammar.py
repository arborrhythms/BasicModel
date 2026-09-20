"""A selected surface attachment keeps semantics and learns its own spelling.

No runtime word table assigns an operator here. The attachment's numerical
carrier retains the marker for the ordinary grammar chooser; its declared
semantic projection keeps the complete content subtree.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")

import torch
import pytest

import Language


def test_surface_attachment_keeps_marker_information_and_recomposes():
    layer = Language.SurfaceLayer(nInput=8, nOutput=8)
    left, right = torch.randn(3, 8), torch.randn(3, 8)
    parent = layer.compose(left, right)
    assert not torch.equal(parent, layer.compose(left + .5, right))
    a, b = layer.generate(parent)
    torch.testing.assert_close(layer.compose(a, b), parent)
    assert layer.semantic_operand == 1


def test_surface_generation_learns_distinct_children_from_full_width_values():
    torch.manual_seed(827)
    layer = Language.SurfaceLayer(nInput=8, nOutput=8)
    parent, left, right = torch.randn(3, 8), torch.randn(3, 8), torch.randn(3, 8)
    optimizer = torch.optim.Adam(layer.parameters(), lr=.03)
    for _ in range(400):
        optimizer.zero_grad()
        a, b = layer.generate(parent)
        loss = (a - left).square().mean() + (b - right).square().mean()
        loss.backward()
        optimizer.step()
    a, b = layer.generate(parent)
    assert float(loss.detach()) < .001
    torch.testing.assert_close(layer.compose(a, b), parent, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(a, b)


def test_surface_inverse_uses_only_supplied_occurrence_witness():
    layer = Language.SurfaceLayer(nInput=8, nOutput=8)
    left, right = torch.randn(3, 8), torch.randn(3, 8)
    parent = layer.compose(left, right)
    for side, reference in (("left", left), ("right", right)):
        a, b = layer.reconstruct(parent, reference, side)
        torch.testing.assert_close(a, left, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(b, right, atol=1e-5, rtol=1e-5)
    compiled = torch.compile(layer.generate, backend="eager", fullgraph=True)
    a, b = compiled(parent)
    torch.testing.assert_close(layer.compose(a, b), parent, atol=1e-5, rtol=1e-5)


def test_selected_surface_preserves_complete_content_meaning(monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    from test_selected_relation_meaning import _program_owner
    _cs, _grammar, registry, owner, _leaves, program, _a, _b = _program_owner(
        monkeypatch, interrogative=True)
    entry = program()
    original = owner.program_meaning(entry, registry)
    surface = len(owner._compose_binary_rules)
    object.__setattr__(owner, "_compose_binary_rules", owner._compose_binary_rules + (
        SimpleNamespace(method_name="surface"),))
    actions = torch.cat((torch.tensor([[0, -1, 2]]), entry.actions,
                         torch.tensor([[1, surface, -1]])))
    attached = replace(entry, rows=torch.cat((entry.rows, entry.rows[:1])),
                       word_rows=torch.cat((entry.word_rows, entry.word_rows[:1])),
                       activations=torch.cat((entry.activations, entry.activations[:1])),
                       concept_ids=torch.cat((entry.concept_ids, entry.concept_ids[:1])),
                       leaves=torch.cat((entry.leaves, torch.randn_like(entry.leaves[:1]))),
                       lexical_forms=(None,) * 3, actions=actions)
    recovered = owner.program_meaning(attached, registry)
    assert recovered.metadata() == original.metadata()
    torch.testing.assert_close(recovered.roles, original.roles)


def test_generate_stack_preserves_infix_order_for_semantic_answers():
    from types import SimpleNamespace
    from Models import BasicModel
    model = SimpleNamespace(conceptualSpace=SimpleNamespace(stm=SimpleNamespace(capacity=5)))
    idea = torch.arange(1, 25).reshape(2, 3, 4).float()
    operand = BasicModel._walk_operand(model, idea, infix_rows=(True, False))
    # The walk emits top first into a right-to-left buffer.
    torch.testing.assert_close(operand[0, :3], idea[0])
    torch.testing.assert_close(operand[1, :3], idea[1].flip(0))


@pytest.mark.slow
def test_real_text_has_a_complete_selected_meaning(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from test_compiled_word_chunk import _tiny_canonical_model
    from test_output_walk import _capture_program_probe
    from GrammarLessons import compose_loss, generate_loss
    torch.manual_seed(931)
    curriculum = json.loads((Path(__file__).resolve().parents[1] / "data/grammar_wording.json").read_text())
    training = curriculum["train"]
    model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets="8",
                                  batch_size=64, concept_rows=2048, dimension=64,
                                  chooser_depth=2)
    model._tensor_peer_while_eager = True
    model._chart_compose_per_word = lambda: None
    model.reconstruct_in_loop = False
    model.eval()
    def capture(texts):
        # Mirror runBatch's clearing of the previous completed brick.
        model._tensor_final_end_slots = None
        model._tensor_sentence_roots_live = None
        return _capture_program_probe(model, texts)
    programs = []
    with torch.no_grad():
        for lo in range(0, len(training), 64):
            u = capture([row["text"] for row in training[lo:lo+64]])
            programs.extend(u.answer_program)
    optimizer = model.getOptimizer(lr=.003)
    for step in range(1000):
        indices = torch.randperm(len(training))[:8].tolist()
        optimizer.zero_grad(set_to_none=True)
        loss = generate_loss(model.languageSpace, [programs[i] for i in indices],
                             [training[i] for i in indices], model.grammatical_thoughts,
                             model._grammar_target_word)
        loss.backward()
        optimizer.step()
        if step % 1000 == 0:
            print("GENERATE", step, float(loss.detach()), flush=True)
    for step in range(8000):
        indices = torch.randperm(len(training))[:64].tolist()
        optimizer.zero_grad(set_to_none=True)
        loss = compose_loss(model.languageSpace, [programs[i] for i in indices],
                            [training[i] for i in indices])
        loss.backward()
        optimizer.step()
        if step % 1000 == 0:
            print("TRAIN", step, float(loss.detach()), flush=True)
    evaluation = curriculum["validation"] + curriculum["test"]
    train_wordings = {row.get("relation_wording") for row in training}
    assert all(row["relation_wording"] not in train_wordings
               for row in evaluation if row.get("holdout") == "final")
    with torch.no_grad():
        observed = capture([row["text"] for row in evaluation])
    mistakes = []
    for row, program in zip(evaluation, observed.answer_program):
        meaning = model.languageSpace.program_meaning(program, model.grammatical_thoughts)
        if row["form"] == "lift":
            if meaning is not None:
                mistakes.append((row["text"], "invented relation"))
            continue
        refs = tuple(("sym", int(program.concept_ids[i])) for i in row["operands"])
        expected = model.grammatical_thoughts.form(row["form"], *refs, mode="assertive")
        if meaning is None or meaning.role_refs != expected.role_refs:
            names = [model.languageSpace._compose_binary_rules[int(action[1])].method_name
                     for action in program.actions if int(action[0]) == 1]
            mistakes.append((row["text"], names, None if meaning is None else meaning.role_refs, expected.role_refs))
    print("HELDOUT", len(evaluation), "MISTAKES", mistakes, flush=True)
    from What import What
    with torch.no_grad():
        construction = model.reverseOutput(observed, model.resolveAnswer(
            observed, tuple(What.present(i) for i in range(len(evaluation)))))
    wrong_output = [(row["text"], text, row["generation"]["text"])
                    for row, text in zip(evaluation, construction.texts)
                    if "generation" in row and text != row["generation"]["text"]]
    print("GENERATED", construction.texts, "MISTAKES", wrong_output, flush=True)
    assert any(step["operation"] == "generate:lexical_inverse" for step in construction.trace)
    assert len(model._concept_owner()._row_surfaces) >= 178
    assert not mistakes, mistakes
    assert not wrong_output, wrong_output
    generated_rows = [i for i, row in enumerate(evaluation) if "generation" in row]
    with torch.no_grad():
        reread = capture([construction.texts[i] for i in generated_rows])
    for row_index, program in zip(generated_rows, reread.answer_program):
        prior = model.languageSpace.program_meaning(observed.answer_program[row_index], model.grammatical_thoughts)
        meaning = model.languageSpace.program_meaning(program, model.grammatical_thoughts)
        assert meaning is not None
        assert meaning.metadata() == prior.metadata()


def test_normal_batch_trains_supplied_grammar_lessons(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from test_compiled_word_chunk import _tiny_canonical_model
    curriculum = json.loads((Path(__file__).resolve().parents[1] / "data/grammar_wording.json").read_text())
    chosen = ("wheels belong to bicycles", "bicycles are equal to wheels")
    lessons = [next(row for row in curriculum["train"] if row["text"] == text) for text in chosen]
    torch.manual_seed(942)
    model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets="8")
    model._tensor_peer_while_eager = True
    model._chart_compose_per_word = lambda: None
    data = model.inputSpace.data
    data.grammar_lessons = {"train": lessons}
    data.has_supervised_outputs = False
    chooser = model.languageSpace._tree_layer(2).chooser
    generator = model.languageSpace.generate_policy
    surface = next(op.gl for op in model.languageSpace._tree_layer(2).ops
                   if getattr(getattr(op, "gl", op), "rule_name", None) == "surface")
    owners = (chooser, generator, surface.marker_map, surface.marker_prior)
    before = [[p.detach().clone() for p in owner.parameters()] for owner in owners]
    optimizer = model.getOptimizer(lr=.003)
    try:
        inputs = model.inputSpace.prepInput(list(chosen))
        result, _ = model.runBatch(train=True, batchNum=0, batchSize=2,
            split="train", optimizer=optimizer, source_rows=[0, 1],
            batch_override=(inputs, torch.empty(2, 0)))
        assert result is not None
        for owner, original in zip(owners, before):
            assert any(not torch.equal(old, new) for old, new in zip(original, owner.parameters()))
        understanding = model._last_understanding
        assert understanding.input_reconstruction is not None
        assert model._grammar_lesson_loss(understanding, split="test", source_rows=[0, 1]) is None
        with pytest.raises(ValueError, match="does not match"):
            model._grammar_lesson_loss(understanding, split="train", source_rows=[1, 0])
        data.grammar_lessons = {}
        assert model._grammar_lesson_loss(understanding, split="train", source_rows=[0, 1]) is None
    finally:
        data.grammar_lessons = {}
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()
