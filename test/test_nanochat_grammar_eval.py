"""Unit gates for the frozen NanoChat-sized grammar evaluation harness."""

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import torch

import eval_nanochat_grammar as gate
from Spaces import ConceptualSpace


VOCAB = [
    "amber", "birch", "cedar", "daisy", "elm", "fern", "grove",
    "hazel", "iris", "juniper", "kelp", "lilac", "maple", "nettle",
    "olive", "pine", "quince", "reed", "sage", "thyme", "umber",
    "violet", "willow", "yarrow", "zinnia",
]


def _documents():
    docs = [f"training document {i}." for i in range(9)]
    sentences = []
    for i in range(40):
        rotated = VOCAB[i % len(VOCAB):] + VOCAB[:i % len(VOCAB)]
        sentences.append(" ".join(rotated[:12]) + ".")
    docs.append(" ".join(sentences))  # doc 9 is the test split
    return docs


def test_manifest_is_deterministic_document_heldout_and_frozen():
    kwargs = dict(
        num_items=8, seed=7, heldout_split="test",
        min_prefix_words=4, max_prefix_words=8,
        num_choices=4, max_completion_bytes=96,
        shard_names=["synthetic.parquet"], max_docs=10,
    )
    first = gate.build_manifest_from_documents(_documents(), **kwargs)
    second = gate.build_manifest_from_documents(_documents(), **kwargs)
    assert first == second
    assert gate.validate_manifest(first) is first
    assert len(first["items"]) == 8
    assert {item["doc_index"] for item in first["items"]} == {9}
    assert first["metadata"]["heldout_split"] == "test"
    assert first["metadata"]["item_sha256"] == gate._manifest_digest(
        first["items"])

    for item in first["items"]:
        assert len(item["candidates"]) == 4
        assert item["candidates"][item["answer_index"]] == item["target"]
        assert Counter(gate.surface_words(item["prefix"])) == Counter(
            gate.surface_words(item["shuffled_prefix"]))
        assert item["prefix"] != item["shuffled_prefix"]


def test_manifest_round_trip_and_digest_guard(tmp_path):
    manifest = gate.build_manifest_from_documents(
        _documents(), num_items=3, seed=11, num_choices=4,
        max_completion_bytes=96, max_docs=10)
    path = tmp_path / "manifest.json"
    gate.write_manifest(path, manifest)
    assert gate.load_manifest(path) == manifest

    broken = gate.load_manifest(path)
    broken["items"][0]["target"] = "tampered"
    path.write_text(__import__("json").dumps(broken), encoding="utf-8")
    try:
        gate.load_manifest(path)
    except ValueError as exc:
        assert "answer" in str(exc) or "sha256" in str(exc)
    else:
        raise AssertionError("tampered manifest should fail validation")


def test_word_distribution_score_uses_final_active_word_per_row():
    from Layers import SymbolExpectation
    layer=SymbolExpectation(n_symbols=2,max_depth=2,n_dim=2,concept_dim=2)
    layer.set_symbol_bank(torch.eye(layer.concept_dim))
    class PrefixHead(torch.nn.Module):
        def forward(self, keys, context, activation):
            mass=torch.einsum('bd,bnd->bn',context[:,2:],keys)
            return torch.stack((mass,torch.zeros_like(mass)),-1)
    layer._inter_predictor=PrefixHead()
    codes=torch.tensor([[[1.,0.],[1.,0.],[0.,0.]],
                        [[1.,0.],[0.,1.],[0.,0.]],
                        [[0.,0.],[0.,1.],[1.,0.]],
                        [[0.,0.],[0.,1.],[0.,1.]]])
    active=torch.tensor([[True,True,False],[True,True,False],[False,True,True],[False,True,True]])
    model=SimpleNamespace(symbolSpace=SimpleNamespace(expectation=layer),
        _word_expectation_input=(codes,None,None,None),
        _attention_words=SimpleNamespace(accepted=active),
        inputSpace=SimpleNamespace(_word_active_mask=active))
    scores,targets,predictions,counts=gate.word_candidate_scores(model,choices=2)
    mass=torch.tensor([[1.,0.],[0.,1.]]).sigmoid()
    expected=-(mass/mass.sum(-1,keepdim=True)).log().reshape(-1)
    torch.testing.assert_close(scores,expected)
    torch.testing.assert_close(targets,torch.eye(2).repeat(2,1))
    assert counts.tolist()==[2,2,2,2]
    assert torch.equal(predictions[0],predictions[1])
    assert torch.equal(predictions[2],predictions[3])
    assert not torch.equal(predictions[0],predictions[2])


def test_candidate_waves_preserve_the_complete_choice_bank(monkeypatch):
    from contextlib import contextmanager
    data = SimpleNamespace()
    sizes, offsets = [], []
    @contextmanager
    def runtime_batch(texts):
        data.train_input = texts
        sizes.append(len(texts))
        yield
    data.runtime_batch = runtime_batch
    model = SimpleNamespace(conceptualSpaces=[SimpleNamespace(stm=SimpleNamespace(_init_batch=2))],
        inputSpace=SimpleNamespace(prepInput=lambda texts: texts))
    def read(texts):
        counts = [len(text.split()) for text in texts]
        width = max(counts)
        active = torch.arange(width)[None] < torch.tensor(counts)[:, None]
        words = torch.zeros(len(texts), width, 1)
        for row, text in enumerate(texts):
            words[row, counts[row]-1, 0] = int(text.split()[-1])
        model._word_expectation_input = (words,)
        model.inputSpace._word_active_mask = active
        model._attention_words = SimpleNamespace(accepted=active)
    model._lex_embed_stem = read
    monkeypatch.setattr(gate, '_reset_model', lambda model: None)
    monkeypatch.setattr(gate, '_validate_candidate_reached', lambda *a, **k: offsets.extend(k['offsets']))
    def score(model, choices, captured):
        words, active, accepted = captured
        assert choices == 4 and words.shape == (8, 3, 1)
        assert torch.equal(active, accepted)
        final = active.sum(1)-1
        assert words[torch.arange(8), final, 0].tolist() == list(range(8))
        assert active.sum(1).tolist() == [2]*4 + [3]*4
        return 'complete bank'
    monkeypatch.setattr(gate, 'word_candidate_scores', score)
    items = [dict(prefix='one ', candidates=list(map(str, range(4)))),
             dict(prefix='one two ', candidates=list(map(str, range(4, 8))))]
    assert gate._score_control_batch(model, data, items, 'intact', 4) == 'complete bank'
    assert sizes == [2]*4 and offsets == [1]*4 + [2]*4


def test_word_gate_matches_full_forward_without_building_unused_answer(monkeypatch):
    from test_mm_xor import _fresh_model
    model, _, data = _fresh_model('data/XOR_grammar.xml')
    model.eval()
    model.set_sigma(0)
    item = dict(prefix='hello ', shuffled_prefix='hello ',
                candidates=['world', 'there'])
    texts = [item['prefix'] + candidate for candidate in item['candidates']]
    with gate.frozen_online_learning(model), torch.no_grad():
        gate._reset_model(model)
        with data.runtime_batch(texts):
            model.forward(model.inputSpace.prepInput(list(data.train_input)))
        gate._validate_candidate_reached(model, [item], 'intact', 2)
        reference = gate.word_candidate_scores(model, choices=2)
        gate._reset_model(model)
        def unused_answer(*args, **kwargs):
            raise AssertionError('word evaluation built the unused sentence/answer graph')
        monkeypatch.setattr(model, 'forward', unused_answer)
        actual = gate._score_control_batch(model, data, [item], 'intact', 2)
    for before, after in zip(reference, actual):
        torch.testing.assert_close(after, before, rtol=0, atol=0)




def test_frozen_control_scoring_disables_autograd(monkeypatch):
    from test_mm_xor import _fresh_model
    model, _, data = _fresh_model('data/XOR_grammar.xml')
    model.eval()
    owner = model.symbolSpace.expectation
    expect = owner.expect
    calls = []
    def inference(*args, **kwargs):
        calls.append(torch.is_grad_enabled())
        assert not torch.is_grad_enabled(), 'held-out scoring retained a backward graph'
        return expect(*args, **kwargs)
    monkeypatch.setattr(owner, 'expect', inference)
    item = dict(prefix='hello ', candidates=['world', 'there'])
    with gate.frozen_online_learning(model):
        gate._score_control_batch(model, data, [item], 'intact', 2)
    assert calls == [False, False]






def test_checkpoint_prewarm_builds_reducer_and_full_wholes_inventory():
    class FakeCodebook:
        nVectors = 8192

    codebook = FakeCodebook()
    reducer_calls = []
    to_calls = []
    model = SimpleNamespace(
        _stm_reducer=lambda: reducer_calls.append(True),
        wholeSpace=SimpleNamespace(
            nVectors=8192,
            subspace=SimpleNamespace(codebook=lambda: codebook)),
        to=lambda target: to_calls.append(target),
    )

    gate._prewarm_checkpoint_shapes(model, torch.device("cpu"))

    assert reducer_calls == [True]
    assert codebook.nVectors == 8192
    assert to_calls == [torch.device("cpu")]
