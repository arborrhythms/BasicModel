"""Supervised training on the ``reverseOutput()`` path (Alec 2026-09-09:
"this will need supervised training on the output() path; do we have
tests that provide that?").

On the parallel XOR topology the answer path memorizes supervised labels
(the Step-7 causal test already trained it).  On the SERIAL grammar
topology (``MM_phrase_decode``: symbolicOrder 1, complete.grammar, word
analysis, 1024 wide) it did not until the answer path was seeded from
the grammar's ROOT IDEA (``Understanding.answer_seed``): the ``symbols``
tensor there is the symbol-space activation over a codebook with two
active rows at initialization and is the same for every sentence
(relative spread ~0.03 against ~0.5 for the root idea).  With the seed
the path memorizes eight labels at lr 1e-3 (5e-3 diverges on this
1024-wide topology)."""
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")

import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
_BIN = _ROOT / "bin"
_DATA = _ROOT / "data"
if str(_BIN) not in sys.path:
    sys.path.insert(0, str(_BIN))

from What import What  # noqa: E402


def _build(config_path, dataset):
    import Language
    from util import init_config
    from data import TheData
    import Models

    init_config(path=str(config_path), defaults_path=str(_DATA / "model.xml"))
    Language.TheGrammar._configured = False
    TheData.load(dataset)
    torch.manual_seed(0)
    m, _ = Models.BaseModel.from_config(str(config_path), data=TheData)
    return m.to("cpu")


def _spread(t):
    n = t.shape[0]
    t = t.reshape(n, -1)
    d = torch.cdist(t, t)
    return float(d[~torch.eye(n, dtype=torch.bool)].mean() / (t.norm(dim=-1).mean() + 1e-9))


def _train_accuracy(m):
    data = m.inputSpace.data
    n = data.what_extent("train")
    m.eval()
    with torch.no_grad():
        x = m.inputSpace.prepInput(list(data.train_input))
        y = m.outputSpace.prepOutput(list(data.train_output))
        answers = m.what(tuple(What.supervised(i) for i in range(n)), x)
        pred = torch.stack([a.what.reshape(-1) for a in answers])
        target = y.reshape(n, -1)
    m.train()
    if target.shape[-1] == 1:
        return float(((pred > 0.5).float() == target).float().mean())
    return float((pred.argmax(-1) == target.argmax(-1)).float().mean())


def _memorize(m, epochs, lr=5e-3, batch=8):
    opt = m.getOptimizer(lr=lr)
    best = 0.0
    for epoch in range(1, epochs + 1):
        m.train()
        m.runEpoch(optimizer=opt, batchSize=batch, split="train")
        if epoch % 10 == 0:
            best = max(best, _train_accuracy(m))
            if best >= 1.0:
                break
    return best


@pytest.fixture(scope="module")
def xor_synth_config(tmp_path_factory):
    src = (_DATA / "MM_xor.xml").read_text()
    src = src.replace("<architecture>", "<architecture>\n    <answerSynthesis>true</answerSynthesis>", 1)
    path = tmp_path_factory.mktemp("cfg") / "MM_xor_synth.xml"
    path.write_text(src)
    return path


@pytest.fixture(scope="module")
def serial_synth_config(tmp_path_factory):
    src = (_DATA / "MM_phrase_decode.xml").read_text()
    src = src.replace("<ideaDecode>true</ideaDecode>",
                      "<ideaDecode>true</ideaDecode>\n    <answerSynthesis>true</answerSynthesis>\n"
                      "    <stmReduceTau>0.05</stmReduceTau>\n    <transformChooser>mlp</transformChooser>")
    path = tmp_path_factory.mktemp("cfg") / "MM_phrase_synth.xml"
    path.write_text(src)
    return path


def test_output_path_memorizes_supervised_labels_on_the_parallel_topology(xor_synth_config):
    m = _build(xor_synth_config, "xor")
    assert m.answer_synthesis
    assert _memorize(m, epochs=60) >= 1.0


def test_serial_answer_path_varies_with_the_input(serial_synth_config):
    """The answer path can only be trained on what varies with the input:
    the symbolic root that seeds synthesis must differ across different
    sentences at least as much as the conceptual state does."""
    m = _build(serial_synth_config, "phrases")
    m.eval()
    data = m.inputSpace.data
    with torch.no_grad():
        x = m.inputSpace.prepInput(list(data.train_input))
        u = m.understand(x)
        c = m.reverseOutput(u, tuple(What.supervised(i) for i in range(len(data.train_input))))
    conceptual = _spread(u.conceptual_state)
    seed = _spread(u.answer_seed if torch.is_tensor(u.answer_seed) else u.symbolic_state)
    actual = _spread(c.actual)
    assert conceptual > 0.3, conceptual                    # the understanding varies
    assert seed > 0.3, seed                                # so does what seeds the answer
    assert actual > 0.1, actual                            # and the emitted answer


def test_output_path_memorizes_supervised_labels_on_the_serial_topology(serial_synth_config):
    m = _build(serial_synth_config, "phrases")
    assert torch.is_tensor(m.understand(m.inputSpace.prepInput(
        list(m.inputSpace.data.train_input[:2]))).answer_seed)
    assert _memorize(m, epochs=40, lr=1e-3) >= 1.0
