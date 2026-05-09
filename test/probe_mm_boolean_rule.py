"""Linguistic readout: what rule did the trained MM_boolean model learn?

Trains a model briefly, then probes:
  1. Per-literal C-tier concept activations (what each pure input
     produces just before the DNF wrapper).
  2. Per-AND-term firing strengths for each pure literal — identifies
     which AND-terms learned A, B, C, ¬A, ¬B, ¬C, non-A patterns.
  3. AND-term -> OR-fold weights — which terms participate in the
     final classification.
  4. Composite probes: combined inputs ("A B", "C", "not A", etc.)
     and final classification.

Run directly: python -m test.probe_mm_boolean_rule
"""

import os
import sys
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch
import Models
import Language
from util import init_config, TheXMLConfig

_CONFIG = os.path.join(_PROJECT, "data", "MM_boolean.xml")
_DEFAULTS = os.path.join(_PROJECT, "data", "model.xml")


def _train(seed=42, epochs=400):
    torch.manual_seed(seed)
    init_config(path=_CONFIG, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    cfg = TheXMLConfig.data
    arch = cfg.get("architecture", {})
    dat = arch.get("data", {})
    Models.TheData.load("inline", dat=dat)
    m, _ = Models.BasicModel.from_config(_CONFIG)
    n_train = len(Models.TheData.train_input)
    optimizer = torch.optim.Adam(m.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    loader = m.inputSpace.data.data_loader(split="train", num_streams=n_train)
    for _ in range(epochs):
        m.train()
        inp_items, out_items = next(iter(loader))
        ipt = m.inputSpace.prepInput(inp_items)
        tgt = m.outputSpace.prepOutput(out_items)
        if ipt is None or tgt is None:
            continue
        optimizer.zero_grad()
        _, _, output, _ = m.forward(ipt)
        t = tgt.to(output.device)
        while t.dim() < output.dim():
            t = t.unsqueeze(-1)
        t = t.expand_as(output)
        loss = criterion(output, t)
        loss.backward()
        optimizer.step()
    return m, Models.TheData


def _classify(m, sentence):
    """Run sentence through full pipeline and return scalar score."""
    ipt = m.inputSpace.prepInput([sentence])
    m.eval()
    with torch.no_grad():
        _, _, output, _ = m.forward(ipt)
    return float(output.detach().reshape(-1).mean().item())


def _dnf_layer(m):
    """Return the DNFConceptLayer wrapper from ConceptualSpace, if present."""
    cs = m.conceptualSpace
    layer = getattr(cs, "pi", None)
    if layer is None or not hasattr(layer, "neg"):
        return None
    return layer


def _readout_dnf_weights(m):
    """Print Sigma AND-fold and Pi OR-fold weight summaries."""
    dnf = _dnf_layer(m)
    if dnf is None:
        print("(no DNF wrapper found on conceptualSpace.pi)")
        return
    print(f"DNFConceptLayer: ternary={dnf.ternary} "
          f"and_fold={type(dnf.and_fold).__name__ if dnf.and_fold else None} "
          f"or_fold={type(dnf.or_fold).__name__ if dnf.or_fold else None}")
    if dnf.and_fold is not None:
        # Inner linear's weight: shape [nOutput, nInput]
        W_and = dnf.and_fold.layer.compute_W_current().detach().cpu()
        print(f"AND-fold W shape: {tuple(W_and.shape)}")
        # Each row = one AND-term combining literal channels
        print("  Top-magnitude AND-terms (rows of W):")
        row_norms = W_and.abs().sum(dim=1)
        top = torch.argsort(row_norms, descending=True)[:6].tolist()
        for k in top:
            row = W_and[k]
            top_lits = torch.argsort(row.abs(), descending=True)[:5].tolist()
            lits = [(int(j), float(row[j].item())) for j in top_lits]
            print(f"    term[{k:02d}] norm={row_norms[k]:.2f} "
                  f"top_literals={lits}")
    if dnf.or_fold is not None:
        W_or = dnf.or_fold.layer.compute_W_current().detach().cpu()
        print(f"OR-fold W shape: {tuple(W_or.shape)}")
        row_norms = W_or.abs().sum(dim=1)
        top = torch.argsort(row_norms, descending=True)[:6].tolist()
        print("  Top-magnitude OR-terms (rows of W):")
        for k in top:
            row = W_or[k]
            top_terms = torch.argsort(row.abs(), descending=True)[:4].tolist()
            tinfo = [(int(j), float(row[j].item())) for j in top_terms]
            print(f"    out[{k:02d}] norm={row_norms[k]:.2f} "
                  f"top_terms={tinfo}")


def _probe_literals(m):
    """Classify each pure percept and report final score.

    Under the privation/shamatha architecture, 'not A' tokenizes to
    a positive percept 'abs_A' and 'non-A' to 'non_A'. The probes
    below cover all single-percept inputs across the three surface
    forms — there is no negation transform on percepts, so each is
    a distinct codebook entry.
    """
    probes = [
        "A", "B", "C", "0",
        "not A", "not B", "not C",      # tokenize to abs_A, abs_B, abs_C
        "non-A", "non-B", "non-C",      # tokenize to non_A, non_B, non_C
    ]
    print("\nPure-percept classifications:")
    for s in probes:
        score = _classify(m, s)
        print(f"  {s:>10s} -> {score:+.3f}")


def _probe_pairs(m):
    """Classify combinations of two literals."""
    probes = [
        ("A B",       1.0),  # A∧B
        ("B A",       1.0),
        ("A 0 B",     1.0),
        ("A C",       0.0),
        ("B C",       1.0),  # ¬A (A absent) ∧ C
        ("not A C",   1.0),
        ("not A B",   0.0),
        ("non-A C",   0.0),  # non-A doesn't satisfy ¬A
        ("non-B C",   1.0),  # A absent => ¬A; C present
        ("0 C",       1.0),
        ("0 0",       0.0),
        ("not C A B", 1.0),
        ("not B A",   0.0),
    ]
    print("\nPair / sentence classifications:")
    for s, expected in probes:
        score = _classify(m, s)
        flag = " " if (score >= 0.5) == (expected >= 0.5) else "x"
        print(f"  {flag} {s:>14s} (target={expected:.0f}) -> {score:+.3f}")


def _probe_test_sentences(m):
    print("\nHeld-out test sentences (formula-correct labels):")
    for s, expected in (("A B", 1.0), ("C", 1.0), ("not A", 0.0)):
        score = _classify(m, s)
        flag = " " if (score >= 0.5) == (expected >= 0.5) else "x"
        print(f"  {flag} {s:>10s} (target={expected:.0f}) -> {score:+.3f}")


def _summarize_rule(m):
    """Read off the rule the network appears to be implementing.

    Heuristic: a literal that drives high output is asserted; a literal
    that drives low output is denied. The two AND-terms with strongest
    OR-fold contribution define the disjunction structure.
    """
    print("\nLinguistic readout — what rule is the network learning?")
    # Score each pure literal so we know what each input asserts.
    pure = {s: _classify(m, s) for s in
            ["A", "B", "C", "not A", "not B", "not C",
             "non-A", "non-B", "non-C"]}
    target = "(A and B) or ((not A) and C)"
    print(f"  Target rule:   {target}")
    print(f"  Pure-literal output scores: {pure}")
    print("  A pair (X, Y) likely participates in an AND-term iff "
          "score(X Y) >> max(score(X), score(Y)).")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, data = _train(seed=42, epochs=400)
        print("=" * 70)
        _readout_dnf_weights(m)
        _probe_literals(m)
        _probe_pairs(m)
        _probe_test_sentences(m)
        _summarize_rule(m)
