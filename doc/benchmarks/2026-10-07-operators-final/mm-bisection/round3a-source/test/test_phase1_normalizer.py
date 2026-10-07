"""Unit tests for Phase 1 Normalizer helper."""
import sys
from pathlib import Path

# basicmodel/bin must come before bin/ so that basicmodel's parse.py is found
# first (both directories contain a parse.py with different APIs).  Insert in
# reverse order so basicmodel/bin ends up at index 0.
_project = Path(__file__).resolve().parent.parent           # basicmodel/
_wo_root = _project.parent                                   # WikiOracle/
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))

import torch

from Models import Normalizer
from data import TheData


def test_normalizer_input_roundtrip():
    """normalize -> denormalize on 'input' is exact to float epsilon."""
    TheData.input_min = torch.tensor(-3.0)
    TheData.input_max = torch.tensor(5.0)
    TheData.output_min = torch.tensor(0.0)
    TheData.output_max = torch.tensor(1.0)
    n = Normalizer(TheData)
    x = torch.tensor([-3.0, 0.0, 5.0])
    y = n.normalize(x, which="input")
    x2 = n.denormalize(y, which="input")
    assert torch.allclose(x, x2, atol=1e-6), f"roundtrip failed: {x} vs {x2}"


def test_normalizer_matches_thedata():
    """Normalizer produces byte-identical output to TheData.normalize."""
    TheData.output_min = torch.tensor(-2.0)
    TheData.output_max = torch.tensor(4.0)
    n = Normalizer(TheData)
    x = torch.randn(3, 5)
    a = n.normalize(x, which="output")
    b = TheData.normalize(x, which="output")
    assert torch.equal(a, b)
    a = n.denormalize(x, which="output")
    b = TheData.denormalize(x, which="output")
    assert torch.equal(a, b)


def test_model_attaches_normalizer_to_every_space():
    """Every space on a constructed model has a non-None .normalizer."""
    import os
    from Models import BaseModel, Normalizer

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "basicmodel", "data", "MM_xor.xml",
    )
    # Load data first (same order as ModelFactory.run).
    TheData.load("xor")
    model, _ = BaseModel.from_config(config_path, data=TheData)

    # Collect every space from self.spaces (covers all spaces in both
    # BasicModel and BasicModel, including symbolSpace when present).
    spaces_to_check = list(enumerate(model.spaces))

    assert spaces_to_check, "model.spaces is empty — no spaces to check"

    for i, space in spaces_to_check:
        assert space.normalizer is not None, (
            f"spaces[{i}] ({type(space).__name__}).normalizer is None"
        )
        assert isinstance(space.normalizer, Normalizer), (
            f"spaces[{i}] ({type(space).__name__}).normalizer is "
            f"{type(space.normalizer).__name__}, not Normalizer"
        )


def test_no_thedata_reads_in_pipeline_methods():
    """Guard: Spaces.py must not reference TheData inside any method used by
    the forward/reverse pipeline. Construction-time references in __init__
    (embedding_source setup, where-encoding max computation) are allowed.
    """
    import re
    src = (_project / "bin" / "Spaces.py").read_text()

    pipeline_methods = [
        "_apply_normalization",
        "_apply_reverse_normalization",
        "denormalize",
    ]
    for name in pipeline_methods:
        pattern = rf"    def {re.escape(name)}\([^)]*\):.*?(?=\n    def |\nclass )"
        for match in re.finditer(pattern, src, flags=re.DOTALL):
            body = match.group(0)
            for bad in ("TheData.normalize", "TheData.denormalize"):
                assert bad not in body, (
                    f"{bad!r} leaked into pipeline method {name!r}:\n{body[:400]}"
                )

    for section_marker in ("def forward(self, vspace)", "def reverse(self, vspace)"):
        start = src.find("class OutputSpace(")
        assert start != -1, "OutputSpace class not found in Spaces.py"
        body_start = src.find(section_marker, start)
        if body_start == -1:
            continue
        body_end = src.find("\n    def ", body_start + len(section_marker))
        body = src[body_start:body_end]
        for bad in ("TheData.normalize", "TheData.denormalize"):
            assert bad not in body, (
                f"{bad!r} leaked into OutputSpace.{section_marker!r}:\n{body[:400]}"
            )
