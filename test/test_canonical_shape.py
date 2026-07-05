import math, os, sys, unittest
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))

from architecture import canonical_shape, MANDATORY_CODEBOOK_SPACE_ROLES


def test_canonical_shape_table():
    # 2026-06-06 dim-convention unification (see bin/architecture.py docstring):
    # every INTERIOR space_role carries the SAME band so the formula
    # nDim = nWhat + nWhere + nWhen is uniform. SS/SymbolSpace band slots
    # ride along as inert padding. The ONLY principled (0,0) exception is
    # OutputSpace: the terminal answer has no .where/.when to mux.
    # 2026-07-04 encoding pass: the band widened (2, 2) -> (2, 4) -- .when is
    # the 4-dim start ladder (WhenStartDurationEncoding).
    assert canonical_shape("InputSpace")      == (2, 4)
    assert canonical_shape("PartSpace") == (2, 4)
    assert canonical_shape("ModalSpace")      == (2, 4)
    assert canonical_shape("ConceptualSpace") == (2, 4)
    assert canonical_shape("WholeSpace")   == (2, 4)
    assert canonical_shape("OutputSpace")     == (0, 0)
    assert canonical_shape("SymbolSpace")       == (2, 4)


def test_mandatory_codebook_space_roles():
    # 2026-06-04: the mandatory-codebook constraint was reverted. XOR_exact
    # (and other invertible-passthrough fixtures) need <codebook>none</codebook>
    # on PartSpace / WholeSpace -- a full-width invertible passthrough
    # with no VQ snap, so the forward<->reverse chain round-trips exactly and
    # the butterfly pi/sigma compute XOR with cross-slot reach. No space_role is
    # mandatory now; a config opts into a codebook explicitly via
    # <codebook>quantize</codebook>.
    assert MANDATORY_CODEBOOK_SPACE_ROLES == set()


def test_unknown_section_raises():
    try:
        canonical_shape("BogusSpace")
    except ValueError:
        return
    raise AssertionError("unknown section should raise")


if __name__ == "__main__":
    unittest.main()
