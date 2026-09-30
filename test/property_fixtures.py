"""Small analysis owner with the same taught primitive properties as WholeSpace."""
from types import SimpleNamespace

import torch

from Layers import char_class_region
from PerceptProperties import PrimitiveProperties
from Spaces import _CANONICAL_PROPERTY_ROWS


def property_reader(*, analysis_mode="word", **attributes):
    definitions = PrimitiveProperties(len(_CANONICAL_PROPERTY_ROWS))
    byte_ids = torch.arange(256)
    for row, (_name, kind) in enumerate(_CANONICAL_PROPERTY_ROWS):
        examples = (char_class_region(byte_ids, [kind]) > 0).float()
        definitions.teach(row, byte_ids, examples)
    return SimpleNamespace(
        analysis_mode=analysis_mode,
        subspace=SimpleNamespace(what=SimpleNamespace(primitive_properties=definitions)),
        well_known_atoms={name: row for row, (name, _kind) in enumerate(_CANONICAL_PROPERTY_ROWS)},
        **attributes)
